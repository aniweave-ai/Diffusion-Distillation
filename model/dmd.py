from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from model.base import SelfForcingModel
from pipeline.image_edit_training import ImageEditTrainingPipeline


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: ImageEditTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        # self.min_step = int(0.02 * self.num_train_timestep)
        # self.max_step = int(0.98 * self.num_train_timestep)
        self.min_step = 0
        self.max_step = self.num_train_timestep

        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _compute_kl_grad(
        self,
        noisy_img: torch.Tensor,  # Latent
        image_tensor: torch.Tensor,  # Latent
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Updated KL Grad: Operates entirely in Latent Space to skip VAE Decoding.
        """

        # Step 1: Compute the fake score (returns flow_pred_latents)
        self.switch_to_fake()

        v_fake_cond = self.fake_score.compute_score(
            noisy_img=noisy_img,
            image_tensor=image_tensor,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )

        if self.fake_guidance_scale != 0.0:
            v_fake_uncond = self.fake_score.compute_score(
                noisy_img=noisy_img,
                image_tensor=image_tensor,
                conditional_dict=unconditional_dict,
                timestep=timestep,
            )
            v_fake = (
                v_fake_cond + (v_fake_cond - v_fake_uncond) * self.fake_guidance_scale
            )
        else:
            v_fake = v_fake_cond

        # Step 2: Compute the real score (returns flow_pred_latents)
        self.switch_to_real()
        v_real_cond = self.real_score.compute_score(
            noisy_img=noisy_img,
            image_tensor=image_tensor,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )

        v_real_uncond = self.real_score.compute_score(
            noisy_img=noisy_img,
            image_tensor=image_tensor,
            conditional_dict=unconditional_dict,
            timestep=timestep,
        )

        v_real = v_real_cond + (v_real_cond - v_real_uncond) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (in Latent Velocity Space)
        # grad = v_fake - v_real
        grad = v_fake - v_real

        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            # We must convert the normalizer logic to Latent Space.
            # We encode the condition image to latents once to calculate the 'p_real'
            with torch.no_grad():
                # In Flow Matching, the prediction is velocity (v).
                # To match the paper's (target - pred) logic, we use the teacher's velocity
                # as the proxy for the distance to the real manifold.
                # Alternatively: use the mean magnitude of v_real directly.
                normalizer = torch.abs(v_real).mean(dim=[1, 2, 3], keepdim=True)
                grad = grad / (normalizer + 1e-6)

        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video
        batch_size = image_or_video.shape[0]
        num_frame = 1
        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = (
                denoised_timestep_to
                if self.ts_schedule and denoised_timestep_to is not None
                else self.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.ts_schedule_max and denoised_timestep_from is not None
                else self.num_train_timestep
            )
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
            )
            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = (
                    self.timestep_shift
                    * (timestep / 1000)
                    / (1 + (self.timestep_shift - 1) * (timestep / 1000))
                    * 1000
                )
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = (
                self.scheduler.add_noise(
                    image_or_video.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep.flatten(0, 1),
                )
                .detach()
                .unflatten(0, (batch_size, num_frame))
            )
            # the scheduler and _get_timestep is used for video, so we have to remove the frame dimension
            noisy_latent = noisy_latent.squeeze(1)
            timestep = timestep.squeeze(0)

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_img=noisy_latent,
                image_tensor=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(),
                (original_latent.double() - grad.double()).detach(),
                reduction="mean",
            )
        return dmd_loss, dmd_log_dict

    # TODO: apply BSMNTW
    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        # Calculate latent shape from pixel shape
        # image_or_video_shape is [B, C, H, W] (pixels)
        B, C, latent_H, latent_W = image_or_video_shape

        # Latent shape is [B, 1, 16, H, W] (B, F, C, H, W)
        latent_shape = [B, 1, 16, latent_H, latent_W]

        self.switch_to_generator()
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = (
            self._run_generator(
                image_or_video_shape=latent_shape,  # Pass latent shape
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
            )
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )

        del pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to

        return dmd_loss, dmd_log_dict

    # TODO: apply BSMNTW
    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict = None,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        # Calculate latent shape from pixel shape
        B, _, latent_H, latent_W = image_or_video_shape

        latent_shape = [B, 1, 16, latent_H, latent_W]

        with torch.no_grad():
            self.switch_to_generator()
            generated_image, _, denoised_timestep_from, denoised_timestep_to = (
                self._run_generator(
                    image_or_video_shape=latent_shape,  # Pass latent shape
                    conditional_dict=conditional_dict,
                    initial_latent=initial_latent,
                )
            )

        # Step 2: Compute the fake prediction
        min_timestep = (
            denoised_timestep_to
            if self.ts_schedule and denoised_timestep_to is not None
            else self.min_score_timestep
        )
        max_timestep = (
            denoised_timestep_from
            if self.ts_schedule_max and denoised_timestep_from is not None
            else self.num_train_timestep
        )
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            latent_shape[0],
            1,  # num_frame is 1 for image
        )

        if self.timestep_shift > 1:
            critic_timestep = (
                self.timestep_shift
                * (critic_timestep / 1000)
                / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000))
                * 1000
            )

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1),
        ).unflatten(0, latent_shape[:2])

        noisy_generated_image = noisy_generated_image.squeeze(1)
        critic_timestep = critic_timestep.squeeze(0)

        # pack image noise to latent:
        noisy_latents = self.qwen_image_edit_wrapper._pack_latents(
            noisy_generated_image, B, latent_H, latent_W
        )

        self.switch_to_fake()
        _, pred_fake_image = self.fake_score(
            noisy_latents=noisy_latents,  # Pass latents
            conditional_dict=conditional_dict,
            timestep=critic_timestep,
            image_latents=initial_latent,  # Pass condition latents
            unpack_output=True,
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            flow_pred = self.qwen_image_edit_wrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image,
                xt=noisy_generated_image,
                timestep=critic_timestep,
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image,
                xt=noisy_generated_image,
                timestep=critic_timestep,
            ).unflatten(0, latent_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image,
            x_pred=pred_fake_image,
            noise=critic_noise,
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep,
            flow_pred=flow_pred,
        )

        # Step 5: Debugging Log
        critic_log_dict = {"critic_timestep": critic_timestep.detach()}

        return denoising_loss, critic_log_dict
