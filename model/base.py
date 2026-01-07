from typing import Tuple

import torch
from torch import nn

from pipeline.image_edit_training import ImageEditTrainingPipeline
from utils.loss import get_denoising_loss
from utils.qwen_image_edit_wrapper import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    QwenImageEditWrapper,
    calculate_dimensions,
)


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(
                args.denoising_step_list, dtype=torch.long
            )
            if args.warp_denoising_step:
                timesteps = torch.cat(
                    (
                        self.scheduler.timesteps.cpu(),
                        torch.tensor([0], dtype=torch.float32),
                    )
                )
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        teacher_lora_path = args.teacher_lora_path
        student_lora_path = args.student_lora_path
        critic_lora_path = args.critic_lora_path
        model_name = args.model_name

        self.qwen_image_edit_wrapper = QwenImageEditWrapper(
            model_name=model_name,
            teacher_lora_path=teacher_lora_path,
            student_lora_path=student_lora_path,
            critic_lora_path=critic_lora_path,
            device=device,
        )
        self.qwen_image_edit_wrapper.freeze_all()

        self.generator = self.qwen_image_edit_wrapper
        self.real_score = self.qwen_image_edit_wrapper
        self.fake_score = self.qwen_image_edit_wrapper

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        self.text_encoder = self.qwen_image_edit_wrapper.pipe.text_encoder

    def _get_wrapper(self):
        """Helper to get the unwrapped QwenImageEditWrapper."""
        if hasattr(self.qwen_image_edit_wrapper, "module"):
            return self.qwen_image_edit_wrapper.module
        return self.qwen_image_edit_wrapper

    def switch_to_generator(self):
        self._get_wrapper().switch_to_generator()

    def switch_to_real(self):
        self._get_wrapper().switch_to_real()

    def switch_to_fake(self):
        self._get_wrapper().switch_to_fake()

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int = 1,
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        """
        timestep = torch.randint(
            min_timestep,
            max_timestep,
            [batch_size, 1],
            device=self.device,
            dtype=torch.long,
        ).repeat(1, num_frame)
        return timestep

    def run_vae_encoder(self, image: torch.Tensor) -> torch.Tensor:
        image_width, image_height = image.shape[2], image.shape[3]
        aspect_ratio = image_width / image_height

        vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, aspect_ratio)
        vae_image = (
            self.real_score.pipe.image_processor.preprocess(
                image, vae_height, vae_width
            )
            .to(device=self.device, dtype=self.dtype)
            .unsqueeze(2)
        )
        image_latent = self.qwen_image_edit_wrapper.encode_vae_img(vae_image)

        return image_latent

    def encode_prompt(self, image: torch.Tensor, prompt: str) -> dict:
        image_width, image_height = image.shape[2], image.shape[3]
        aspect_ratio = image_width / image_height

        condition_width, condition_height = calculate_dimensions(
            CONDITION_IMAGE_SIZE, aspect_ratio
        )
        condition_image = self.qwen_image_edit_wrapper.pipe.image_processor.preprocess(
            image, condition_height, condition_width
        )

        prompt_embeds, prompt_masks = self.qwen_image_edit_wrapper.pipe.encode_prompt(
            prompt=prompt,
            image=condition_image,
            device=self.device,
        )

        cond_dict = {
            "prompt_embeds": prompt_embeds.to(self.dtype),
            "prompt_embeds_mask": prompt_masks.to(self.dtype),
        }

        return cond_dict


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Simulate the generator's input from noise using backward simulation (Trajectory Replay).
        Adapted for Image Editing.

        Input:
            - image_or_video_shape: [B, C, H, W] or [B, 1, C, H, W].
            - initial_latent: The Source Image Latent (the image we want to edit).
            - conditional_dict: Contains text embeddings, masks, etc.
        Output:
            - pred_image: The predicted x0 at the exit step [B, C, H, W].
            - gradient_mask: None (Apply loss to full image).
            - denoised_timestep_from: The step where simulation stopped.
            - denoised_timestep_to: The next step (usually 0 if final).
        """

        # Step 2: Prepare Conditioning (Source Image)
        # In Image Editing, 'initial_latent' is the source image we want to edit.
        # We map it to 'source_image_latent' which ImageEditTrainingPipeline expects.
        if initial_latent is not None:
            # Handle 5D input from video dataloaders [B, F, C, H, W] or [B, C, F, H, W] -> [B, C, H, W]
            if initial_latent.dim() == 5:
                if initial_latent.shape[1] == 1:
                    initial_latent = initial_latent.squeeze(1)
                elif initial_latent.shape[2] == 1:
                    initial_latent = initial_latent.squeeze(2)

            conditional_dict["source_image_latent"] = initial_latent

        # Step 3: Prepare Noise
        # Normalize shape to [B, C, H, W]
        noise_shape = list(image_or_video_shape)

        if len(noise_shape) == 5:
            # Drop frame dimension if present.
            # We handle both [B, F, C, H, W] and [B, C, F, H, W]
            if noise_shape[1] == 1:
                # [B, F, C, H, W] -> [B, C, H, W]
                noise_shape = [
                    noise_shape[0],
                    noise_shape[2],
                    noise_shape[3],
                    noise_shape[4],
                ]
            elif noise_shape[2] == 1:
                # [B, C, F, H, W] -> [B, C, H, W]
                noise_shape = [
                    noise_shape[0],
                    noise_shape[1],
                    noise_shape[3],
                    noise_shape[4],
                ]
            else:
                # Default to dropping index 1 if neither is 1 (e.g. video)
                noise_shape = [
                    noise_shape[0],
                    noise_shape[2],
                    noise_shape[3],
                    noise_shape[4],
                ]

        noise = torch.randn(noise_shape, device=self.device, dtype=self.dtype)
        # Step 4: Run Backward Simulation
        # This calls ImageEditTrainingPipeline.inference_with_trajectory
        pred_image, denoised_timestep_from, denoised_timestep_to = (
            self._consistency_backward_simulation(
                noise=noise,
                conditional_dict=conditional_dict,
            )
        )

        # Step 5: Handle Outputs
        # pred_image should be [B, C, H, W] (or [B, 1, C, H, W] depending on wrapper return)
        # Ensure it is 4D for consistency with standard image losses
        if pred_image.dim() == 5:
            if pred_image.shape[1] == 1:
                pred_image = pred_image.squeeze(1)
            elif pred_image.shape[2] == 1:
                pred_image = pred_image.squeeze(2)

        # Gradient Mask:
        # For video, this masked out history frames. For single image editing,
        # we usually want to train on the entire image.
        gradient_mask = None

        return (
            pred_image,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
        )

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            conditional_dict=conditional_dict,
            source_image_latent=conditional_dict.get("source_image_latent"),
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """

        self.inference_pipeline = ImageEditTrainingPipeline(
            model_name=self.args.model_name,
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
        )
