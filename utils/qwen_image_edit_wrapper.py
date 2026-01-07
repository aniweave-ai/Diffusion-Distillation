import math
import types

import torch
from diffusers import QwenImageEditPlusPipeline

from utils.scheduler import FlowMatchScheduler, SchedulerInterface

REAL_LORA_NAME = "real"
FAKE_LORA_NAME = "fake"
GENERATOR_LORA_NAME = "generator"

VAE_IMAGE_SIZE = 1024 * 1024
CONDITION_IMAGE_SIZE = 384 * 384


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height


class QwenImageEditWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen-Image-Edit-2509",
        timestep_shift=3.0,
        teacher_lora_path=None,
        student_lora_path=None,
        critic_lora_path=None,
        rank=32,
        device="cuda",
    ):
        super().__init__()
        self.model_name = model_name

        # 1. Load the Pipeline
        # We load in bfloat16 to save memory, assuming Ampere+ GPU
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)

        # 2. Extract Components
        # Qwen uses a VAE scale factor (usually 16) + internal patch size (2)
        # We need this for shape calculations in forward()
        self.vae_scale_factor = self.pipe.vae_scale_factor

        # 3. Setup Scheduler
        self.scheduler = FlowMatchScheduler(shift=timestep_shift)
        self.scheduler.set_timesteps(1000, training=True)
        self.post_init()  # Bind helper methods to scheduler

        # 4. Setup LoRA / PEFT for DMD Student
        # Qwen specific modules

        if teacher_lora_path is not None:
            # Load the pre-trained weights (Teacher)
            print("Loading Teacher LoRA weights from:", teacher_lora_path)

            self.pipe.load_lora_weights(teacher_lora_path, adapter_name=REAL_LORA_NAME)
            if critic_lora_path is None:
                self.pipe.load_lora_weights(
                    teacher_lora_path, adapter_name=FAKE_LORA_NAME
                )

            if student_lora_path is None:
                self.pipe.load_lora_weights(
                    teacher_lora_path, adapter_name=GENERATOR_LORA_NAME
                )

        if student_lora_path is not None:
            # Load the pre-trained weights (Student)
            print("Loading Student LoRA weights from:", student_lora_path)

            self.pipe.load_lora_weights(
                student_lora_path, adapter_name=GENERATOR_LORA_NAME
            )

        if critic_lora_path is not None:
            # Load the pre-trained weights (Critic)
            print("Loading Critic LoRA weights from:", critic_lora_path)
            self.pipe.load_lora_weights(critic_lora_path, adapter_name=FAKE_LORA_NAME)

        if (
            student_lora_path is None
            and critic_lora_path is None
            and teacher_lora_path is None
        ):
            print("No LoRA weights loaded. Using base model.")

        # Expose transformer directly for easy access
        self.transformer = self.pipe.transformer

    def freeze_all(self):
        print("freez all model parameters")
        self.transformer.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)

    def enable_adapters_for_training(self):
        """
        NEW: Enables gradients for Generator and Critic LoRAs ONCE.
        Call this method immediately after initialization in your BaseModel.
        """
        print("Enabling gradients for Generator and Critic LoRA adapters...")

        trainable_count = 0
        for name, param in self.transformer.named_parameters():
            # Check if parameter belongs to Generator OR Fake (Critic) LoRA
            if GENERATOR_LORA_NAME in name or FAKE_LORA_NAME in name:
                param.requires_grad = True
                trainable_count += param.numel()

        print(f"Total trainable parameters enabled: {trainable_count}")

    def set_adapter_trainable(self, adapter_name, freeze_others=True):
        """
        Activates the specific adapter and sets ONLY its parameters to requires_grad=True.
        All other adapters and base model weights are frozen if freeze_others is True.
        """
        print(
            f"Setting active adapter to '{adapter_name}' and unfreezing specific weights..."
        )

        # 1. Switch the active adapter in PEFT
        self.transformer.set_adapter(adapter_name)

        # 2. Unfreeze only the specific adapter's LoRA weights
        # First, ensure everything is frozen if requested
        if freeze_others:
            self.transformer.requires_grad_(False)

        trainable_params = 0
        for name, param in self.pipe.transformer.named_parameters():
            # PEFT naming convention usually includes the adapter name at the end
            # e.g., "transformer.layers.0.attn.q_proj.lora_A.generator"
            if "lora" in name and adapter_name in name:
                param.requires_grad = True
                trainable_params += param.numel()

        print(f"Trainable params for {adapter_name}: {trainable_params}")

    def get_adapter_parameters(self, adapter_name):
        """
        Returns a list of parameters belonging specifically to the requested adapter.
        Used to initialize Optimizers.
        """
        specific_params = []
        for name, param in self.model.named_parameters():
            if "lora" in name and adapter_name in name:
                specific_params.append(param)
        return specific_params

    def switch_to_generator(self):
        """Activates the Student LoRA (Generator)."""
        self.transformer.set_adapter(GENERATOR_LORA_NAME)

        # prevent fake and genertor grad being false during training, that will cause grad missing
        for name, param in self.transformer.named_parameters():
            if GENERATOR_LORA_NAME in name or FAKE_LORA_NAME in name:
                param.requires_grad = True

    def switch_to_real(self):
        """Activates the Teacher LoRA (Real Score)."""
        self.transformer.set_adapter(REAL_LORA_NAME)

        # prevent fake and genertor grad being false during training, that will cause grad missing
        for name, param in self.transformer.named_parameters():
            if GENERATOR_LORA_NAME in name or FAKE_LORA_NAME in name:
                param.requires_grad = True

    def switch_to_fake(self):
        """Activates the Student LoRA (Fake Score)."""
        self.transformer.set_adapter(FAKE_LORA_NAME)

        # prevent fake and genertor grad being false during training, that will cause grad missing
        for name, param in self.transformer.named_parameters():
            if GENERATOR_LORA_NAME in name or FAKE_LORA_NAME in name:
                param.requires_grad = True

    def enable_gradient_checkpointing(self) -> None:
        self.transformer.enable_gradient_checkpointing()

    def encode_vae_img(self, img: torch.Tensor) -> torch.Tensor:
        latents_dist = self.pipe.vae.encode(img).latent_dist

        # TODO: add a seed to determine the same image
        clean_latents_2d = latents_dist.sample()

        latents_mean = (
            torch.tensor(self.pipe.vae.config.latents_mean)
            .view(1, self.pipe.latent_channels, 1, 1, 1)
            .to(clean_latents_2d.device, clean_latents_2d.dtype)
        )
        latents_std = (
            torch.tensor(self.pipe.vae.config.latents_std)
            .view(1, self.pipe.latent_channels, 1, 1, 1)
            .to(clean_latents_2d.device, clean_latents_2d.dtype)
        )
        image_latents = (clean_latents_2d - latents_mean) / latents_std

        return image_latents

    def decode_vae_latent(self, vae_latent) -> torch.Tensor:
        latents_mean = (
            torch.tensor(self.pipe.vae.config.latents_mean)
            .view(1, self.pipe.latent_channels, 1, 1, 1)
            .to(vae_latent.device, vae_latent.dtype)
        )
        latents_std = (
            torch.tensor(self.pipe.vae.config.latents_std)
            .view(1, self.pipe.latent_channels, 1, 1, 1)
            .to(vae_latent.device, vae_latent.dtype)
        )
        latents_denorm = vae_latent * latents_std + latents_mean
        video_output = self.pipe.vae.decode(latents_denorm, return_dict=False)[0]
        image_tensor = video_output[:, :, 0]  # Remove frame dim

        return image_tensor

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction (velocity) to x0 prediction.
        Formula: x0 = xt - sigma_t * velocity
        """
        # Use float64 for precision during inversion
        original_dtype = flow_pred.dtype
        flow_pred = flow_pred.double()
        xt = xt.double()

        # Retrieve sigmas/timesteps from scheduler
        sigmas = self.scheduler.sigmas.to(device=xt.device, dtype=torch.double)
        sched_timesteps = self.scheduler.timesteps.to(
            device=xt.device, dtype=torch.double
        )

        # Handle broadcasting of timestep
        # timestep: [B] -> [B, 1, 1]
        timestep_f = timestep.flatten().to(torch.double)

        # Map input timestep to sigma
        # This assumes input timestep matches one of the scheduler steps (training=True usually implies 0-1000)
        # Using searchsorted or min diff is safer than direct indexing
        sigma_t_list = []
        for t_val in timestep_f:
            idx = (torch.abs(sched_timesteps - t_val)).argmin()
            sigma_t_list.append(sigmas[idx])

        sigma_t = torch.tensor(sigma_t_list, device=xt.device, dtype=torch.double)

        # Reshape for broadcasting against [B, L, C] or [B, C, H, W]
        if xt.dim() == 3:  # [B, L, C] - Transformers
            sigma_t = sigma_t.view(-1, 1, 1)
        elif xt.dim() == 4:  # [B, C, H, W] - CNNs
            sigma_t = sigma_t.view(-1, 1, 1, 1)

        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def _pack_latents(
        self,
        latents: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Helper to pack spatial latents [B, C, H, W] into tokens [B, Seq, C].
        Handles 5D inputs [B, F, C, H, W] or [B, C, F, H, W] by squeezing the frame dimension.
        """
        if latents.dim() == 5:
            if latents.shape[1] == 1:
                latents = latents.squeeze(1)
            elif latents.shape[2] == 1:
                latents = latents.squeeze(2)

        packed = self.pipe._pack_latents(
            latents,
            batch_size,
            self.pipe.latent_channels,
            height,
            width,
        )

        # Ensure it's 3D [B, Seq, C]
        if packed.dim() == 4 and packed.shape[1] == 1:
            packed = packed.squeeze(1)

        return packed

    def _unpack_latents(
        self,
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Helper to unpack tokens [B, Seq, C] back to spatial latents [B, C, H, W].
        height, width are pixel level
        """
        latents_2d: torch.Tensor = self.pipe._unpack_latents(
            tokens,  # These are the final denoised latents
            height=height,
            width=width,
            vae_scale_factor=self.pipe.vae_scale_factor,
        )
        latents_2d = latents_2d.squeeze(2)  # remove frame dimesion
        return latents_2d

    def compute_score(
        self,
        noisy_img: torch.Tensor,  # Input: Latents [B, 16, H, W]
        conditional_dict: dict,
        timestep: torch.Tensor,
        image_tensor: torch.Tensor,  # Input: Condition Latents [B, 16, H, W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DMD Forward Pass for Pixel-to-Pixel Gradient calculation.
        """

        batch_sz = noisy_img.shape[0]
        prompt_embeds = conditional_dict.get("prompt_embeds")
        prompt_embeds_mask = conditional_dict.get("prompt_embeds_mask")

        # 1. Normalize Timestep
        transformer_timestep = (timestep / 1000.0).to(dtype=self.pipe.transformer.dtype)

        # 2. Encode Pixels to Latents (if needed)
        # We must work in Latent Space for the Transformer

        noisy_img_latents = noisy_img
        condition_img_latents = image_tensor

        # If input is latents, we need to infer the original pixel height/width for shape calc
        # Latent factor is 16.
        # Shape is [B, 16, H, W]
        image_latent_height, image_latent_width = noisy_img_latents.shape[-2:]

        # 3. Convert 2D Grid [B, C, H, W] -> Sequence [B, Seq, C] for Transformer
        noisy_tokens = self._pack_latents(
            noisy_img_latents,
            batch_sz,
            image_latent_height,
            image_latent_width,
        )
        condition_tokens = self._pack_latents(
            condition_img_latents,
            batch_sz,
            image_latent_height,
            image_latent_width,
        )

        # 4. Concatenate on Sequence Dimension (Tokens, not Channels)
        # Qwen-Image-Edit concatenates the image sequences
        latent_model_input = torch.cat([noisy_tokens, condition_tokens], dim=1)

        # 5. Prepare Image Shapes (32x32 pixel patches = 2x2 latent patches)
        # Latent factor is 16, Qwen patch factor is 2. Total 32.
        h_token = image_latent_height // 2
        w_token = image_latent_width // 2

        target_shape = (1, h_token, w_token)
        shapes_per_sample = [target_shape, target_shape]
        img_shapes = [shapes_per_sample] * batch_sz

        # 6. Text Masking
        txt_seq_lens = None
        if prompt_embeds_mask is not None:
            txt_seq_lens = prompt_embeds_mask.int().sum(dim=1).tolist()

        # 7. Model Forward Pass
        model_output = self.transformer(
            hidden_states=latent_model_input,
            timestep=transformer_timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        # Slice only the target prediction (the first sequence)
        num_target_tokens = noisy_tokens.size(1)
        flow_pred_tokens = model_output[:, :num_target_tokens]

        pixel_height = image_latent_height * self.pipe.vae_scale_factor
        pixel_width = image_latent_width * self.pipe.vae_scale_factor

        # Unflatten back to 4D Latents [B, 4, H, W]
        flow_pred_latents = self._unpack_latents(
            flow_pred_tokens,
            pixel_height,
            pixel_width,
        )
        return flow_pred_latents

    def forward(
        self,
        noisy_latents: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        guidance_scale: float | None = None,
        height: int = 1024,
        width: int = 1024,
        image_latents: torch.Tensor | None = None,  # The original image (Condition)
        unpack_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        DMD Forward Pass.

        Args:
            noisy_latents: [B, Seq_Len, C] - The noisy target image.
            conditional_dict: Contains 'prompt_embeds', masks, etc.
            timestep: [B] - Current timestep.
            image_latents: [B, Seq_Len, C] - The reference/condition image (clean).
                           Required for Image Edit tasks.

        Returns:
            flow_pred: The model output (velocity).
            pred_x0: The predicted clean latent (used for DMD loss).
        """
        prompt_embeds = conditional_dict.get("prompt_embeds")
        prompt_embeds_mask = conditional_dict.get("prompt_embeds_mask")

        # 1. Normalize Timestep
        # Qwen/DiT models typically expect t in [0, 1] derived from t/1000
        transformer_timestep = timestep / 1000.0
        transformer_timestep = transformer_timestep.to(dtype=noisy_latents.dtype)

        # 2. Prepare Guidance (CFG/Distilled Guidance)
        # DMD usually operates at guidance=1.0 during training unless distilling specific CFG
        if self.transformer.config.guidance_embeds:
            if guidance_scale is None:
                guidance_scale = 1.0
            guidance = torch.full(
                [1], guidance_scale, device=noisy_latents.device, dtype=torch.float32
            )
            guidance = guidance.expand(noisy_latents.shape[0])
        else:
            guidance = None

        # 3. Concatenate Latents for Image Editing
        # Structure: [Noisy_Target, Clean_Condition]
        # Ensure noisy_latents is 3D [B, Seq, C]
        if noisy_latents.dim() == 4 and noisy_latents.shape[1] == 1:
            noisy_latents = noisy_latents.squeeze(1)

        latent_model_input = noisy_latents
        if image_latents is not None:
            # Check if image_latents is spatial [B, F, C, H, W] and pack if needed
            if image_latents.dim() == 5 or image_latents.dim() == 4:
                # We need H, W for packing.
                # If 5D: [B, F, C, H, W] -> H=shape[3], W=shape[4]
                # If 4D: [B, C, H, W] -> H=shape[2], W=shape[3]
                if image_latents.dim() == 5:
                    l_h, l_w = image_latents.shape[3], image_latents.shape[4]
                else:
                    l_h, l_w = image_latents.shape[2], image_latents.shape[3]

                image_latents = self._pack_latents(
                    image_latents,
                    image_latents.shape[0],
                    l_h,
                    l_w,
                )

            # Ensure image_latents is 3D [B, Seq, C]
            if image_latents.dim() == 4 and image_latents.shape[1] == 1:
                image_latents = image_latents.squeeze(1)

            latent_model_input = torch.cat([noisy_latents, image_latents], dim=1)

        # 4. Prepare Image Shapes for RoPE
        # This is CRITICAL for Qwen. It acts on 1D sequences but needs to know the 2D grid size.
        # effective scale = vae_scale (16) * patch_merge (2) = 32
        # For 1024px -> 32x32 tokens

        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        h_token = latent_height // 2
        w_token = latent_width // 2

        batch_size = noisy_latents.shape[0]

        # Shape of the target image
        target_shape = (1, h_token, w_token)

        # Construct the shape list.
        # If we have conditioning, Qwen expects a list of shapes corresponding to the concat order.
        if image_latents is not None:
            # We assume condition image has same resolution as target for editing
            shapes_per_sample = [target_shape, target_shape]
        else:
            shapes_per_sample = [target_shape]

        img_shapes = [shapes_per_sample] * batch_size

        # 5. Prepare Text Lengths
        txt_seq_lens = None
        if prompt_embeds_mask is not None:
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).int().tolist()

        # 6. Model Forward Pass
        # We assume self.transformer is the PEFT model

        model_output = self.transformer(
            hidden_states=latent_model_input,
            timestep=transformer_timestep,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        # 7. Post-Process Output
        # The output contains predictions for [Target + Condition].
        # We slice to keep only the Target prediction.
        # noisy_latents.size(1) is the sequence length of the target.
        flow_pred = model_output[:, : noisy_latents.size(1)]

        # 8. Convert to x0 (Clean Prediction)
        # This is needed for DMD Loss: L_dmd = E[ D_fake(pred_x0) - D_real(pred_x0) ]
        pred_x0 = self._convert_flow_pred_to_x0(flow_pred, noisy_latents, timestep)

        if unpack_output:
            # Unflatten back to 4D Latents [B, 4, H/16, W/16]
            flow_pred = self._unpack_latents(
                flow_pred,
                height,
                width,
            )

            pred_x0 = self._unpack_latents(
                pred_x0,
                height,
                width,
            )

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        # Bind methods for utility usage only if they don't exist
        if not hasattr(scheduler, "convert_x0_to_noise"):
            scheduler.convert_x0_to_noise = types.MethodType(
                SchedulerInterface.convert_x0_to_noise, scheduler
            )
        if not hasattr(scheduler, "convert_noise_to_x0"):
            scheduler.convert_noise_to_x0 = types.MethodType(
                SchedulerInterface.convert_noise_to_x0, scheduler
            )
        if not hasattr(scheduler, "convert_velocity_to_x0"):
            scheduler.convert_velocity_to_x0 = types.MethodType(
                SchedulerInterface.convert_velocity_to_x0, scheduler
            )
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()

    def clip_grad_norm_(self, max_norm):
        """
        Mimics the FSDP/Lightning clip_grad_norm_ interface
        for a standard PyTorch module.
        """
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.nn.utils.clip_grad_norm_(trainable_params, max_norm)
