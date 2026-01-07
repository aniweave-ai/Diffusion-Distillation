from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from utils.qwen_image_edit_wrapper import QwenImageEditWrapper
from utils.scheduler import SchedulerInterface


class ImageEditTrainingPipeline:
    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: QwenImageEditWrapper,
    ):
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list

        # Ensure step 0 is removed for inference trajectory to avoid numerical instability at t=0
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

    def _get_random_exit_step(self, num_denoising_steps, device):
        """
        Randomly select a timestep to stop the backward simulation.
        Syncs across ranks to ensure all GPUs stop at the same step for the batch.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            # Pick one random index for the whole batch
            exit_index = torch.randint(
                low=0, high=num_denoising_steps, size=(1,), device=device
            )
        else:
            exit_index = torch.empty(1, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(exit_index, src=0)

        return exit_index.item()

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        source_image_latent: Optional[torch.Tensor] = None,
        pixel_height: int = 1024,
        pixel_width: int = 1024,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Simulate the image editing trajectory for DMD/Consistency training.

        Input:
            - noise: [B, Seq_Len, C] - The noisy input.
            - source_image_latent: [B, Seq_Len, C] - The reference/condition image (clean).
                           Required for Image Edit tasks.
        """

        # 1. Handle Input Shape (Spatial Latents [B, latent_channel, H, W] -> Tokens [B, Seq, C])
        if noise.dim() == 4:
            batch_size, channels, latent_height, latent_width = noise.shape
            # Pack to tokens for the transformer loop
            curr_noisy_input = self.generator._pack_latents(
                noise, batch_size, latent_height, latent_width
            )
        elif noise.dim() == 3:
            # Already tokens (shouldn't happen with new dmd.py logic, but good for safety)
            curr_noisy_input = noise.clone()
            batch_size, seq_len, num_channels = noise.shape
            # We need height/width for unpacking later.
            # Assuming standard aspect ratio if not provided, or derive from seq_len if possible.
            # But better to rely on pixel_height/pixel_width args.
            latent_height = pixel_height // self.generator.pipe.vae_scale_factor
            latent_width = pixel_width // self.generator.pipe.vae_scale_factor
        else:
            raise ValueError(f"Unsupported noise shape: {noise.shape}")

        # Handle source_image_latent packing if needed
        if source_image_latent is not None and source_image_latent.dim() == 4:
            b, c, h, w = source_image_latent.shape
            source_image_latent = self.generator._pack_latents(
                source_image_latent, b, h, w
            )

        # 2. Determine where to stop the simulation (The "Self-Forcing" part)
        num_denoising_steps = len(self.denoising_step_list)
        exit_step_index = self._get_random_exit_step(
            num_denoising_steps, device=noise.device
        )

        timestep_tensor = torch.full(
            (batch_size,),
            1,
            device=noise.device,
            dtype=torch.long,
        )

        # 3. Denoising Loop
        for index, current_timestep_val in enumerate(self.denoising_step_list):
            # Create timestep tensor [B]
            timestep_tensor = torch.full(
                (batch_size,),
                current_timestep_val,
                device=noise.device,
                dtype=torch.long,
            )

            # Check if this is our stop point
            is_exit_step = index == exit_step_index

            # --- Generator Call ---
            with torch.set_grad_enabled(is_exit_step):
                # We call the Qwen Wrapper.
                # Note: We pass 'pixel_height/width' not latent dimensions.
                model_output = self.generator(
                    noisy_latents=curr_noisy_input,  # [B, Seq, C]
                    conditional_dict=conditional_dict,
                    timestep=timestep_tensor,
                    height=pixel_height,
                    width=pixel_width,
                    image_latents=source_image_latent,  # [B, Seq, C]
                )
                # Qwen Wrapper returns (flow_pred, pred_x0)
                # pred_x0 is [B, Seq, C]
                _, pred_x0 = model_output

                # Ensure pred_x0 is correct shape/type just in case
                denoised_pred = pred_x0

            if is_exit_step:
                # --- EXIT ---
                # We have reached the random step. Return the generator's x0 prediction.
                # The outer loop will compare this 'denoised_pred' against the real target image (DMD Loss).

                # Unpack back to spatial latents [B, 4, H, W]
                # We use the latent dimensions derived from input noise
                denoised_pred_spatial = self.generator._unpack_latents(
                    denoised_pred, pixel_height, pixel_width
                )

                # Calculate timestep indices for logging
                # (Find where current_timestep_val sits in the full 0-1000 scheduler)
                denoised_timestep_from = (
                    1000
                    - torch.argmin(
                        (
                            self.scheduler.timesteps.to(noise.device)
                            - current_timestep_val
                        ).abs(),
                        dim=0,
                    ).item()
                )

                if index == len(self.denoising_step_list) - 1:
                    denoised_timestep_to = 0
                else:
                    next_t = self.denoising_step_list[index + 1]
                    denoised_timestep_to = (
                        1000
                        - torch.argmin(
                            (self.scheduler.timesteps.to(noise.device) - next_t).abs(),
                            dim=0,
                        ).item()
                    )

                return (
                    denoised_pred_spatial,
                    denoised_timestep_from,
                    denoised_timestep_to,
                )

            else:
                # --- CONTINUE ---
                # Trajectory Consistency: Use the predicted x0 to sample x_{t-1} (or x_{next_step})
                # This simulates the error accumulation of the model.
                with torch.no_grad():
                    next_timestep_val = self.denoising_step_list[index + 1]

                    # Prepare timesteps for scheduler [B]
                    next_timestep_tensor = torch.full(
                        (batch_size,),
                        next_timestep_val,
                        device=noise.device,
                        dtype=torch.long,
                    )

                    # Add noise back to the predicted x0 to get x_{next_step}
                    # FlowMatchScheduler.add_noise(original_samples, noise, timesteps)
                    # Result is x_t = (1-sigma)x0 + sigma*epsilon (for Rectified Flow)
                    curr_noisy_input = self.scheduler.add_noise(
                        denoised_pred,
                        torch.randn_like(denoised_pred),
                        next_timestep_tensor,
                    )

        # Fallback (should be covered by exit_step logic)
        return curr_noisy_input.unsqueeze(1), 0, 0
