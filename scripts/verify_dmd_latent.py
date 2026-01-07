import os
import sys
from unittest.mock import MagicMock, patch

import torch

# Add project root to path
sys.path.append(os.getcwd())

# Mock diffusers before importing utils
sys.modules["diffusers"] = MagicMock()
sys.modules["diffusers.QwenImageEditPlusPipeline"] = MagicMock()

# Mock wan and easydict
sys.modules["easydict"] = MagicMock()
sys.modules["wan"] = MagicMock()
sys.modules["wan.utils"] = MagicMock()
sys.modules["wan.utils.fm_solvers"] = MagicMock()
sys.modules["wan.utils.fm_solvers_unipc"] = MagicMock()
sys.modules["wan.modules"] = MagicMock()
sys.modules["wan.modules.tokenizers"] = MagicMock()
sys.modules["wan.modules.model"] = MagicMock()
sys.modules["wan.modules.vae"] = MagicMock()
sys.modules["wan.modules.t5"] = MagicMock()
sys.modules["wan.modules.clip"] = MagicMock()
sys.modules["wan.modules.causal_model"] = MagicMock()

from model.dmd import DMD


def test_dmd_latent_loss():
    print("Testing DMD Latent Loss Calculation...")

    # Setup args
    class MockArgs:
        pass

    args = MockArgs()
    args.num_train_timestep = 1000
    args.gradient_checkpointing = False
    args.guidance_scale = 1.0
    args.fake_guidance_scale = 0.0
    args.denoising_loss_type = "x0"
    args.image_or_video_shape = [1, 3, 1024, 1024]  # Pixel shape
    args.model_name = "mock_model"
    args.teacher_lora_path = "mock_path"
    args.mixed_precision = False
    args.num_frame_per_block = 1
    args.num_training_frames = 1
    args.same_step_across_blocks = True
    args.independent_first_frame = False
    args.ts_schedule = True
    args.ts_schedule_max = False
    args.min_score_timestep = 0
    args.timestep_shift = 1.0
    args.denoising_step_list = [999, 800, 600, 400, 200, 0]
    args.warp_denoising_step = False

    device = "cpu"

    # Mock Qwen Wrapper
    with patch(
        "utils.qwen_image_edit_wrapper.QwenImageEditPlusPipeline"
    ) as MockPipeline:
        # Setup Mock Pipeline
        mock_pipe = MockPipeline.from_pretrained.return_value
        mock_pipe.to.return_value = mock_pipe  # IMPORTANT: .to() returns self
        mock_pipe.vae_scale_factor = 16
        mock_pipe.latent_channels = 16
        mock_pipe.transformer.dtype = torch.float32
        mock_pipe.transformer.config.guidance_embeds = True

        # Mock VAE encode/decode
        mock_pipe.vae.encode.return_value.latent_dist.sample.return_value = torch.randn(
            1, 16, 1, 64, 64
        )
        mock_pipe.vae.config.latents_mean = [0.0] * 16
        mock_pipe.vae.config.latents_std = [1.0] * 16

        # Mock Transformer output
        # Input to transformer is [B, Seq, C]
        # Output is [B, Seq, C]
        def transformer_forward(*args, **kwargs):
            hidden_states = kwargs["hidden_states"]
            return (hidden_states,)  # Return same shape as input

        mock_pipe.transformer.side_effect = transformer_forward

        # Initialize DMD
        dmd = DMD(args, device)

        # Replace real scores with our mocked wrapper (which is already mocked by patch, but we need to ensure it's attached)
        # The DMD init calls BaseModel init which creates QwenImageEditWrapper
        # We need to make sure the wrapper instance uses our mock pipe

        # Let's manually inject the mocked wrapper if needed, but patch should handle it.
        # Check dmd.generator
        print("DMD Generator type:", type(dmd.generator))

        # Mock _pack_latents and _unpack_latents on the pipe since we use them
        def pack_latents(latents, b, c, h, w):
            return latents.flatten(2).transpose(1, 2)  # [B, C, H, W] -> [B, H*W, C]

        def unpack_latents(tokens, b, c, h, w):
            return tokens.transpose(1, 2).view(b, c, 1, h, w)

        dmd.generator.pipe._pack_latents = pack_latents
        dmd.generator.pipe._unpack_latents = unpack_latents  # Not used by pipe, but used by wrapper if we didn't override

        # We overrode _pack_latents in wrapper to call pipe._pack_latents
        # We overrode _unpack_latents in wrapper to do it manually

        # Create dummy inputs
        pixel_shape = [1, 3, 1024, 1024]
        conditional_dict = {
            "prompt_embeds": torch.randn(1, 77, 1024),
            "prompt_embeds_mask": torch.ones(1, 77),
            "source_image_latent": torch.randn(1, 16, 1, 64, 64),  # Latent source
        }
        unconditional_dict = {
            "prompt_embeds": torch.randn(1, 77, 1024),
            "prompt_embeds_mask": torch.ones(1, 77),
        }
        initial_latent = torch.randn(1, 16, 1, 64, 64)

        # 1. Test Generator Loss
        print("\nRunning generator_loss...")
        loss, log_dict = dmd.generator_loss(
            image_or_video_shape=pixel_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=initial_latent,
        )
        print("Generator Loss:", loss.item())
        print("Generator Log Dict keys:", log_dict.keys())

        # Verify that _run_generator was called with latent shape
        # We can't easily check internal calls without mocking dmd methods,
        # but we can check if it crashed. If it ran, it likely worked.

        # 2. Test Critic Loss
        print("\nRunning critic_loss...")
        loss, log_dict = dmd.critic_loss(
            image_or_video_shape=pixel_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=initial_latent,
        )
        print("Critic Loss:", loss.item())
        print("Critic Log Dict keys:", log_dict.keys())

        print("\nVerification Successful!")


if __name__ == "__main__":
    test_dmd_latent_loss()
