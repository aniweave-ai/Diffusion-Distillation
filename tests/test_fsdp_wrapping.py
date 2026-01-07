import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

# 1. Create a fake diffusers package
diffusers = types.ModuleType("diffusers")
diffusers.__path__ = []
diffusers.models = types.ModuleType("diffusers.models")
diffusers.models.__path__ = []
diffusers.models.modeling_utils = types.ModuleType("diffusers.models.modeling_utils")
diffusers.configuration_utils = types.ModuleType("diffusers.configuration_utils")
sys.modules["diffusers"] = diffusers
sys.modules["diffusers.models"] = diffusers.models
sys.modules["diffusers.models.modeling_utils"] = diffusers.models.modeling_utils
sys.modules["diffusers.configuration_utils"] = diffusers.configuration_utils

# 2. Mock other dependencies
# Ensure 'utils' exists in sys.modules and is treated as a package
if "utils" not in sys.modules:
    sys.modules["utils"] = types.ModuleType("utils")
utils = sys.modules["utils"]
utils.__path__ = []

mock_wrapper_module = types.ModuleType("utils.qwen_image_edit_wrapper")
mock_wrapper_module.CONDITION_IMAGE_SIZE = 384 * 384
mock_wrapper_module.VAE_IMAGE_SIZE = 1024 * 1024
mock_wrapper_module.FAKE_LORA_NAME = "fake"
mock_wrapper_module.GENERATOR_LORA_NAME = "generator"
mock_wrapper_module.calculate_dimensions = MagicMock()

# Use a MagicMock for the class itself
MockWrapperClass = MagicMock()
mock_wrapper_module.QwenImageEditWrapper = MockWrapperClass
sys.modules["utils.qwen_image_edit_wrapper"] = mock_wrapper_module
setattr(utils, "qwen_image_edit_wrapper", mock_wrapper_module)

sys.modules["utils.scheduler"] = MagicMock()
sys.modules["utils.loss"] = MagicMock()
sys.modules["pipeline.image_edit_training"] = MagicMock()
sys.modules["model.diffusion"] = MagicMock()
sys.modules["utils.wan_wrapper"] = MagicMock()
sys.modules["pipeline"] = MagicMock()
sys.modules["utils.dataset"] = MagicMock()

# 3. Import classes to test
from model.dmd import DMD


class TestFSDPWrapping(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.gradient_checkpointing = False
        self.config.num_train_timestep = 1000
        self.config.mixed_precision = True
        self.config.sharding_strategy = "full"
        self.config.generator_fsdp_wrap_strategy = "size"
        self.config.model_name = "test_model"
        self.config.teacher_lora_path = "test_path"
        self.config.warp_denoising_step = False
        self.device = "cpu"

        # Reset the mock class for each test
        MockWrapperClass.reset_mock()
        self.mock_wrapper_instance = MockWrapperClass.return_value
        self.mock_wrapper_instance.get_scheduler.return_value = MagicMock()
        self.mock_wrapper_instance.get_scheduler.return_value.timesteps = (
            torch.linspace(0, 1000, 1000)
        )

    def test_adapter_switching(self):
        model = DMD(self.config, self.device)

        model.switch_to_generator()
        self.mock_wrapper_instance.switch_to_generator.assert_called_once()

        model.switch_to_real()
        self.mock_wrapper_instance.switch_to_real.assert_called_once()

        model.switch_to_fake()
        self.mock_wrapper_instance.switch_to_fake.assert_called_once()

    def test_adapter_switching_with_fsdp(self):
        model = DMD(self.config, self.device)

        # Wrap it (simulate FSDP)
        wrapped_model = MagicMock()
        wrapped_model.module = model.qwen_image_edit_wrapper
        model.qwen_image_edit_wrapper = wrapped_model

        model.switch_to_generator()
        self.mock_wrapper_instance.switch_to_generator.assert_called_once()

    def test_shared_model_wrapping(self):
        model = DMD(self.config, self.device)

        self.assertIs(model.generator, model.qwen_image_edit_wrapper)

        with patch("trainer.distillation.fsdp_wrap") as mock_fsdp_wrap:
            mock_fsdp_wrap.side_effect = lambda m, **kwargs: MagicMock(module=m)

            # Simulate Trainer initialization logic
            model.qwen_image_edit_wrapper = mock_fsdp_wrap(
                model.qwen_image_edit_wrapper,
                sharding_strategy=self.config.sharding_strategy,
                mixed_precision=self.config.mixed_precision,
                wrap_strategy=self.config.generator_fsdp_wrap_strategy,
            )
            model.generator = model.qwen_image_edit_wrapper
            model.real_score = model.qwen_image_edit_wrapper
            model.fake_score = model.qwen_image_edit_wrapper

            self.assertEqual(mock_fsdp_wrap.call_count, 1)
            self.assertIs(model.generator, model.qwen_image_edit_wrapper)


if __name__ == "__main__":
    unittest.main()
