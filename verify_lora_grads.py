import sys
from unittest.mock import MagicMock

import torch

# Mock diffusers and QwenImageEditPlusPipeline to avoid loading large models
sys.modules["diffusers"] = MagicMock()
sys.modules["diffusers"].QwenImageEditPlusPipeline = MagicMock()
sys.modules["utils.scheduler"] = MagicMock()

# Now import the wrapper
from utils.qwen_image_edit_wrapper import QwenImageEditWrapper


def test_lora_grads():
    print("Testing LoRA gradient settings...")

    # Mock the pipeline and transformer
    mock_pipe = MagicMock()
    mock_transformer = torch.nn.Module()

    # Create some dummy parameters
    # 1. Normal parameter
    layer1 = torch.nn.Linear(10, 10)
    mock_transformer.add_module("layer1", layer1)

    # 2. LoRA parameter (simulate by adding a module with lora in name)
    lora_module_A = torch.nn.Linear(10, 10)
    mock_transformer.add_module("layer1_lora_A", lora_module_A)
    # 3. Another LoRA parameter
    lora_module_B = torch.nn.Linear(10, 10)
    mock_transformer.add_module("layer1_lora_B", lora_module_B)

    mock_pipe.transformer = mock_transformer
    mock_pipe.vae_scale_factor = 8

    # Mock from_pretrained to return our mock pipe
    sys.modules[
        "diffusers"
    ].QwenImageEditPlusPipeline.from_pretrained.return_value = mock_pipe

    # Instantiate wrapper
    wrapper = QwenImageEditWrapper(model_name="test", device="cpu")

    print(f"Wrapper model: {wrapper.model}")
    print(f"Mock transformer: {mock_transformer}")
    print(f"Is same object: {wrapper.model is mock_transformer}")
    print("Model parameters:", list(wrapper.model.named_parameters()))

    # Call the method
    wrapper.set_trainable_lora_only()

    # Verify
    for name, param in wrapper.model.named_parameters():
        print(f"Checking {name}: requires_grad={param.requires_grad}")
        if "lora" in name.lower():
            assert param.requires_grad == True, f"{name} should be trainable"
        else:
            assert param.requires_grad == False, f"{name} should NOT be trainable"

    print("\nSUCCESS: Only LoRA parameters are trainable.")


if __name__ == "__main__":
    test_lora_grads()
