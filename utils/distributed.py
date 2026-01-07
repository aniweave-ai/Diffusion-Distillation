import os
import time
from datetime import timedelta
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

from utils.qwen_image_edit_wrapper import GENERATOR_LORA_NAME


def fsdp_state_dict(model):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    # 1. Check if the model is wrapped in FSDP
    if isinstance(model, FSDP):
        fsdp_fullstate_save_policy = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
        ):
            checkpoint = model.state_dict()
        return checkpoint

    # 2. Handle Standard Model (Single GPU / No FSDP)
    else:
        # We manually move state_dict to CPU to mimic the offload_to_cpu=True behavior
        # and ensure it doesn't spike VRAM during save.
        return {k: v.cpu() for k, v in model.state_dict().items()}


def fsdp_wrap(
    module,
    sharding_strategy="full",
    mixed_precision=False,
    wrap_strategy="size",
    min_num_params=int(5e7),
    transformer_module=None,
    ignored_modules=None,
    cpu_offload=False,
):
    # --- CHANGE START: Auto-disable FSDP for Single GPU ---
    if dist.is_initialized() and dist.get_world_size() == 1:
        print("World Size is 1: Disabling FSDP and using standard CUDA model.")
        return module.to(torch.cuda.current_device())
    # --- CHANGE END ---

    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy, min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        ignored_modules=ignored_modules,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False,
    )
    return module


def barrier():
    if dist.is_initialized():
        dist.barrier()


def launch_distributed_job(backend: str = "nccl"):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=30),
    )
    torch.cuda.set_device(local_rank)


class EMA_FSDP:
    def __init__(self, fsdp_module: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self._init_shadow(fsdp_module)

    @torch.no_grad()
    def _init_shadow(self, fsdp_module):
        # Handle FSDP case
        if isinstance(fsdp_module, FSDP):
            with FSDP.summon_full_params(
                fsdp_module, writeback=False, offload_to_cpu=True, rank0_only=True
            ):
                for n, p in fsdp_module.module.named_parameters():
                    self.shadow[n] = p.detach().clone().float().cpu()
        # Handle Standard/Single-GPU case
        else:
            for n, p in fsdp_module.named_parameters():
                # ema is only work on generator, and we train lora so we don't have to copy the entire model.
                if GENERATOR_LORA_NAME not in n:
                    continue
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay

        # Helper to update shadow params
        def update_params(module_params):
            for n, p in module_params:
                if n in self.shadow:
                    self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1.0 - d)

        torch.cuda.synchronize()
        start_time = time.time()

        if isinstance(fsdp_module, FSDP):
            with FSDP.summon_full_params(
                fsdp_module, writeback=False, offload_to_cpu=True, rank0_only=True
            ):
                update_params(fsdp_module.module.named_parameters())
        else:
            update_params(fsdp_module.named_parameters())
        torch.cuda.synchronize()
        ema_update_time = time.time() - start_time
        return {
            "ema_update_time": ema_update_time,
        }

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}

    def copy_to(self, fsdp_module):
        # load EMA weights into the generator
        if isinstance(fsdp_module, FSDP):
            with FSDP.summon_full_params(fsdp_module, writeback=True):
                for n, p in fsdp_module.module.named_parameters():
                    if n in self.shadow:
                        p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
        else:
            for n, p in fsdp_module.named_parameters():
                if n in self.shadow:
                    p.data.copy_(self.shadow[n].to(p.dtype, device=p.device))
