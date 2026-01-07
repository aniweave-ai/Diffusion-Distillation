import os
import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import save_file
from tqdm import tqdm

import wandb
from model import DMD
from utils.dataset import ImageEditDataset, cycle
from utils.distributed import (
    EMA_FSDP,
    fsdp_state_dict,
    fsdp_wrap,
    launch_distributed_job,
)
from utils.misc import merge_dict_list, set_seed
from utils.qwen_image_edit_wrapper import FAKE_LORA_NAME, GENERATOR_LORA_NAME


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            if config.wandb_host:
                wandb.login(host=config.wandb_host, key=config.wandb_key)
            else:
                wandb.login(key=config.wandb_key, host="https://api.wandb.ai")
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = DMD(config, device=self.device)

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # Wrap the shared model once to avoid double wrapping error
        self.model.qwen_image_edit_wrapper = fsdp_wrap(
            self.model.qwen_image_edit_wrapper,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )

        # Update references to the wrapped model
        self.model.generator = self.model.qwen_image_edit_wrapper
        self.model.real_score = self.model.qwen_image_edit_wrapper
        self.model.fake_score = self.model.qwen_image_edit_wrapper

        self.model.generator.switch_to_generator()

        target_generator_params = [
            param
            for name, param in self.model.generator.named_parameters()
            if param.requires_grad and GENERATOR_LORA_NAME in name
        ]

        print(f"Generator params: {len(target_generator_params)}")

        self.generator_optimizer = torch.optim.AdamW(
            target_generator_params,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        self.model.fake_score.switch_to_fake()
        target_critic_params = [
            param
            for name, param in self.model.fake_score.named_parameters()
            if param.requires_grad and FAKE_LORA_NAME in name
        ]
        print(f"Critic params: {len(target_critic_params)}")
        self.critic_optimizer = torch.optim.AdamW(
            target_critic_params,
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay,
        )

        # Step 3: Initialize the dataloader

        dataset = ImageEditDataset(config.data_path)

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8
        )

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        self.ema_weight = config.get("ema_weight", -1.0)
        self.ema_start_step = config.get("ema_start_step", 0)
        self.generator_ema = None
        if (self.ema_weight > 0.0) and (self.step >= self.ema_start_step):
            print(f"Setting up EMA with weight {self.ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "resume_ckpt", False):
            print(f"Resuming training from {config.resume_ckpt}")

            # Set resume step
            if getattr(config, "resume_step", False):
                self.step = config.resume_step
                print(f"Resuming from step {self.step}")

            # Load generator_ema checkpoint (if exists)
            generator_ema_path = os.path.join(config.resume_ckpt, "generator_ema.pt")
            if os.path.exists(generator_ema_path):
                # Initialize EMA if not already initialized (needed for loading state)
                if self.generator_ema is None and self.ema_weight > 0.0:
                    print("Initializing EMA for resume...")
                    generator_state_dict = torch.load(
                        generator_ema_path, map_location="cpu"
                    )
                    # FSDP will automatically handle dtype conversion
                    self.model.generator.load_state_dict(
                        generator_state_dict, strict=True
                    )
                    self.generator_ema = EMA_FSDP(
                        self.model.generator, decay=self.ema_weight
                    )
                    print("Generator EMA checkpoint loaded successfully")
            else:
                print(
                    f"Info: Generator EMA checkpoint not found at {generator_ema_path}"
                )

            # Load generator checkpoint
            generator_path = os.path.join(config.resume_ckpt, "generator.pt")
            if os.path.exists(generator_path):
                print(f"Loading generator from {generator_path}")
                generator_state_dict = torch.load(generator_path, map_location="cpu")
                # FSDP will automatically handle dtype conversion
                self.model.generator.load_state_dict(generator_state_dict, strict=True)
                print("Generator checkpoint loaded successfully")
            else:
                print(f"Warning: Generator checkpoint not found at {generator_path}")

            # Load critic checkpoint
            critic_path = os.path.join(config.resume_ckpt, "critic.pt")
            if os.path.exists(critic_path):
                print(f"Loading critic from {critic_path}")
                critic_state_dict = torch.load(critic_path, map_location="cpu")
                # FSDP will automatically handle dtype conversion
                self.model.fake_score.load_state_dict(critic_state_dict, strict=True)
                print("Critic checkpoint loaded successfully")
            else:
                print(f"Warning: Critic checkpoint not found at {critic_path}")

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        # if self.step < config.ema_start_step:
        #     self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")

        # 1. Gather full weights (CPU)
        full_gen_sd = fsdp_state_dict(self.model.generator)
        full_critic_sd = fsdp_state_dict(self.model.fake_score)

        # Helper: Keep only specific LoRA branch and rename keys
        def clean_lora_weights(state_dict, distinct_keyword):
            """
            Filters keys containing 'lora' AND the distinct_keyword.
            Removes the distinct_keyword from the name to normalize it.
            """
            clean_sd = {}
            target_str = f".{distinct_keyword}"  # e.g. ".generator" or ".fake"

            for k, v in state_dict.items():
                if "lora" in k and target_str in k:
                    # Rename: transformer.blocks...lora_A.generator.weight
                    #      -> transformer.blocks...lora_A.weight
                    new_k = k.replace(target_str, "")
                    clean_sd[new_k] = v
            return clean_sd

        # 2. Process Generator: Remove ".generator"
        generator_lora_sd = clean_lora_weights(full_gen_sd, "generator")

        # 3. Process Critic: Remove ".fake"
        critic_lora_sd = clean_lora_weights(full_critic_sd, "fake")

        # 4. Handle EMA (EMA tracks Generator, so we treat it like Generator)
        ema_lora_sd = None
        if (self.ema_weight > 0.0) and (self.ema_start_step < self.step):
            full_ema_sd = self.generator_ema.state_dict()
            # EMA usually has the same structure as the generator
            ema_lora_sd = clean_lora_weights(full_ema_sd, "generator")

        # Sanity Checks
        if not generator_lora_sd:
            print("WARNING: Generator LoRA dict is empty! Check keys for '.generator'")
        if not critic_lora_sd:
            print("WARNING: Critic LoRA dict is empty! Check keys for '.fake'")

        # 5. Save to disk (Only on Rank 0)
        if self.is_main_process:
            save_dir = os.path.join(
                self.output_path, f"checkpoint_model_{self.step:06d}"
            )
            os.makedirs(save_dir, exist_ok=True)

            # --- Save Generator LoRA ---
            gen_path = os.path.join(save_dir, "generator_lora.safetensors")
            save_file(generator_lora_sd, gen_path)
            print(f"Generator LoRA ({len(generator_lora_sd)} keys) saved to {gen_path}")

            # --- Save Critic LoRA ---
            critic_path = os.path.join(save_dir, "critic_lora.safetensors")
            save_file(critic_lora_sd, critic_path)
            print(f"Critic LoRA ({len(critic_lora_sd)} keys) saved to {critic_path}")

            # --- Save EMA LoRA ---
            if ema_lora_sd is not None:
                ema_path = os.path.join(save_dir, "generator_ema_lora.safetensors")
                save_file(ema_lora_sd, ema_path)
                print(f"Generator EMA LoRA saved to {ema_path}")

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # if self.step % 20 == 0:
        # torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            img = batch["img"].to(self.device)  # Shape: [B, 3, 1024, 1024])
            conditional_dict = self.model.encode_prompt(image=img, prompt=text_prompts)
            unconditional_dict = self.model.encode_prompt(
                image=img, prompt=self.config.negative_prompt
            )

            image_latent = self.model.run_vae_encoder(img)

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            self.model.switch_to_generator()
            # TODO: generator grad will missing since lora switch, fine a way to escape this issue
            torch.cuda.synchronize()
            start_time = time.time()

            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                initial_latent=image_latent,
            )
            torch.cuda.synchronize()
            generator_loss_time = time.time() - start_time

            torch.cuda.empty_cache()

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator
            )

            generator_log_dict.update(
                {
                    "generator_loss": generator_loss,
                    "generator_grad_norm": generator_grad_norm,
                    "generator_loss_time": generator_loss_time,
                }
            )

            return generator_log_dict
        else:
            self.model.switch_to_fake()
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        torch.cuda.synchronize()
        start_time = time.time()
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            initial_latent=image_latent,
        )
        torch.cuda.synchronize()
        critic_loss_time = time.time() - start_time

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic
        )

        critic_log_dict.update(
            {
                "critic_loss": critic_loss,
                "critic_grad_norm": critic_grad_norm,
                "critic_loss_time": critic_loss_time,
            }
        )

        return critic_log_dict

    def train(self):
        start_step = self.step

        # specific for tqdm: try to get total steps from config, else use a default
        total_steps = getattr(
            self.config,
            "max_iters",
            getattr(self.config, "total_training_step", 100000),
        )

        # Initialize Progress Bar (Only on Rank 0)
        pbar = tqdm(
            initial=self.step,
            total=total_steps,
            disable=not self.is_main_process,
            desc="DMD Training",
            dynamic_ncols=True,
        )

        while True:
            # We remove the simple print to avoid cluttering the progress bar
            # if self.is_main_process:
            #     print(f"training step {self.step} ...")

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            # --- Train the generator ---
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    ema_update_dict = self.generator_ema.update(self.model.generator)
                    generator_log_dict.update(ema_update_dict)
            else:
                # Initialize empty so we don't crash if accessing later
                generator_log_dict = {}

            # --- Train the critic ---
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (
                (self.step >= self.ema_start_step)
                and (self.generator_ema is None)
                and (self.ema_weight > 0)
            ):
                self.generator_ema = EMA_FSDP(
                    self.model.generator, decay=self.ema_weight
                )

            # Save the model
            if (
                (not self.config.no_save)
                and (self.step - start_step) > 0
                and self.step % self.config.log_iters == 0
            ):
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # --- Logging & Progress Bar ---
            if self.is_main_process:
                wandb_loss_dict = {}

                # Extract scalar values safely
                c_loss = critic_log_dict["critic_loss"].mean().item()
                c_grad = critic_log_dict["critic_grad_norm"].mean().item()

                g_loss = 0.0
                g_grad = 0.0
                dmd_grad = 0.0

                if TRAIN_GENERATOR and "generator_loss" in generator_log_dict:
                    g_loss = generator_log_dict["generator_loss"].mean().item()
                    g_grad = generator_log_dict["generator_grad_norm"].mean().item()
                    dmd_grad = (
                        generator_log_dict.get(
                            "dmdtrain_gradient_norm", torch.tensor(0.0)
                        )
                        .mean()
                        .item()
                    )

                    wandb_loss_dict.update(
                        {
                            "generator_loss": g_loss,
                            "generator_grad_norm": g_grad,
                            "dmdtrain_gradient_norm": dmd_grad,
                            "generator_loss_time": generator_log_dict.get(
                                "generator_loss_time", 0.0
                            ),
                            "ema_update_time": generator_log_dict.get(
                                "ema_update_time", 0.0
                            ),
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": c_loss,
                        "critic_grad_norm": c_grad,
                        "critic_loss_time": critic_log_dict.get(
                            "critic_loss_time", 0.0
                        ),
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

                # Update TQDM Bar
                pbar.update(1)
                postfix_str = {
                    "C_Loss": f"{c_loss:.4f}",
                    "G_Loss": f"{g_loss:.4f}" if TRAIN_GENERATOR else "-",
                }
                pbar.set_postfix(postfix_str)

            # GC
            # if self.step % self.config.gc_interval == 0:
            #     # Use tqdm.write so it doesn't break the bar layout
            #     # if dist.get_rank() == 0:
            #     #     tqdm.write("DistGarbageCollector: Running GC.")
            #     gc.collect()
            #     torch.cuda.empty_cache()

            # Timing
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log(
                            {"per iteration time": current_time - self.previous_time},
                            step=self.step,
                        )
                    self.previous_time = current_time

            # Stop condition
            if self.step >= total_steps:
                if self.is_main_process:
                    pbar.close()
                    print("Training finished.")
                break
