
"""
Minimal fine-tuning script for Sana

* Loads a Sana pipeline and moves components to CUDA
* Reads images from a folder
* Finetunes ONLY the Transformer inside the pipeline for N iterations (default: 50)
* Logs key metrics to Weights & Biases (optional)
* Uses correct RF/Flow training target, masks, CFG at validation, and bf16 autocast on H100
* Initializes Flow timesteps to 20 and samples training timesteps correctly from the scheduler

Notes
* This script assumes all images have the same caption; pass it via args.target_concept
* We freeze the VAE and text encoder; only the Transformer is trainable.
* Dtypes: weights remain fp32; autocast to bf16 in forward passes.
"""

import os
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from torchvision import transforms

from diffusers import (
    SanaPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderDC,
    SanaTransformer2DModel,
)
from diffusers.utils import is_wandb_available
from transformers import GemmaTokenizerFast, Gemma2Model
from safetensors.torch import load_file

if is_wandb_available():
    import wandb

# Text dropout probability for CFG-friendly training (fixed per request)
TEXT_DROPOUT_P = 0.1


def parse_args():
    p = argparse.ArgumentParser("Fine-tune Sana Transformer on an imagefolder")
    p.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Efficient-Large-Model/SANA_Sprint_0.6B_1024px_teacher_diffusers",
    )
    p.add_argument(
        "--load_transformer_weights_from",
        type=str,
        default=None,
    )
    p.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/sharifm/students/lz1/work/Meta-Unlearning/dataset/hrm_s/",
        help="Folder with images (imagefolder layout).",
    )
    p.add_argument("--target_concept", type=str, default="nudity", help="Text guidance passed during FT")
    p.add_argument("--output_dir", type=str, default="./sana_ft_outputs")
    p.add_argument("--exp_name", type=str, default="sana_ft")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument(
        "--center_crop",
        action="store_true",
        help="Use center crop instead of random crop.",
    )
    p.add_argument("--random_flip", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_iters", type=int, default=50, help="Training iterations.")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    # p.add_argument("--weight_decay", type=float, default=1e-4)
    # p.add_argument("--adam_beta1", type=float, default=0.9)
    # p.add_argument("--adam_beta2", type=float, default=0.999)
    # p.add_argument("--adam_eps", type=float, default=1e-8)
    # p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    # Gradient accumulation (effective batch size control)
    p.add_argument("--grad_accum_steps", type=int, default=8)

    # Flow / sampling
    p.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Flow steps (also used for training timestep sampling).",
    )
    p.add_argument("--eta", type=float, default=0.0)

    # Validation
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument(
        "--validation_prompts",
        type=str,
        nargs="*",
        default=[
            "a scenic mountain landscape, sunrise",
            "a bowl of fresh berries on a wooden table",
            "an abstract geometric composition, vibrant",
            "japan body",
        ],
    )
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--num_images_per_prompt", type=int, default=1)

    # Logging
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="sana_ft")
    p.add_argument("--allow_tf32", action="store_true")
    return p.parse_args()


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def encode_prompt(tokens: GemmaTokenizerFast, encoder: Gemma2Model, texts, device):
    """
    Returns (embeddings, attention_mask) for a list of texts.
    Matches Sana pipeline conventions: right padding, max_length<=300, and final index selection.
    """
    tokens.padding_side = "right"
    tok = tokens(
        texts,
        padding="max_length",
        truncation=True,
        max_length=min(tokens.model_max_length, 300),
        return_tensors="pt",
    )
    input_ids = tok["input_ids"].to(device)
    attn_mask = tok["attention_mask"].to(device)
    emb = encoder(input_ids=input_ids, attention_mask=attn_mask, return_dict=False)[
        0
    ]  # (B, L, D)

    max_len = min(tokens.model_max_length, 300)
    select_index = [0] + list(range(-max_len + 1, 0))
    emb = emb[:, select_index]
    attn_mask = attn_mask[:, select_index]
    return emb, attn_mask


def build_dataloader(args, tokenizer):
    """
    ImageFolder without captions -> unconditional (“”) text per sample.
    """
    ds = load_dataset("imagefolder", data_dir=args.dataset_dir)

    tfm = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip()
            if args.random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess(ex):
        imgs = [im.convert("RGB") for im in ex["image"]]
        ex["pixel_values"] = [tfm(im) for im in imgs]
        # unconditional text
        ex["text"] = [args.target_concept] * len(imgs)
        return ex

    ds = ds["train"].with_transform(preprocess)

    def collate(batch):
        pixel_values = (
            torch.stack([b["pixel_values"] for b in batch])
            .to(memory_format=torch.contiguous_format)
            .float()
        )
        # put text in batch to keep API consistent; we’ll re‑encode each step for the current batch size
        texts = [b.get("text", "") for b in batch]
        return {"pixel_values": pixel_values, "texts": texts}

    return DataLoader(
        ds,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=True,
    )


@torch.no_grad()
def validate(args, vae, text_encoder, tokenizer, transformer, device, step):
    """
    Simple image generation to sanity‑check training. Uses CFG.
    """
    pipe = SanaPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        torch_dtype=torch.bfloat16,  # inference in bf16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    images = []
    for prompt in args.validation_prompts:
        out = pipe(
            prompt=prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images_per_prompt,
        )
        images.extend(out.images)

    if args.use_wandb:
        wandb.log(
            {
                "val/images": [
                    wandb.Image(
                        im,
                        caption=f"{i}: {args.validation_prompts[i % len(args.validation_prompts)]}",
                    )
                    for i, im in enumerate(images)
                ]
            },
            step=step,
        )

    del pipe
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    args.exp_name = f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device("cuda")

    # 1) Load pipeline and move to CUDA
    pipe = SanaPipeline.from_pretrained(args.pretrained_model_name_or_path)
    scheduler: DPMSolverMultistepScheduler = pipe.scheduler
    tokenizer: GemmaTokenizerFast = pipe.tokenizer
    vae: AutoencoderDC = pipe.vae
    text_encoder: Gemma2Model = pipe.text_encoder
    transformer: SanaTransformer2DModel = pipe.transformer

    if args.load_transformer_weights_from is not None:
        print(f"Loading transformer weights from {args.load_transformer_weights_from}")
        transformer_weights = load_file(args.load_transformer_weights_from)
        unexpected_keys = transformer.load_state_dict(transformer_weights, strict=False)
        assert unexpected_keys.unexpected_keys == [], unexpected_keys.unexpected_keys
    else:
        print("Keeping the original transformer weights")

    # Sanity checks
    if getattr(scheduler.config, "prediction_type", None) != "flow_prediction":
        raise ValueError(
            f"Unexpected scheduler.prediction_type={scheduler.config.prediction_type}; Sana uses 'flow_prediction'."
        )

    vae.requires_grad_(False).to(device)
    text_encoder.requires_grad_(False).to(device)
    transformer.requires_grad_(True).train().to(device)

    # 2) Data
    train_loader = build_dataloader(args, tokenizer)

    # 3) Optimizer
    optim_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        optim_params,
        lr=args.learning_rate,
        # betas=(args.adam_beta1, args.adam_beta2),
        # eps=args.adam_eps,
        # weight_decay=args.weight_decay,
    )

    # 4) WandB
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.exp_name, config=vars(args))
        wandb.watch(transformer, log=None)

    # 5) Training
    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    # scaler = None  # not using GradScaler for bf16
    scheduler.set_timesteps(
        args.num_inference_steps, device=device
    )  # IMPORTANT: 20 RF steps
    timesteps = scheduler.timesteps  # (num_inference_steps,) e.g., length 20

    for epoch in range(10**9):  # iterate loader until reaching num_iters
        for batch_idx, batch in enumerate(train_loader):
            transformer.train()

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            bsz = pixel_values.shape[0]

            # Encode text (unconditional “”) and get masks
            texts = batch["texts"]
            # Apply text-dropout per-sample (p=0.1)
            if TEXT_DROPOUT_P > 0:
                import random
                texts = [t if random.random() > TEXT_DROPOUT_P else "" for t in texts]
            with torch.no_grad():
                # Text on same device as encoder
                text_encoder.to(device)
                enc_hid, enc_mask = encode_prompt(
                    tokenizer, text_encoder, texts, device=device
                )

            # VAE encode -> latents in transformer’s device
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                latents = vae.encode(pixel_values)[
                    "latent"
                ]  # (B, C, 32, 32) for 1024px DC VAE
                latents = (latents * vae.config.scaling_factor).to(
                    device, non_blocking=True
                )

            # RF noise sampling: choose per‑sample indices from scheduler.timesteps
            with torch.no_grad():
                idx = torch.randint(
                    low=0, high=timesteps.shape[0], size=(bsz,), device=device
                )
                sched_t = timesteps[idx]  # shape (B,)
                # add noise in latent space
                noise = torch.randn_like(latents)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps=sched_t)
                # scale timesteps as in Sana
                scaled_timesteps = (sched_t * transformer.config.timestep_scale).to(
                    latents.dtype
                )

            # Target for Rectified Flow
            # See Sana paper: velocity prediction (noise - latents)
            if scheduler.config.prediction_type != "flow_prediction":
                raise ValueError(
                    "Scheduler must be in 'flow_prediction' mode for Sana targets."
                )
            target = (noise - latents).float()

            # Forward + loss (bf16 compute; fp32 params)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_pred = transformer(
                    noisy_latents,
                    timestep=scaled_timesteps,
                    encoder_hidden_states=enc_hid.to(device),
                    encoder_attention_mask=enc_mask.to(device),
                    return_dict=False,
                )[0]

                loss = F.mse_loss(model_pred.float(), target, reduction="mean")

            # Gradient accumulation: scale loss and backprop
            (loss / args.grad_accum_steps).backward()
            running_loss += loss.detach().float().item()
            # Step on accumulation boundary
            if ((batch_idx + 1) % args.grad_accum_steps) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                # Logs (average over the accumulation window)
                avg_loss = running_loss / args.grad_accum_steps
                running_loss = 0.0
                if args.use_wandb:
                    wandb.log(
                        {
                            'train/loss': float(avg_loss),
                            'train/lr': optimizer.param_groups[0]['lr'],
                            'train/timestep_index_mean': float(idx.float().mean().item()),
                        },
                        step=global_step,
                    )
                # Periodic validation with CFG
                if (global_step % args.eval_every == 0) or (global_step == 1):
                    validate(args, vae, text_encoder, tokenizer, transformer, device, global_step)
                if global_step >= args.num_iters:
                    break
        if global_step >= args.num_iters:
            break

    # Save transformer weights
    save_dir = os.path.join(args.output_dir, args.exp_name, f"step_{global_step:06d}")
    os.makedirs(save_dir, exist_ok=True)
    transformer.save_pretrained(save_dir)
    if args.use_wandb:
        wandb.summary["final_step"] = global_step
        wandb.summary["save_dir"] = save_dir
        wandb.finish()

    print(f"[sana_ft] Done. Saved transformer to: {save_dir}")


if __name__ == "__main__":
    main()
