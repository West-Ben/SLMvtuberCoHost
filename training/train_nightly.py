"""
training/train_nightly.py — nightly QLoRA adapter on base_core (Gemma 4 E4B).

  Unsloth FastVisionModel, 4-bit, language layers only (finetune_vision_layers=False)
  r=16, alpha=r, 1 epoch, LR 2e-5..5e-5, batch 1 + gradient accumulation,
  use_gradient_checkpointing="unsloth".

Prompts are NOT re-templated: each row already carries the exact prompt text
from prompt_format (byte-identical to inference) plus the completion. The
collator tokenises prompt+completion with the processor (which expands each
<|image|> placeholder) and masks the loss to the response only — the
train_on_responses_only rule with the Gemma 4 markers (prompt_format
INSTRUCTION_PART / RESPONSE_PART): labels are -100 up to the end of the last
RESPONSE_PART. Unsloth's train_on_responses_only() works on pre-tokenised text
datasets; with a lazy vision collator we apply the identical rule here.

Trainer: TRL SFTTrainer by default; --trainer hf uses the standard HF Trainer
(fallback if SFTTrainer misbehaves). Runs the same locally or in Colab — pass
--image-root and --base for the Colab paths.

Run:
  python -m training.train_nightly --run-id <datasets/RUN_ID> [--check] [--export-gguf]
  --check   tokenise the first example, print token/label/image stats, exit (no training)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prompt_format    # noqa: E402

IGNORE_INDEX = -100
DEFAULTS = dict(r=16, lr=3e-5, epochs=1, grad_accum=8, max_len=4096, seed=3407)


# ─── Pure helpers (unit-tested without torch) ─────────────

def find_last_subsequence(seq: list[int], sub: list[int]) -> int:
    """Start index of the last occurrence of sub in seq, or -1."""
    n = len(sub)
    for i in range(len(seq) - n, -1, -1):
        if seq[i:i + n] == sub:
            return i
    return -1


def responses_only_labels(input_ids: list[int], response_ids: list[int]) -> list[int]:
    """Mask everything up to and including the last response marker."""
    start = find_last_subsequence(input_ids, response_ids)
    if start < 0:
        raise ValueError("response marker not found — prompt was not built by prompt_format")
    cut = start + len(response_ids)
    labels = [IGNORE_INDEX] * cut + list(input_ids[cut:])
    if all(l == IGNORE_INDEX for l in labels):
        raise ValueError("no response tokens to train on")
    return labels


class ReactionCollator:
    """Batch size 1. Loads frames lazily, tokenises the exact text, masks to the response."""

    def __init__(self, processor, image_root: Path, max_len: int):
        self.processor = processor
        self.tok = getattr(processor, "tokenizer", processor)
        self.image_root = Path(image_root)
        self.max_len = max_len
        self.response_ids = self.tok(prompt_format.RESPONSE_PART, add_special_tokens=False)["input_ids"]

    def encode(self, ex: dict) -> dict:
        from PIL import Image
        import torch
        imgs = [Image.open(self.image_root / p).convert("RGB") for p in ex["images"]]
        text = ex["prompt"] + ex["completion"]
        kw = dict(text=[text], return_tensors="pt", add_special_tokens=False)   # <bos> is in the text
        if imgs:
            kw["images"] = [imgs]
        enc = dict(self.processor(**kw))
        ids = enc["input_ids"][0].tolist()
        if len(ids) > self.max_len:
            raise ValueError(f"example {ex.get('id')} is {len(ids)} tokens > max_len {self.max_len}")
        enc["labels"] = torch.tensor([responses_only_labels(ids, self.response_ids)])
        return enc

    def __call__(self, batch: list[dict]) -> dict:
        if len(batch) != 1:
            raise ValueError("ReactionCollator expects per_device_train_batch_size=1")
        return self.encode(batch[0])


def _read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ─── Training ─────────────────────────────────────────────

def load_model(base: str, r: int, seed: int):
    from unsloth import FastVisionModel
    model, processor = FastVisionModel.from_pretrained(
        base, load_in_4bit=True, use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,      # language layers only
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=r, lora_alpha=r, lora_dropout=0, bias="none", random_state=seed,
    )
    return model, processor


def check(processor, rows: list[dict], image_root: Path, max_len: int):
    col = ReactionCollator(processor, image_root, max_len)
    ex = rows[0]
    enc = col.encode(ex)
    ids = enc["input_ids"][0].tolist()
    labels = enc["labels"][0].tolist()
    trained = [t for t, l in zip(ids, labels) if l != IGNORE_INDEX]
    image_token_id = getattr(processor, "image_token_id", None)
    print(json.dumps({
        "id": ex.get("id"), "tokens": len(ids), "trained_tokens": len(trained),
        "images": len(ex["images"]), "has_pixel_values": "pixel_values" in enc,
        "image_tokens": ids.count(image_token_id) if image_token_id is not None else None,
        "response_marker_ids": col.response_ids,
        "trained_text": col.tok.decode(trained),
    }, indent=2))
    if ex["images"] and image_token_id is not None and ids.count(image_token_id) <= len(ex["images"]):
        raise SystemExit("image placeholders were not expanded by the processor — check IMAGE_TOKEN vs processor")


def train(args):
    run_dir = Path(args.run_id) if Path(args.run_id).is_dir() else ROOT / "datasets" / args.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = _read(run_dir / "train.jsonl")
    model, processor = load_model(args.base, args.r, args.seed)
    if args.check:
        check(processor, rows, Path(args.image_root), args.max_len)
        return

    from datasets import Dataset
    from unsloth import FastVisionModel
    FastVisionModel.for_training(model)
    dataset = Dataset.from_list(rows)
    collator = ReactionCollator(processor, Path(args.image_root), args.max_len)
    out_dir = Path(args.out or ROOT / "adapters" / args.version)
    common = dict(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        optim="adamw_8bit",
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )
    if args.trainer == "sft":
        from trl import SFTConfig, SFTTrainer
        trainer = SFTTrainer(
            model=model, train_dataset=dataset, data_collator=collator,
            processing_class=getattr(processor, "tokenizer", processor),
            args=SFTConfig(**common, dataset_kwargs={"skip_prepare_dataset": True}, max_length=args.max_len),
        )
    else:
        from transformers import Trainer, TrainingArguments
        trainer = Trainer(model=model, train_dataset=dataset, data_collator=collator,
                          args=TrainingArguments(**common))
    stats = trainer.train()

    lora_dir = out_dir / "lora"
    model.save_pretrained(str(lora_dir))
    processor.save_pretrained(str(lora_dir))
    # Unique filename per version: the orchestrator maps llama-server adapters by file name.
    gguf = export_gguf(lora_dir, args.base, out_dir / f"{args.version}.gguf") if args.export_gguf else None

    from training import registry
    registry.register({
        "version": args.version, "parent": manifest.get("parent"),
        "data_range": manifest["data_range"], "dataset": run_dir.name,
        "path": str(lora_dir.relative_to(ROOT)) if lora_dir.is_relative_to(ROOT) else str(lora_dir),
        "gguf": str(gguf.relative_to(ROOT)) if gguf and gguf.is_relative_to(ROOT) else (str(gguf) if gguf else None),
        "prompt_version": manifest["prompt_version"], "grammar_hash": manifest["grammar_hash"],
        "train": {"r": args.r, "alpha": args.r, "lr": args.lr, "epochs": args.epochs,
                  "grad_accum": args.grad_accum, "trainer": args.trainer,
                  "loss": round(float(stats.training_loss), 4)},
    })
    print(f"registered candidate {args.version} → run `python -m training.eval_gate {args.version}`")


def export_gguf(lora_dir: Path, base: str, outfile: Path) -> Path:
    """LoRA → GGUF for llama-server --lora (llama.cpp convert_lora_to_gguf.py)."""
    from config import LLAMACPP_DIR
    subprocess.run([sys.executable, str(LLAMACPP_DIR / "convert_lora_to_gguf.py"), str(lora_dir),
                    "--base", base, "--outfile", str(outfile)], check=True)
    return outfile


if __name__ == "__main__":
    from config import BASE_CORE_HF
    from training import registry
    from datetime import datetime
    ap = argparse.ArgumentParser(description="Nightly QLoRA adapter (language layers only)")
    ap.add_argument("--run-id", required=True, help="datasets/<run_id> (or a path)")
    ap.add_argument("--base", default=BASE_CORE_HF)
    ap.add_argument("--image-root", default=str(ROOT), help="root that row image paths are relative to")
    ap.add_argument("--version", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--r", type=int, default=DEFAULTS["r"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"], help="2e-5 .. 5e-5")
    ap.add_argument("--epochs", type=float, default=DEFAULTS["epochs"])
    ap.add_argument("--grad-accum", type=int, default=DEFAULTS["grad_accum"])
    ap.add_argument("--max-len", type=int, default=DEFAULTS["max_len"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--trainer", choices=("sft", "hf"), default="sft")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--export-gguf", action="store_true")
    a = ap.parse_args()
    if not 2e-5 <= a.lr <= 5e-5:
        raise SystemExit(f"--lr {a.lr} outside the 2e-5..5e-5 nightly range")
    a.version = a.version or registry.next_version(registry.load(), datetime.now().strftime("%Y%m%d"))
    train(a)
