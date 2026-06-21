"""Single-process Apple Silicon SFT entry point for MiniOneRec smoke runs."""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Any

import fire
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset as HFDataset
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

from data import FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset


CATEGORY_NAMES = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
}


class TokenExtender:
    """Read the unique semantic-ID tokens from an existing index file."""

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path)

    def get_new_tokens(self) -> list[str]:
        with self.index_path.open(encoding="utf-8") as handle:
            indices = json.load(handle)
        if not isinstance(indices, dict):
            raise ValueError(f"SID index must be a JSON object: {self.index_path}")

        tokens: set[str] = set()
        for item_id, item_tokens in indices.items():
            if not isinstance(item_tokens, list) or not item_tokens:
                raise ValueError(f"Invalid SID token list for item {item_id}")
            if not all(isinstance(token, str) and token for token in item_tokens):
                raise ValueError(f"Invalid SID token for item {item_id}")
            tokens.update(item_tokens)
        return sorted(tokens)


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available in the current Python/PyTorch environment")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in the current Python/PyTorch environment")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {requested}")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        dtype = dtype_map[requested.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {requested}") from error
    if device.type == "mps" and dtype == torch.bfloat16:
        raise ValueError("bfloat16 is not supported by this MPS smoke-training entry point")
    return dtype


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mask_old_token_rows(parameter: nn.Parameter, original_vocab_size: int) -> None:
    def mask_gradient(gradient: torch.Tensor) -> torch.Tensor:
        masked = gradient.clone()
        masked[:original_vocab_size].zero_()
        return masked

    parameter.register_hook(mask_gradient)


def freeze_for_sid_training(model: nn.Module, original_vocab_size: int) -> dict[str, int]:
    """Freeze the backbone and retain gradients only for newly appended token rows."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    input_module = model.get_input_embeddings()
    output_module = model.get_output_embeddings()
    if input_module is None or not hasattr(input_module, "weight"):
        raise ValueError("Model has no trainable input embedding weight")

    parameters: list[nn.Parameter] = [input_module.weight]
    if output_module is not None and hasattr(output_module, "weight"):
        parameters.append(output_module.weight)

    unique_parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            unique_parameters.append(parameter)
            seen.add(id(parameter))

    vocab_size = int(input_module.weight.shape[0])
    if not 0 <= original_vocab_size < vocab_size:
        raise ValueError(
            f"original_vocab_size must be smaller than resized vocab: "
            f"{original_vocab_size} >= {vocab_size}"
        )

    for parameter in unique_parameters:
        if parameter.ndim < 2 or int(parameter.shape[0]) != vocab_size:
            raise ValueError("Input/output embedding vocab dimensions do not match")
        parameter.requires_grad = True
        _mask_old_token_rows(parameter, original_vocab_size)

    return {
        "new_token_count": vocab_size - original_vocab_size,
        "trainable_tensors": len(unique_parameters),
        "trainable_parameters": sum(parameter.numel() for parameter in unique_parameters),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def training_argument_values(
    *,
    device: torch.device,
    output_dir: str,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
    learning_rate: float,
    gradient_checkpointing: bool,
    seed: int,
) -> dict[str, Any]:
    is_mps = device.type == "mps"
    effective_gradient_checkpointing = gradient_checkpointing and not is_mps
    return {
        "output_dir": output_dir,
        "per_device_train_batch_size": micro_batch_size,
        "per_device_eval_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": num_epochs,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.03,
        "bf16": False,
        "fp16": False,
        "optim": "adamw_torch",
        "weight_decay": 0.0,
        "logging_steps": 1,
        "eval_strategy": "no" if is_mps else "epoch",
        "save_strategy": "no" if is_mps else "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": not is_mps,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "gradient_checkpointing": effective_gradient_checkpointing,
        "dataloader_pin_memory": device.type == "cuda",
        "report_to": "none",
        "use_cpu": device.type == "cpu",
        "seed": seed,
        "data_seed": seed,
    }


def _to_hf_dataset(dataset: Any, seed: int) -> HFDataset:
    if len(dataset) == 0:
        raise ValueError("Dataset is empty after sampling")
    first = dataset[0]
    converted = HFDataset.from_dict(
        {key: [dataset[index][key] for index in range(len(dataset))] for key in first}
    )
    return converted.shuffle(seed=seed)


def evaluate_loss_on_cpu(
    model: nn.Module,
    dataset: Any,
    data_collator: Any,
    batch_size: int = 1,
) -> float:
    """Evaluate loss on CPU to avoid the post-update MPS no-grad matmul crash."""

    model.to("cpu")
    model.eval()
    losses: list[float] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    with torch.no_grad():
        for batch in loader:
            cpu_batch = {
                key: value.to("cpu") if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            loss = model(**cpu_batch).loss
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite CPU evaluation loss: {loss.item()}")
            losses.append(float(loss.item()))
    if not losses:
        raise ValueError("Evaluation dataset is empty")
    return sum(losses) / len(losses)


def train(
    base_model: str,
    train_file: str,
    eval_file: str,
    output_dir: str,
    category: str,
    sid_index_path: str,
    item_meta_path: str,
    sample: int = 32,
    seed: int = 42,
    batch_size: int = 4,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
    cutoff_len: int = 128,
    freeze_LLM: bool = True,
    gradient_checkpointing: bool = False,
    device: str = "mps",
    dtype: str = "float32",
) -> dict[str, Any]:
    """Run a small single-process SFT job using an existing SID index."""

    required_paths = [train_file, eval_file, sid_index_path, item_meta_path]
    missing = [path for path in required_paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Required input files are missing: {missing}")
    if category not in CATEGORY_NAMES:
        raise ValueError(f"Unsupported category: {category}")
    if sample == 0 or sample < -1:
        raise ValueError("sample must be -1 or a positive integer")
    if micro_batch_size < 1 or batch_size < micro_batch_size:
        raise ValueError("batch_size must be >= micro_batch_size >= 1")
    if batch_size % micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size")

    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define an eos_token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=resolved_dtype,
        low_cpu_mem_usage=True,
    )
    original_vocab_size = len(tokenizer)
    sid_tokens = TokenExtender(sid_index_path).get_new_tokens()
    added_token_count = tokenizer.add_tokens(sid_tokens)
    if added_token_count != len(sid_tokens):
        raise ValueError(
            f"Expected to add {len(sid_tokens)} SID tokens, added {added_token_count}; "
            "base tokenizer already contains one or more SID tokens"
        )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    freeze_stats: dict[str, int] | None = None
    if freeze_LLM:
        freeze_stats = freeze_for_sid_training(model, original_vocab_size)
    effective_gradient_checkpointing = gradient_checkpointing and resolved_device.type != "mps"
    if gradient_checkpointing and not effective_gradient_checkpointing:
        warnings.warn(
            "Gradient checkpointing is disabled on MPS because the Trainer path can trigger "
            "an MPSGraph matmul shape crash.",
            RuntimeWarning,
        )
    if effective_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.to(resolved_device)

    # data.py 中的数据集会在临时 Pandas Series 上解析字符串字段；该操作不会写回
    # 原始 DataFrame，但会触发 SettingWithCopyWarning，因此在独立 MPS 入口中屏蔽它。
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    category_text = CATEGORY_NAMES[category]

    # 采用多任务 SFT，而不是只学习“历史 SID -> 下一个 SID”：
    # 1. SidSFTDataset 学习核心序列推荐任务；
    # 2. SidItemFeatDataset 建立商品标题与 SID 的双向语义映射；
    # 3. FusionSeqRecDataset 将 SID 交互历史与目标商品标题关联起来。
    # 三个数据集返回相同的 input_ids/attention_mask/labels 结构，因此可以串接。
    # 注意：sample 会分别作用于每个子数据集；sample=8 时总训练量约为 24 条。
    train_data = ConcatDataset(
        [
            # 核心任务：历史 SID 序列 -> 下一个 item 的 SID。
            SidSFTDataset(
                train_file=train_file,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=sample,
                seed=seed,
                category=category_text,
            ),
            # 辅助任务：item title -> SID，以及 SID -> item title。
            SidItemFeatDataset(
                item_file=item_meta_path,
                index_file=sid_index_path,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=sample,
                seed=seed,
                category=category_text,
            ),
            # 融合任务：历史 SID 序列 -> 下一个 item 的自然语言标题。
            FusionSeqRecDataset(
                train_file=train_file,
                item_file=item_meta_path,
                index_file=sid_index_path,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=sample,
                seed=seed,
                category=category_text,
            ),
        ]
    )

    # 验证集只保留核心推荐任务，使 eval_loss 明确反映“下一 item SID 预测”，
    # 避免 title/SID 映射和标题生成任务的不同难度稀释该指标。
    eval_data = SidSFTDataset(
        train_file=eval_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category_text,
    )

    # Trainer 使用 Hugging Face Dataset；转换后统一按 seed 打乱，保证可复现。
    hf_train_dataset = _to_hf_dataset(train_data, seed)
    hf_eval_dataset = _to_hf_dataset(eval_data, seed)

    gradient_accumulation_steps = batch_size // micro_batch_size
    argument_values = training_argument_values(
        device=resolved_device,
        output_dir=output_dir,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        gradient_checkpointing=effective_gradient_checkpointing,
        seed=seed,
    )
    training_args = transformers.TrainingArguments(**argument_values)
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    )
    trainer = transformers.Trainer(
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_eval_dataset,
        args=training_args,
        data_collator=data_collator,
        callbacks=(
            []
            if resolved_device.type == "mps"
            else [EarlyStoppingCallback(early_stopping_patience=3)]
        ),
    )

    train_result = trainer.train()
    if resolved_device.type == "mps":
        torch.mps.synchronize()
        eval_loss = evaluate_loss_on_cpu(
            trainer.model,
            hf_eval_dataset,
            data_collator,
            batch_size=micro_batch_size,
        )
        torch.mps.empty_cache()
    else:
        eval_loss = float(trainer.evaluate()["eval_loss"])

    trainer.save_state()
    final_checkpoint = Path(output_dir) / "final_checkpoint"
    trainer.model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)

    result = {
        "device": str(resolved_device),
        "dtype": str(resolved_dtype),
        "sid_token_count": len(sid_tokens),
        "train_examples": len(hf_train_dataset),
        "eval_examples": len(hf_eval_dataset),
        "final_checkpoint": str(final_checkpoint),
        "train_metrics": train_result.metrics,
        "eval_loss": eval_loss,
    }
    if freeze_stats is not None:
        result["freeze_stats"] = freeze_stats
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    fire.Fire(train)
