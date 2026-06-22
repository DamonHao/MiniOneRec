#!/usr/bin/env python3
"""Run the original MiniOneRec evaluation protocol on Apple Silicon MPS."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from data import EvalSidDataset


CATEGORY_NAMES = {
    "Industrial_and_Scientific": "industrial and scientific items",
    "Office_Products": "office products",
    "Toys_and_Games": "toys and games",
    "Sports": "sports and outdoors",
    "Books": "books",
}


class ChunkedLinear(nn.Module):
    """Apply one Linear layer in output chunks without changing its weights."""

    def __init__(self, linear: nn.Linear, chunk_size: int):
        super().__init__()
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.linear = linear
        self.chunk_size = chunk_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight_chunks = self.linear.weight.split(self.chunk_size, dim=0)
        if self.linear.bias is None:
            bias_chunks = [None] * len(weight_chunks)
        else:
            bias_chunks = self.linear.bias.split(self.chunk_size, dim=0)
        return torch.cat(
            [
                F.linear(inputs, weight, bias)
                for weight, bias in zip(weight_chunks, bias_chunks, strict=True)
            ],
            dim=-1,
        )


class SidTokenTrie:
    """Represent the token paths accepted by the original SID constraint."""

    _TERMINAL = object()

    def __init__(self, sequences: Iterable[Sequence[int]], eos_token_id: int):
        self._root: dict[Any, Any] = {}
        self._eos_token_id = int(eos_token_id)
        sequence_count = 0
        for sequence in sequences:
            token_ids = [int(token_id) for token_id in sequence]
            if not token_ids:
                raise ValueError("SID token sequence must not be empty")
            node = self._root
            for token_id in token_ids:
                node = node.setdefault(token_id, {})
            node[self._TERMINAL] = True
            sequence_count += 1
        if sequence_count == 0:
            raise ValueError("SID token trie requires at least one sequence")

    def allowed_tokens(self, prefix: Sequence[int]) -> list[int]:
        if prefix and int(prefix[-1]) == self._eos_token_id:
            return [self._eos_token_id]
        node = self._root
        for token_id in prefix:
            child = node.get(int(token_id))
            if child is None:
                return []
            node = child
        allowed = sorted(key for key in node if key is not self._TERMINAL)
        if self._TERMINAL in node:
            allowed.append(self._eos_token_id)
        return allowed


def configure_tokenizer(tokenizer: Any) -> None:
    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise ValueError("tokenizer must define eos_token and eos_token_id")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"


def load_sid_catalog(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"SID info file not found: {path}")
    semantic_ids = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            sid = line.split("\t", 1)[0].strip()
            if not sid:
                raise ValueError(f"empty SID at {path}:{line_number}")
            semantic_ids.append(sid)
    if not semantic_ids:
        raise ValueError(f"SID info file is empty: {path}")
    return semantic_ids


def load_original_evaluation_data(
    test_file: Path,
    tokenizer: Any,
    category_text: str,
    seed: int,
    max_samples: int = -1,
) -> tuple[list[dict[str, list[int]]], list[dict[str, Any]]]:
    """Use EvalSidDataset exactly as evaluate.py does, then optionally slice for smoke tests."""

    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or a positive integer")
    dataset = EvalSidDataset(
        train_file=str(test_file),
        tokenizer=tokenizer,
        max_len=2560,
        category=category_text,
        test=True,
        K=0,
        seed=seed,
    )
    limit = len(dataset) if max_samples == -1 else min(max_samples, len(dataset))
    encodings = [dataset[index] for index in range(limit)]
    records = dataset.get_all()[:limit]
    return encodings, records


def build_original_sid_sequences(
    tokenizer: Any,
    semantic_ids: Sequence[str],
    base_model: str,
) -> list[list[int]]:
    """Reproduce evaluate.py's Response-prefix tokenization and prefix_index slicing."""

    response_text = [f"### Response:\n{sid}\n" for sid in semantic_ids]
    tokenized = [tokenizer(text).input_ids for text in response_text]
    if "llama" in base_model.lower():
        tokenized = [token_ids[1:] for token_ids in tokenized]
    prefix_index = 4 if "gpt2" in base_model.lower() else 3
    sequences = []
    for token_ids in tokenized:
        if len(token_ids) <= prefix_index:
            raise ValueError("SID response tokenization is shorter than prefix_index")
        sequences.append([*token_ids[prefix_index:], tokenizer.eos_token_id])
    return sequences


def constrained_beam_search(
    model: nn.Module,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    trie: SidTokenTrie,
    num_beams: int,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
    fixed_shape: bool = True,
) -> list[list[int]]:
    """Run constrained cumulative-log-probability search with fixed MPS tensor shapes."""

    if prompt_input_ids.ndim != 1 or prompt_attention_mask.ndim != 1:
        raise ValueError("prompt tensors must be one-dimensional")
    beams: list[tuple[list[int], float, bool]] = [([], 0.0, False)]

    for _step in range(max_new_tokens):
        active = [beam for beam in beams if not beam[2]]
        if not active:
            break
        input_rows = []
        mask_rows = []
        for tokens, _score, _finished in active:
            generated = torch.tensor(
                tokens,
                dtype=prompt_input_ids.dtype,
                device=prompt_input_ids.device,
            )
            combined = torch.cat((prompt_input_ids, generated))
            combined_mask = torch.cat((prompt_attention_mask, torch.ones_like(generated)))
            if fixed_shape:
                fixed_length = prompt_input_ids.shape[0] + max_new_tokens
                padding_length = fixed_length - combined.shape[0]
                combined = F.pad(combined, (0, padding_length), value=pad_token_id)
                combined_mask = F.pad(combined_mask, (0, padding_length), value=0)
            input_rows.append(combined)
            mask_rows.append(combined_mask)
        while len(input_rows) < num_beams:
            input_rows.append(input_rows[-1])
            mask_rows.append(mask_rows[-1])
        input_ids = torch.stack(input_rows)
        attention_mask = torch.stack(mask_rows)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if fixed_shape:
            last_token_index = int(attention_mask[0].sum()) - 1
            logits_to_keep: int | torch.Tensor = torch.tensor(
                [last_token_index], device=input_ids.device
            )
        else:
            logits_to_keep = 1

        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            ).logits[:, -1, :]

        candidates = [beam for beam in beams if beam[2]]
        for beam_index, (tokens, score, _finished) in enumerate(active):
            allowed_tokens = trie.allowed_tokens(tokens)
            if not allowed_tokens:
                raise RuntimeError(f"beam prefix is outside the SID catalog: {tokens}")
            log_probs = torch.log_softmax(logits[beam_index].float(), dim=-1)
            allowed = torch.tensor(allowed_tokens, device=log_probs.device)
            allowed_scores = log_probs.index_select(0, allowed)
            keep = min(num_beams, len(allowed_tokens))
            values, positions = torch.topk(allowed_scores, k=keep)
            for value, position in zip(values.tolist(), positions.tolist(), strict=True):
                token_id = allowed_tokens[position]
                candidates.append(
                    ([*tokens, token_id], score + value, token_id == eos_token_id)
                )
        beams = sorted(candidates, key=lambda beam: beam[1], reverse=True)[:num_beams]

    if len(beams) != num_beams:
        raise RuntimeError(f"beam search returned {len(beams)} candidates, expected {num_beams}")
    unfinished = [tokens for tokens, _score, finished in beams if not finished]
    if unfinished:
        raise RuntimeError(
            f"beam search did not finish within {max_new_tokens} tokens: {unfinished[:2]}"
        )
    return [tokens for tokens, _score, _finished in beams]


def format_original_prediction_records(
    records: Sequence[dict[str, Any]],
    predictions: Sequence[Sequence[str]],
) -> list[dict[str, Any]]:
    if len(records) != len(predictions):
        raise ValueError("prediction count does not match evaluation records")
    formatted = []
    for record, candidates in zip(records, predictions, strict=True):
        output = dict(record)
        output["predict"] = list(candidates)
        output.pop("dedup", None)
        formatted.append(output)
    return formatted


def calculate_original_metrics(
    records: Sequence[dict[str, Any]],
    valid_sids: set[str],
    topk: Sequence[int] = (1, 3, 5, 10, 20, 50),
) -> dict[str, Any]:
    """Reproduce calc.py, including its CC early-stop behavior."""

    if not records:
        raise ValueError("cannot calculate metrics for empty records")
    hit = {str(k): 0.0 for k in topk}
    ndcg = {str(k): 0.0 for k in topk}
    cc = 0
    for record in records:
        predictions = [str(value).strip('"\n').strip() for value in record["predict"]]
        raw_target = record["output"]
        if isinstance(raw_target, list):
            target = str(raw_target[0]).strip('"').strip()
        else:
            target = str(raw_target).strip(' \n"')
        rank = None
        for index, candidate in enumerate(predictions):
            if candidate not in valid_sids:
                cc += 1
            if candidate == target:
                rank = index + 1
                break
        for k in topk:
            if rank is not None and rank <= k:
                hit[str(k)] += 1.0
                ndcg[str(k)] += 1.0 / math.log2(rank + 1)
    count = len(records)
    return {
        "sample_count": count,
        "cc": cc,
        "hr": {key: value / count for key, value in hit.items()},
        "ndcg": {key: value / count for key, value in ndcg.items()},
    }


def parse_topk(raw_value: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw_value.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("topk must be a comma-separated list of integers") from error
    if not values or any(value < 1 for value in values):
        raise ValueError("topk values must be positive")
    if values != sorted(set(values)):
        raise ValueError("topk values must be unique and strictly increasing")
    return values


def resolve_device(requested: str) -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        normalized = "mps" if torch.backends.mps.is_available() else "cpu"
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, mps, cpu")


def resolve_dtype(requested: str, device: torch.device) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    try:
        dtype = mapping[requested.lower()]
    except KeyError as error:
        raise ValueError("dtype must be one of: float32, float16, bfloat16") from error
    if device.type == "mps" and dtype == torch.bfloat16:
        raise ValueError("bfloat16 is not supported by this MPS evaluation path")
    return dtype


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.category not in CATEGORY_NAMES:
        raise ValueError(f"unsupported category: {args.category}")
    if args.batch_size < 1 or args.num_beams < 1 or args.max_new_tokens < 1:
        raise ValueError("batch_size, num_beams and max_new_tokens must be positive")
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("max_samples must be -1 or a positive integer")
    if args.lm_head_chunk_size < 1:
        raise ValueError("lm_head_chunk_size must be positive")
    topk = parse_topk(args.topk)
    if max(topk) > args.num_beams:
        raise ValueError("all topk values must be <= num_beams")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    configure_tokenizer(tokenizer)
    catalog = load_sid_catalog(args.info_file)
    sequences = build_original_sid_sequences(tokenizer, catalog, args.base_model)
    trie = SidTokenTrie(sequences, tokenizer.eos_token_id)
    encodings, base_records = load_original_evaluation_data(
        test_file=args.test_data_path,
        tokenizer=tokenizer,
        category_text=CATEGORY_NAMES[args.category],
        seed=args.seed,
        max_samples=args.max_samples,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    if device.type == "mps":
        model.lm_head = ChunkedLinear(model.lm_head, args.lm_head_chunk_size)
    model.eval()
    model.to(device)

    all_predictions = []
    for encoding in tqdm(encodings, desc="Evaluating"):
        completion_ids = constrained_beam_search(
            model=model,
            prompt_input_ids=torch.tensor(encoding["input_ids"], device=device),
            prompt_attention_mask=torch.tensor(encoding["attention_mask"], device=device),
            trie=trie,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            fixed_shape=device.type == "mps",
        )
        decode_kwargs = {"skip_special_tokens": True}
        if "llama" in args.base_model.lower():
            decode_kwargs["clean_up_tokenization_spaces"] = False
        decoded = tokenizer.batch_decode(completion_ids, **decode_kwargs)
        all_predictions.append(
            [value.split("Response:\n")[-1].strip() for value in decoded]
        )

    records = format_original_prediction_records(base_records, all_predictions)
    write_json(args.result_json_data, records)
    metrics = calculate_original_metrics(records, set(catalog), topk)
    metrics.update(
        {
            "model_path": args.base_model,
            "device": str(device),
            "dtype": args.dtype,
            "num_beams": args.num_beams,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "length_penalty": args.length_penalty,
            "seed": args.seed,
        }
    )
    write_json(args.metrics_path, metrics)
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--info_file", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--test_data_path", type=Path, required=True)
    parser.add_argument("--result_json_data", type=Path, required=True)
    parser.add_argument("--metrics_path", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length_penalty", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--device", default="mps", choices=("auto", "mps", "cpu"))
    parser.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--lm_head_chunk_size", type=int, default=8192)
    parser.add_argument("--topk", default="1,3,5,10,20,50")
    parser.add_argument("--max_samples", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
