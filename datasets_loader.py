# -*- coding: utf-8 -*-
"""datasets_loader.py — Dataset loading and sampling for intervention_analysis.py.

Three standard benchmarks are used:
  - SST-2       Natural language sentences (Stanford Sentiment Treebank).
  - GSM8K       Grade-school math word problems.
  - HumanEval   Python programming prompts.

Examples with fewer than `cut_length` tokens are discarded.
All kept examples are truncated to exactly `cut_length` tokens.
`sample_size` examples are then drawn with a fixed random seed.
"""

import random
import numpy as np
from datasets import load_dataset

# ─── Dataset registry ────────────────────────────────────────────────────────

DATASET_SPECS = [
    {
        "name": "sst2",
        "hf_path": "stanfordnlp/sst2",
        "config": None,
        "split": "train",
        "text_field": "sentence",
        "kind": "natural_language",
    },
    {
        "name": "gsm8k",
        "hf_path": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "text_field": "question",
        "kind": "math",
    },
    {
        "name": "humaneval",
        "hf_path": "openai/openai_humaneval",
        "config": None,
        "split": "test",
        "text_field": "prompt",
        "kind": "code",
    },
]

DEFAULT_SAMPLE_SIZE = 100
DEFAULT_CUT_LENGTH  = 40
DEFAULT_SEED        = 42

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_one_dataset(spec):
    kwargs = {}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    return load_dataset(spec["hf_path"], **kwargs, split=spec["split"])


def _normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def _truncate(tokenizer, text, cut_length):
    """Return *text* truncated to exactly *cut_length* GPT-2 tokens."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[:cut_length])


# ─── Public API ──────────────────────────────────────────────────────────────

def verify_datasets(tokenizer,
                    sample_size=DEFAULT_SAMPLE_SIZE,
                    cut_length=DEFAULT_CUT_LENGTH):
    """Check that every dataset has at least `sample_size` examples with
    >= `cut_length` tokens (i.e. that can be truncated to the target length).

    Prints a summary table and raises ``ValueError`` for any shortfall.
    """
    print("Verifying datasets...")
    print(f"  Cut length: {cut_length} tokens (examples shorter than this are discarded)")
    print(f"  Required per dataset: {sample_size}\n")

    ok = True
    for spec in DATASET_SPECS:
        print(f"  Loading {spec['name']} ({spec['hf_path']}) ...")
        ds = _load_one_dataset(spec)
        count = 0
        for ex in ds:
            text = _normalize_text(ex.get(spec["text_field"], ""))
            if not text:
                continue
            tlen = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if tlen >= cut_length:
                count += 1

        status = "OK" if count >= sample_size else "FAIL"
        print(f"    [{status}] {count} usable examples  (need {sample_size})")

        if count < sample_size:
            ok = False

    if not ok:
        raise ValueError(
            "One or more datasets do not have enough usable examples. "
            "Reduce --sample-size or --cut-length."
        )
    print("\nAll datasets verified OK.\n")


def sample_benchmark_datasets(tokenizer,
                               sample_size=DEFAULT_SAMPLE_SIZE,
                               cut_length=DEFAULT_CUT_LENGTH,
                               seed=DEFAULT_SEED):
    """Load, filter, truncate, and sample from all three benchmark datasets.

    Only examples with at least `cut_length` GPT-2 tokens are kept.
    All selected examples are then decoded back to text after truncation to
    exactly `cut_length` tokens, so every example in the returned corpus has
    the same sequence length.

    Parameters
    ----------
    tokenizer   : GPT2Tokenizer
    sample_size : int  — examples to draw from each dataset.
    cut_length  : int  — token count threshold and truncation target.
    seed        : int  — random seed for reproducibility.

    Returns
    -------
    sampled       : dict[str, list[str]]
        dataset_name → list of truncated text strings (each == cut_length tokens).
    manifest_rows : list[dict]
        One row per sampled example (for audit / reproducibility CSV).
    """
    rng = random.Random(seed)
    sampled = {}
    manifest_rows = []

    for spec in DATASET_SPECS:
        print(f"Sampling {spec['name']} ({spec['hf_path']}) ...")
        ds = _load_one_dataset(spec)

        candidates = []
        for idx, ex in enumerate(ds):
            text = _normalize_text(ex.get(spec["text_field"], ""))
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= cut_length:
                truncated = tokenizer.decode(ids[:cut_length])
                candidates.append({
                    "dataset":               spec["name"],
                    "kind":                  spec["kind"],
                    "hf_path":               spec["hf_path"],
                    "split":                 spec["split"],
                    "source_index":          idx,
                    "text":                  truncated,
                    "original_token_length": len(ids),
                    "token_length":          cut_length,
                })

        if len(candidates) < sample_size:
            raise ValueError(
                f"Dataset '{spec['name']}' only has {len(candidates)} usable examples "
                f"(need {sample_size}). Run verify_datasets() first."
            )

        rng.shuffle(candidates)
        chosen = candidates[:sample_size]

        print(f"  Sampled {len(chosen)} examples  "
              f"(all truncated to {cut_length} tokens; "
              f"mean original length={np.mean([r['original_token_length'] for r in chosen]):.1f})")

        sampled[spec["name"]] = [r["text"] for r in chosen]
        manifest_rows.extend(chosen)

    return sampled, manifest_rows


def sample_long_benchmark_datasets(tokenizer, sample_size=DEFAULT_SAMPLE_SIZE,
                                   cut_length=1024, seed=DEFAULT_SEED,
                                   domains=None):
    """Build exact-length token sequences by deterministic within-domain concatenation.

    Unlike :func:`sample_benchmark_datasets`, this function returns token IDs so a
    decode/re-tokenize round trip cannot alter length.  Source examples are shuffled
    independently per domain, tokenized without special tokens, and concatenated only
    with examples from that domain.  Each returned record includes its source indices
    for a reproducible manifest.
    """
    if sample_size <= 0 or cut_length < 2:
        raise ValueError("sample_size must be positive and cut_length must be at least 2")
    wanted = set(domains or [s["name"] for s in DATASET_SPECS])
    unknown = wanted.difference(s["name"] for s in DATASET_SPECS)
    if unknown:
        raise ValueError(f"Unknown benchmark domain(s): {sorted(unknown)}")
    sampled, manifest = {}, []
    for domain_index, spec in enumerate(DATASET_SPECS):
        if spec["name"] not in wanted:
            continue
        ds = _load_one_dataset(spec)
        pieces = []
        for idx, ex in enumerate(ds):
            text = _normalize_text(ex.get(spec["text_field"], ""))
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids:
                pieces.append((idx, text, list(map(int, ids))))
        if not pieces:
            raise ValueError(f"Dataset '{spec['name']}' contains no tokenizable examples")
        # Use a cut-length-independent anchor order so example i has the same
        # natural 40-token prefix whether this function is asked for 40 or 1024
        # tokens.  This makes separately scheduled E5 modes pairable.
        anchors = [piece for piece in pieces if len(piece[2]) >= min(40, cut_length)]
        anchor_rng = random.Random(seed * 1009 + domain_index * 9176)
        anchor_rng.shuffle(anchors)
        if len(anchors) < sample_size:
            raise ValueError(
                f"Dataset '{spec['name']}' has only {len(anchors)} natural anchor examples "
                f"with at least {min(40, cut_length)} tokens (need {sample_size})"
            )
        records = []
        for example_id in range(sample_size):
            anchor = anchors[example_id]
            ids = list(anchor[2])
            source_indices, source_text = [anchor[0]], [anchor[1]]
            extension = list(pieces)
            extension_rng = random.Random(
                seed * 1000003 + domain_index * 9176 + example_id * 7919
            )
            extension_rng.shuffle(extension)
            cursor = 0
            while len(ids) < cut_length:
                if cursor and cursor % len(extension) == 0:
                    extension_rng.shuffle(extension)
                idx, text, part = extension[cursor % len(extension)]
                cursor += 1
                ids.extend(part)
                source_indices.append(idx)
                source_text.append(text)
            ids = ids[:cut_length]
            if len(ids) != cut_length:
                raise AssertionError("long-context sampler produced an inexact sequence")
            record = {
                "dataset": spec["name"], "example_id": example_id,
                "input_ids": ids, "text": tokenizer.decode(ids),
                "source_indices": source_indices,
                "source_component_count": len(source_indices),
            }
            records.append(record)
            manifest.append({
                "dataset": spec["name"], "kind": spec["kind"],
                "hf_path": spec["hf_path"], "split": spec["split"],
                "example_id": example_id, "source_indices": ";".join(map(str, source_indices)),
                "source_component_count": len(source_indices), "token_length": cut_length,
                "text": record["text"], "sampling": "within_domain_concatenation",
            })
        sampled[spec["name"]] = records
    if not sampled:
        raise ValueError("No benchmark domains were selected")
    return sampled, manifest


def build_degenerate_domains(tokenizer, natural_records, cut_length=DEFAULT_CUT_LENGTH,
                             seed=DEFAULT_SEED, domains=None):
    """Create exact-length dataset-free controls from token-ID natural examples.

    ``natural_records`` maps domains to records containing ``input_ids``.  Generated
    tokens exclude ``eos_token_id`` and use local NumPy generators only.
    """
    requested = domains or ["random_uniform", "random_zipf", "shuffled_natural", "repeat_token"]
    valid = {"random_uniform", "random_zipf", "shuffled_natural", "repeat_token"}
    unknown = set(requested).difference(valid)
    if unknown:
        raise ValueError(f"Unknown synthetic domain(s): {sorted(unknown)}")
    base = [r for records in natural_records.values() for r in records]
    if not base:
        raise ValueError("Cannot build degenerate domains from an empty natural sample")
    eos = tokenizer.eos_token_id
    vocab = np.arange(len(tokenizer), dtype=np.int64)
    if eos is not None:
        vocab = vocab[vocab != int(eos)]
    natural_tokens = np.asarray([
        t for r in base for t in r["input_ids"][:cut_length] if eos is None or t != eos
    ], dtype=np.int64)
    if natural_tokens.size == 0:
        raise ValueError("Natural sample has no allowed tokens")
    values, counts = np.unique(natural_tokens, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    result, manifest = {}, []
    for domain_no, name in enumerate(requested):
        rng = np.random.default_rng(seed * 10007 + domain_no * 7919)
        rows = []
        for i, source in enumerate(base):
            src = np.asarray(source["input_ids"][:cut_length], dtype=np.int64)
            if len(src) != cut_length:
                raise ValueError("Natural control inputs must already have exact cut_length")
            if name == "random_uniform":
                ids = rng.choice(vocab, size=cut_length, replace=True)
                meta = {"distribution": "uniform_allowed_vocabulary"}
            elif name == "random_zipf":
                ids = rng.choice(values, size=cut_length, replace=True, p=probs)
                meta = {"distribution": "empirical_natural_unigram"}
            elif name == "shuffled_natural":
                ids = src[rng.permutation(cut_length)]
                meta = {"distribution": "within_example_permutation"}
            else:
                allowed = src[src != eos] if eos is not None else src
                if len(allowed) == 0:
                    allowed = natural_tokens
                token = int(allowed[(seed + i) % len(allowed)])
                ids = np.full(cut_length, token, dtype=np.int64)
                meta = {"distribution": "repeat", "repeat_token_id": token}
            ids_list = list(map(int, ids))
            row = {"dataset": name, "example_id": i, "input_ids": ids_list,
                   "text": tokenizer.decode(ids_list), "source_domain": source["dataset"],
                   "source_example_id": source["example_id"], **meta}
            rows.append(row)
            manifest.append({k: v for k, v in row.items() if k != "input_ids"})
        result[name] = rows
    return result, manifest


def load_optional_flores(tokenizer, sample_size=DEFAULT_SAMPLE_SIZE,
                         cut_length=DEFAULT_CUT_LENGTH, seed=DEFAULT_SEED):
    """Load Bangla and Chinese FLORES-200 records, raising a readable RuntimeError.

    E5 catches this exception and records a skip reason, making multilingual data
    genuinely optional without concealing configuration or network failures.
    """
    configs = {"flores_bengali": "ben_Beng", "flores_chinese": "zho_Hans"}
    out, manifest = {}, []
    try:
        for domain_no, (name, lang) in enumerate(configs.items()):
            ds = load_dataset("facebook/flores", lang, split="devtest")
            candidates = []
            for idx, ex in enumerate(ds):
                text = _normalize_text(ex.get("sentence", ""))
                ids = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
                if len(ids) >= cut_length:
                    candidates.append((idx, ids[:cut_length]))
            rng = random.Random(seed * 1013 + domain_no)
            rng.shuffle(candidates)
            if len(candidates) < sample_size:
                raise ValueError(f"{lang} has {len(candidates)} usable examples; need {sample_size}")
            records = []
            for example_id, (source_index, ids) in enumerate(candidates[:sample_size]):
                ids = list(map(int, ids))
                record = {"dataset": name, "example_id": example_id, "input_ids": ids,
                          "text": tokenizer.decode(ids), "language_config": lang,
                          "source_index": source_index}
                records.append(record)
                manifest.append({**record, "input_ids": " ".join(map(str, ids)),
                                 "token_length": len(ids)})
            out[name] = records
    except Exception as exc:
        raise RuntimeError(f"Optional FLORES-200 loading failed: {type(exc).__name__}: {exc}") from exc
    return out, manifest
