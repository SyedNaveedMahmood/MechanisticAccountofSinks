"""Offline-only fixtures for the E3/E4/E5 NNsight smoke programs.

Nothing in this module is imported by production experiment entry points.  The smoke
programs create random GPT-2-compatible repositories and a tiny local tokenizer under a
TemporaryDirectory, so NNsight exercises its real local-model loading and tracing path
without a model/dataset download.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


def create_tiny_checkpoint(path: Path, seed: int, *, n_positions: int = 32,
                           n_layer: int = 4, n_head: int = 4,
                           n_embd: int = 32, vocab_size: int = 64):
    """Write one deterministic random checkpoint plus repository-style tokenizer."""
    path.mkdir(parents=True, exist_ok=True)
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    vocab.update({f"t{index}": index + 3 for index in range(vocab_size - 3)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        eos_token="[EOS]")
    tokenizer.save_pretrained(path)

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=len(tokenizer), n_positions=n_positions, n_ctx=n_positions,
        n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        attn_implementation="eager")
    model = GPT2LMHeadModel(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    del model
    return tokenizer


def synthetic_text(length: int, offset: int = 0) -> str:
    return " ".join(f"t{3 + ((offset + index) % 50)}" for index in range(length))


def synthetic_records(length: int = 16, examples_per_domain: int = 1):
    records = {}
    for domain_index, domain in enumerate(("sst2", "gsm8k", "humaneval")):
        rows = []
        for example_id in range(examples_per_domain):
            ids = [3 + ((domain_index * 11 + example_id * 7 + pos) % 50)
                   for pos in range(length)]
            rows.append({
                "dataset": domain,
                "example_id": example_id,
                "input_ids": ids,
                "text": synthetic_text(length, domain_index * 11 + example_id * 7),
                "source_indices": [example_id],
            })
        records[domain] = rows
    return records


def assert_nnsight_metadata(config: dict) -> None:
    assert config["engine_name"] == "nnsight"
    engine = config["engine"]
    required = {
        "name", "nnsight_version", "transformers_version", "torch_version",
        "model_name", "model_revision", "dtype", "device", "execution_location",
        "attn_implementation", "attention_probability_source", "layer_band",
        "intervention_registry_version",
    }
    missing = sorted(required.difference(engine))
    assert not missing, f"missing NNsight provenance keys: {missing}"
    assert engine["name"] == "nnsight"
    assert engine["attn_implementation"] == "eager"
