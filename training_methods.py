"""Utility to prepare GPT-OSS 20B model for different finetuning strategies.

This module exposes a CLI that loads the GPT-OSS 20B model using HuggingFace
`transformers` and adapts it for a selected finetuning method:

* full:      Standard full-parameter finetuning.
* freeze:    Freeze base model weights and only train the final layer.
* lora:      Low-Rank Adaption (LoRA).
* qlora:     Quantized LoRA using 4-bit loading via `bitsandbytes`.
* oft:       Orthogonal Finetuning (OFT).
* qoft:      Quantized OFT (4-bit) variant.

The function ``prepare_model`` returns a model ready for training according to
selected method.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import (
        LoraConfig,
        OFTConfig,
        get_peft_model,
    )
except Exception:  # pragma: no cover - best effort if ``peft`` missing
    LoraConfig = OFTConfig = None
    get_peft_model = None


@dataclass
class ModelAndTokenizer:
    model: "AutoModelForCausalLM"
    tokenizer: "AutoTokenizer"


def prepare_model(model_name: str, method: str) -> ModelAndTokenizer:
    """Load model and adapt it for a given finetuning ``method``.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
    method:
        One of ``full``, ``freeze``, ``lora``, ``qlora``, ``oft`` or ``qoft``.

    Returns
    -------
    ModelAndTokenizer
        Tuple dataclass containing adapted model and tokenizer.
    """

    method = method.lower()

    # --- Load base model -------------------------------------------------
    quantization_config: Optional[BitsAndBytesConfig] = None
    load_in_4bit = method in {"qlora", "qoft"}
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Full finetuning -------------------------------------------------
    if method == "full":
        return ModelAndTokenizer(model, tokenizer)

    # --- Freeze base model ----------------------------------------------
    if method == "freeze":
        for param in model.base_model.parameters():
            param.requires_grad = False
        # still train the final lm head
        return ModelAndTokenizer(model, tokenizer)

    # Remaining methods rely on PEFT
    if get_peft_model is None:
        raise RuntimeError("peft library is required for method '%s'" % method)

    if method == "lora":
        config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
        model = get_peft_model(model, config)
        return ModelAndTokenizer(model, tokenizer)

    if method == "qlora":
        config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1)
        model = get_peft_model(model, config)
        return ModelAndTokenizer(model, tokenizer)

    if method == "oft":
        config = OFTConfig(r=8)
        model = get_peft_model(model, config)
        return ModelAndTokenizer(model, tokenizer)

    if method == "qoft":
        config = OFTConfig(r=16)
        model = get_peft_model(model, config)
        return ModelAndTokenizer(model, tokenizer)

    raise ValueError(f"Unknown method: {method}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="gptoss-20b", help="HuggingFace model name")
    parser.add_argument(
        "--method",
        required=True,
        choices=["full", "freeze", "lora", "qlora", "oft", "qoft"],
        help="Finetuning method to apply",
    )
    args = parser.parse_args()

    prepare_model(args.model_name, args.method)
    print(f"Loaded {args.model_name} with method '{args.method}'")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
