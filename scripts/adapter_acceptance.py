"""Real CUDA qualification; writes only to an explicitly isolated output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_learning_assistant.product.finetune import train, write_json
from self_learning_assistant.prompts import DEFAULT_CHAT_SYSTEM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-context", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    sample = {
        "prompt": "What response style does the user prefer?",
        "target": "The user prefers short, gentle answers.",
    }
    token_length = 0
    if args.long_context:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)  # type: ignore[no-untyped-call]

        def length(padding: int) -> int:
            text = processor.apply_chat_template(
                [
                    {"role": "system", "content": DEFAULT_CHAT_SYSTEM},
                    {
                        "role": "user",
                        "content": "context " * padding + sample["prompt"],
                    },
                    {"role": "assistant", "content": sample["target"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            return len(processor.tokenizer(text, add_special_tokens=False)["input_ids"])

        lower, upper = 0, 512
        while lower < upper:
            midpoint = (lower + upper + 1) // 2
            if length(midpoint) <= 512:
                lower = midpoint
            else:
                upper = midpoint - 1
        token_length = length(lower)
        sample["prompt"] = "context " * lower + sample["prompt"]
    samples = [sample] * 8
    paused = train(
        args.model_dir,
        args.output,
        samples,
        max_steps=2,
        stopped=lambda: (args.output / "progress.json").exists(),
    )
    assert paused["state"] == "paused" and paused["step"] == 1
    assert (args.output / "checkpoint.pt").is_file()
    result = train(args.model_dir, args.output, samples, max_steps=2)
    assert result["state"] == "trained" and result["step"] == 2
    result["input_tokens"] = token_length
    from self_learning_assistant.model.transformers import TransformersModel

    model = TransformersModel(
        args.model_dir,
        adapter_dir=args.output,
        device="cuda",
        load_in_4bit=True,
        enable_thinking=False,
    )
    answer = model.generate(
        [{"role": "user", "content": "用中文简短解释连接池。"}], max_new_tokens=96
    )
    assert answer.strip()
    result["reload_answer"] = answer
    result["qualification"] = "passed"
    write_json(args.output / "qualification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
