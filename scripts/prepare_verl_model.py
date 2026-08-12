"""创建一个只供 veRL 使用的 tokenizer 兼容模型视图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _token_name(token: str, index: int) -> str:
    if token.startswith("<|") and token.endswith("|>"):
        return token[2:-2]
    return f"extra_token_{index}"


def _normalized_extra_special_tokens(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(token) for key, token in value.items()}
    if isinstance(value, list) and all(isinstance(token, str) for token in value):
        return {
            _token_name(token, index): token
            for index, token in enumerate(value)
        }
    raise ValueError("tokenizer_config.json 的 extra_special_tokens 格式不受支持")


def prepare(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if source == output:
        raise ValueError("veRL 兼容视图不能覆盖原始 SFT 模型目录")
    if not source.is_dir():
        raise ValueError(f"SFT 模型目录不存在：{source}")

    tokenizer_config_path = source / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    tokenizer_config["extra_special_tokens"] = _normalized_extra_special_tokens(
        tokenizer_config.get("extra_special_tokens", {})
    )

    output.mkdir(parents=True, exist_ok=True)
    for source_entry in source.iterdir():
        if source_entry.name == "tokenizer_config.json":
            continue
        target = output / source_entry.name
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source_entry.resolve():
                continue
            raise ValueError(f"兼容视图存在冲突文件：{target}")
        target.symlink_to(source_entry.resolve(), target_is_directory=source_entry.is_dir())

    (output / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"source": str(source), "output": str(output), "shared_weights": True},
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == "__main__":
    main()
