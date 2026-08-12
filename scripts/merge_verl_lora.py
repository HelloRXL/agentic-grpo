"""将 veRL 导出的基础权重和 LoRA adapter 合并为可部署的 HF 模型。"""

from __future__ import annotations

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_model(source: Path, output: Path, protocol_source: Path | None = None) -> None:
    adapter = source / "lora_adapter"
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"基础模型不存在: {source / 'config.json'}")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"LoRA adapter 不存在: {adapter}")
    output.mkdir(parents=True, exist_ok=True)

    # veRL merger 生成的是基础 HF 权重；PEFT 负责把训练得到的增量参数合入。
    base = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    merged = model.merge_and_unload()
    merged.save_pretrained(output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    tokenizer.save_pretrained(output)
    # FSDP checkpoint 附带的是 veRL 序列化时的 tokenizer 视图。部署评测必须
    # 恢复与 SFT/rollout 对齐的项目模板，避免多轮历史 Assistant token 漂移。
    metadata_source = protocol_source or source
    for name in ("chat_template.jinja", "airline_sft_protocol.json"):
        source_file = metadata_source / name
        if source_file.is_file():
            (output / name).write_bytes(source_file.read_bytes())
    print(f"merged_model={output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol-source",
        type=Path,
        default=None,
        help="提供经验证的 SFT 模型目录，用其 chat template 与协议文件覆盖 veRL 视图。",
    )
    args = parser.parse_args()
    merge_model(args.source, args.output, args.protocol_source)


if __name__ == "__main__":
    main()
