"""使用 TRL + PEFT 对已过滤的 conversational SFT 数据训练 LoRA。"""

import argparse
import json
from pathlib import Path
from typing import Any

from .chat_template import (
    CHAT_TEMPLATE_KWARGS,
    default_chat_template_path,
    install_chat_template,
    template_protocol,
)
from .sft_labels import build_assistant_labeled_examples


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row.get("messages"), list) for row in rows):
        raise ValueError(f"{path} 不是由 airline_agent.sft_data 生成的非空 SFT 数据")
    return rows


def _training_dependencies() -> tuple[Any, ...]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from trl import SFTConfig, SFTTrainer
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "缺少训练依赖。请先在训练环境安装 requirements-train.txt，"
            "不要在当前 API rollout 环境中盲目安装。"
        ) from error
    return torch, Dataset, LoraConfig, TaskType, AutoTokenizer, (SFTConfig, SFTTrainer)


def _dataset(
    rows: list[dict[str, Any]],
    dataset_class: Any,
    tokenizer: Any,
    max_length: int,
    chat_template_kwargs: dict[str, Any],
    *,
    split_name: str,
) -> Any:
    from tqdm.auto import tqdm

    examples: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=f"构建 {split_name} Action labels", unit="trajectory"):
        row_examples = build_assistant_labeled_examples(
            row["messages"],
            tokenizer,
            max_length=max_length,
            chat_template_kwargs=chat_template_kwargs,
        )
        if not row_examples:
            raise ValueError(f"task_id={row.get('task_id')} 缺少 assistant labels")
        # 多轮 Agent 的 Action 长度差异很大。预计算长度交给 Trainer 分桶，
        # 避免长短样本在同一 batch 中互相 padding。
        for example in row_examples:
            example["length"] = len(example["input_ids"])
        examples.extend(row_examples)
    print(f"{split_name}: {len(rows)} 条轨迹展开为 {len(examples)} 个 Action 样本", flush=True)
    return dataset_class.from_list(examples)


def train(args: argparse.Namespace) -> None:
    train_rows = _load_rows(args.train_data)
    validation_rows = _load_rows(args.validation_data) if args.validation_data else []
    torch, Dataset, LoraConfig, TaskType, AutoTokenizer, trl_classes = _training_dependencies()
    SFTConfig, SFTTrainer = trl_classes
    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA；LoRA SFT 必须在可见 GPU 的训练环境运行")

    use_bf16 = bool(torch.cuda.is_bf16_supported())
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    template = install_chat_template(tokenizer, args.chat_template_file)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("正在按统一 Chat Template 构建 assistant-only labels...", flush=True)
    train_dataset = _dataset(
        train_rows,
        Dataset,
        tokenizer,
        args.max_length,
        CHAT_TEMPLATE_KWARGS,
        split_name="train",
    )
    validation_dataset = (
        _dataset(
            validation_rows,
            Dataset,
            tokenizer,
            args.max_length,
            CHAT_TEMPLATE_KWARGS,
            split_name="validation",
        )
        if validation_rows
        else None
    )
    config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        # 数据已经显式包含 labels；不能依赖原始 Qwen tokenizer 是否提供
        # generation-mask 模板。
        assistant_only_loss=False,
        gradient_checkpointing=args.gradient_checkpointing,
        # batch 内长度更接近，减少 padding 造成的注意力计算浪费；
        # 不改变 labels、有效 batch size 或优化目标。
        group_by_length=True,
        length_column_name="length",
        # ``model`` 传路径时由 TRL 调用 from_pretrained；显式 BF16 避免基座
        # 因默认 FP32 加载而无谓占用两倍权重显存。
        model_init_kwargs={
            "torch_dtype": "bfloat16" if use_bf16 else "float16",
            "attn_implementation": "sdpa",
        },
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if validation_rows else "no",
        load_best_model_at_end=bool(validation_rows),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[item for item in args.target_modules.split(",") if item],
    )
    trainer = SFTTrainer(
        model=args.model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    print("数据和模型已就绪，开始 SFT 训练。", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(args.output_dir))
    trainer.processing_class.save_pretrained(str(args.output_dir))
    _write_protocol(args.output_dir, template)

    if args.merged_output_dir:
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(args.merged_output_dir), safe_serialization=True)
        trainer.processing_class.save_pretrained(str(args.merged_output_dir))
        _write_protocol(args.merged_output_dir, template)


def _write_protocol(output_dir: Path, template: str) -> None:
    """保存 checkpoint 的模板契约，防止 veRL 误用不兼容的 SFT 模型。"""

    (output_dir / "airline_sft_protocol.json").write_text(
        json.dumps(
            template_protocol(template),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="本地 Hugging Face Qwen3-1.7B 模型目录")
    parser.add_argument("--train-data", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/sft/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/sft-lora"))
    parser.add_argument("--merged-output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--max-length",
        type=int,
        default=20480,
        help="tokenizer 后的最大轨迹长度；当前 SFT v1 最长样本为 16,855 token。",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument(
        "--chat-template-file",
        type=Path,
        default=default_chat_template_path(),
        help="Airline 训练和 rollout 共用的 chat template 文件",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="已废弃：Airline JSON Agent 使用 non-thinking prefix-preserving 模板",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()
    if args.enable_thinking:
        raise SystemExit(
            "--enable-thinking 已废弃；请使用默认的 non-thinking prefix-preserving 模板"
        )
    train(args)


if __name__ == "__main__":
    main()
