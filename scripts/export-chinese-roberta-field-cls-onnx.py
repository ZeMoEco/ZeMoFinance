import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification


DEFAULT_MODEL_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_field_cls_model"
DEFAULT_OUTPUT = r"E:\CamXAll\ZEMO\Data\model\finance_field_cls_model\finance_field_cls.onnx"


class FieldClassifierExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, features):
        input_ids = features[:, 0, 0, :]
        attention_mask = features[:, 1, 0, :]
        token_type_ids = features[:, 2, 0, :]
        output = self.model(
            input_ids=input_ids.to(torch.long),
            attention_mask=attention_mask.to(torch.long),
            token_type_ids=token_type_ids.to(torch.long),
        )
        return output.logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export uer/chinese_roberta_L-4_H-256 field classifier to ONNX."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-labels", type=int, default=0)
    return parser.parse_args()


def resolve_num_labels(model_dir: Path, explicit_num_labels: int) -> int:
    if explicit_num_labels > 0:
        return explicit_num_labels

    label_map_paths = [model_dir / "label_map.json", model_dir / "slm" / "label_map.json"]
    for label_map_path in label_map_paths:
        if not label_map_path.exists():
            continue
        with open(label_map_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        labels = data.get("labels")
        if isinstance(labels, list) and len(labels) > 0:
            return len(labels)

    config = AutoConfig.from_pretrained(str(model_dir))
    if getattr(config, "num_labels", None):
        return int(config.num_labels)

    return 8


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}. "
            "Download it with: huggingface-cli download uer/chinese_roberta_L-4_H-256 --local-dir <model-dir>"
        )

    num_labels = resolve_num_labels(model_dir, args.num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=num_labels,
    )
    model = model.float().eval()

    wrapper = FieldClassifierExportWrapper(model).eval()
    # Use one 4D float input for Harmony MindSpore Lite ArkTS. Multi-input
    # 2D RoBERTa graphs can load but may expose an empty getInputs() list.
    features = torch.zeros((1, 3, 1, args.seq_len), dtype=torch.float32)
    features[:, 0, 0, :] = 1.0
    features[:, 1, 0, :] = 1.0

    with torch.no_grad():
        logits = wrapper(features)
    print(f"Dry run logits shape: {tuple(logits.shape)}")

    torch.onnx.export(
        wrapper,
        (features,),
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=False,
    )
    print(f"ONNX exported: {output_path}")
    print(f"num_labels={num_labels}")


if __name__ == "__main__":
    main()
