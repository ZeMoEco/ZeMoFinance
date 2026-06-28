import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_field_cls_model"
DEFAULT_OCR_LOG = r"C:\Users\zoneX\.codex\attachments\3ef7f8ce-72aa-42db-9389-11473cd373c8\pasted-text.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="Probe finance field classifier with OCR log lines.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--ocr-log", default=DEFAULT_OCR_LOG)
    return parser.parse_args()


def parse_ocr_items(path: Path):
    pattern = re.compile(r"rec#(\d+)/(\d+) box=(\d+),(\d+),(\d+),(\d+), confidence=([^,]+), text=(.*?)  at ")
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            matched = pattern.search(line)
            if matched is None:
                continue
            items.append({
                "idx": int(matched.group(1)),
                "x": int(matched.group(3)),
                "y": int(matched.group(4)),
                "text": matched.group(8).strip(),
            })
    return items


def column_hint(index: int) -> str:
    if index in [33, 39, 45]:
        return "名称列"
    if index in [34, 40, 46]:
        return "盈亏列"
    if index in [37, 43, 48]:
        return "市值列"
    if index in [38, 44, 49]:
        return "盈亏率列"
    if index in [35, 41, 50]:
        return "持仓列"
    if index in [36, 42, 47]:
        return "可用列/证券代码列"
    return "未知列"


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    ocr_log = Path(args.ocr_log)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    items = parse_ocr_items(ocr_log)
    target_indexes = {33, 34, 35, 36, 37, 38, 39, 40, 42, 45, 46, 47, 48, 49, 50}
    header = "名称 代码 市值 盈亏 盈亏率 持仓 可用 证券代码"
    row_text = (
        "纳斯达克 5.61 1000 1000 513300 2752.00 0.204% "
        "纳100ETF 28.28 1000 1000 159696 2078.00 1.380% "
        "恒指科技 -11.60 2200 513180 1276.00 -0.901%"
    )

    with torch.no_grad():
        for item in items:
            if item["idx"] not in target_indexes:
                continue
            text = (
                f"页面: 理财持仓; 表头: {header}; 行文本: {row_text}; "
                f"OCR单元格: {item['text']}; 区域/列: {column_hint(item['idx'])}"
            )
            encoded = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            probs = torch.softmax(model(**encoded).logits[0], dim=-1)
            top = torch.topk(probs, 3)
            print(json.dumps({
                "idx": item["idx"],
                "cell": item["text"],
                "hint": column_hint(item["idx"]),
                "pred": labels[int(top.indices[0])],
                "top3": [[labels[int(i)], round(float(v), 3)] for v, i in zip(top.values, top.indices)],
            }, ensure_ascii=False))


if __name__ == "__main__":
    main()
