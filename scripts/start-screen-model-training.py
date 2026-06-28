import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "model" / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
OUT_ROOT = ROOT / "model" / "screen_starter"


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize(text: str):
    return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())


def label_for_row(row: dict) -> str:
    source = row.get("source", "")
    if "guiact" in source.lower():
        out = json.loads(row.get("output", "{}"))
        action = out.get("next_action", "")
        if action.startswith("tap"):
            return "phone_action_tap"
        if action.startswith("scroll"):
            return "phone_action_scroll"
        if action.startswith("type"):
            return "phone_action_type"
        return "phone_action_other"
    if "mobile-actions" in source:
        out = json.loads(row.get("output", "{}"))
        return "tool_" + out.get("tool", "unknown")
    return "screen_summary"


def train_nb(rows):
    labels = Counter()
    token_counts = defaultdict(Counter)
    vocab = set()
    for row in rows:
        label = label_for_row(row)
        text = row.get("instruction", "") + "\n" + row.get("output", "")
        tokens = tokenize(text)
        labels[label] += 1
        token_counts[label].update(tokens)
        vocab.update(tokens)

    vocab_size = max(1, len(vocab))
    total = max(1, sum(labels.values()))
    model = {
        "labels": dict(labels),
        "vocab_size": vocab_size,
        "total": total,
        "token_counts": {label: dict(counts) for label, counts in token_counts.items()},
    }
    return model


def predict(model, text: str):
    tokens = tokenize(text)
    scores = {}
    total = model["total"]
    vocab_size = model["vocab_size"]
    for label, label_count in model["labels"].items():
        counts = model["token_counts"].get(label, {})
        denom = sum(counts.values()) + vocab_size
        score = (label_count + 1) / (total + len(model["labels"]))
        for token in tokens:
            score *= (counts.get(token, 0) + 1) / denom
        scores[label] = score
    return max(scores.items(), key=lambda item: item[1])[0] if scores else "unknown"


def main():
    rows = []
    for name in ["screen_action_sft.jsonl", "mobile_actions_tool_sft.jsonl", "screen_summary_text_only.jsonl"]:
        rows.extend(load_jsonl(PROCESSED_ROOT / name))
    if not rows:
        raise SystemExit("没有训练数据，请先运行 scripts\\prepare-screen-omni-data.py")

    random.seed(42)
    random.shuffle(rows)
    split = max(1, int(len(rows) * 0.9))
    train_rows = rows[:split]
    eval_rows = rows[split:]
    model = train_nb(train_rows)

    correct = 0
    for row in eval_rows:
        pred = predict(model, row.get("instruction", ""))
        if pred == label_for_row(row):
            correct += 1
    acc = correct / max(1, len(eval_rows))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "starter_nb_model.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    (OUT_ROOT / "train_report.json").write_text(json.dumps({
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "eval_accuracy": acc,
        "labels": model["labels"],
        "note": "这是数据链路烟测用的轻量文本基线，不是最终 VLM。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "eval_accuracy": acc,
        "out": str(OUT_ROOT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
