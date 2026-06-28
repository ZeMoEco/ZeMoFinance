import argparse
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "model" / "screen_omni_lite" / "screen_omni_model.min.json"


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\[\[ocr:[^\]]+\]\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str):
    text = normalize_text(text)
    tokens = re.findall(r"[a-z_]+|[0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]", text)
    chars = [t for t in tokens if len(t) == 1 and "\u4e00" <= t <= "\u9fff"]
    return tokens + [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def contains_any(text: str, keywords):
    return any(k.lower() in text for k in keywords)


def rule_predict(model: dict, text: str) -> str:
    value = normalize_text(text)
    patterns = model["slot_patterns"]
    investment_hits = sum(1 for k in patterns["investment_import"]["keywords"] if k.lower() in value)
    if investment_hits >= 2:
        return "investment_import"
    bill = patterns["bill_record"]
    if re.search(bill["amount"], value) and (
        contains_any(value, bill["income_keywords"])
        or contains_any(value, bill["expense_keywords"])
        or contains_any(value, bill["source_keywords"])
    ):
        return "bill_record"
    chat = patterns["chat_todo"]
    todo_hits = sum(1 for k in chat["keywords"] if k.lower() in value)
    has_time = re.search(chat["time"], value) is not None
    has_speaker = re.search(r"(^|\n).{1,12}[:：]\s*.{2,}", text or "") is not None
    has_chat_hint = contains_any(value, ["群聊", "聊天", "微信", "收到", "回复"])
    if todo_hits >= 1 and (has_time or has_speaker or has_chat_hint):
        return "chat_todo"
    return ""


def predict(model: dict, text: str) -> dict:
    ruled = rule_predict(model, text)
    if ruled:
        return {"label": ruled, "method": "rule", "meta": model["labels"][ruled]}

    tokens = tokenize(text)
    total = sum(model["label_counts"].values())
    vocab_size = max(1, int(model["vocab_size"]))
    best_label = ""
    best_score = -1e100
    for label, label_count in model["label_counts"].items():
        counts = model["token_counts"].get(label, {})
        denom = sum(counts.values()) + vocab_size
        score = math.log((label_count + 1.0) / (total + len(model["label_counts"])))
        for token in tokens:
            score += math.log((counts.get(token, 0) + 1.0) / denom)
        if score > best_score:
            best_label = label
            best_score = score
    return {"label": best_label, "method": "naive_bayes", "score": best_score, "meta": model["labels"][best_label]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--text", default="")
    parser.add_argument("--file", default="")
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    else:
        text = args.text
    if not text.strip():
        raise SystemExit("pass --text or --file")
    print(json.dumps(predict(model, text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
