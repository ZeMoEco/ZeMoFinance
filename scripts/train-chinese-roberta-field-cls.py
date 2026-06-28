import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_BASE_MODEL_DIR = r"E:\CamXAll\ZEMO\Data\model\chinese_roberta_L-4_H-256"
DEFAULT_TRAIN_FILE = r"E:\CamXAll\ZEMO\Data\model\finance_field_cls_train.jsonl"
DEFAULT_OUTPUT_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_field_cls_model"
DEFAULT_LABELS = (
    "merchant,counterparty,amount,income_amount,expense_amount,balance,date_time,"
    "payment_method,order_id,bank_card,transaction_type,status,asset_name,asset_code,"
    "market_value,profit,profit_rate,holding,available,quantity,price,net_value,shares,other"
)


class FieldDataset(Dataset):
    def __init__(self, rows, tokenizer, label2id, max_len):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        if "token_type_ids" not in item:
            item["token_type_ids"] = torch.zeros(self.max_len, dtype=torch.long)
        item["labels"] = torch.tensor(self.label2id[row["label"]], dtype=torch.long)
        return item


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune chinese_roberta_L-4_H-256 for finance OCR field classification."
    )
    parser.add_argument("--base-model-dir", default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    return parser.parse_args()


def load_rows(path, label2id):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if text == "":
                continue
            row = json.loads(text)
            if not isinstance(row.get("text"), str) or row["text"].strip() == "":
                raise ValueError(f"Line {line_no}: missing text")
            if row.get("label") not in label2id:
                raise ValueError(f"Line {line_no}: unknown label {row.get('label')}")
            rows.append({"text": row["text"].strip(), "label": row["label"]})
    if not rows:
        raise ValueError(f"No training rows found: {path}")
    return rows


def split_rows(rows, valid_ratio, seed):
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    valid_size = int(len(shuffled) * valid_ratio)
    if valid_size <= 0 and len(shuffled) >= 10:
        valid_size = 1
    valid_rows = shuffled[:valid_size]
    train_rows = shuffled[valid_size:]
    if not train_rows:
        train_rows = shuffled
        valid_rows = []
    return train_rows, valid_rows


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def class_weights(rows, label2id, device):
    counts = [0 for _ in label2id]
    for row in rows:
        counts[label2id[row["label"]]] += 1
    total = sum(counts)
    weights = []
    for count in counts:
        weights.append(total / max(1, len(counts) * count))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, loader, device, weights):
    if loader is None:
        return None
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            labels = batch.pop("labels")
            output = model(**batch)
            loss = F.cross_entropy(output.logits, labels, weight=weights)
            loss_sum += float(loss.item())
            predicted = output.logits.argmax(dim=-1)
            correct += int((predicted == labels).sum().item())
            total += int(labels.numel())
    if total == 0:
        return None
    return {"loss": loss_sum / max(1, len(loader)), "accuracy": correct / total}


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_model_dir = Path(args.base_model_dir)
    train_file = Path(args.train_file)
    output_dir = Path(args.output_dir)
    labels = [label.strip() for label in args.labels.split(",") if label.strip() != ""]
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model dir not found: {base_model_dir}")
    if not train_file.exists():
        raise FileNotFoundError(
            f"Train file not found: {train_file}. "
            "Create JSONL rows like: {\"text\":\"单元格: 5.61; 表头: 盈亏\", \"label\":\"profit\"}"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir))
    rows = load_rows(train_file, label2id)
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)

    train_dataset = FieldDataset(train_rows, tokenizer, label2id, args.max_len)
    valid_dataset = FieldDataset(valid_rows, tokenizer, label2id, args.max_len) if valid_rows else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size) if valid_dataset else None

    model = AutoModelForSequenceClassification.from_pretrained(
        str(base_model_dir),
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    weights = class_weights(train_rows, label2id, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            labels_tensor = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = F.cross_entropy(output.logits, labels_tensor, weight=weights)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())

        train_loss = loss_sum / max(1, len(train_loader))
        metrics = evaluate(model, valid_loader, device, weights)
        if metrics is None:
            print(f"epoch={epoch} train_loss={train_loss:.4f}")
        else:
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"valid_loss={metrics['loss']:.4f} valid_acc={metrics['accuracy']:.4f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as handle:
        json.dump({"labels": labels, "label2id": label2id, "id2label": id2label}, handle, ensure_ascii=False, indent=2)
    print(f"Saved fine-tuned model: {output_dir}")


if __name__ == "__main__":
    main()
