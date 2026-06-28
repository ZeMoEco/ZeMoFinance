import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_TRAIN_FILE = r"E:\CamXAll\ZEMO\Data\model\finance_field_cls_train.jsonl"
DEFAULT_MODEL_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_field_cls_model"
DEFAULT_OUTPUT = r"E:\CamXAll\ZEMO\Data\model\finance_field_cls_model\finance_field_cls.onnx"
DEFAULT_FEATURE_DIM = 4096


class HashLinearClassifier(torch.nn.Module):
    def __init__(self, feature_dim: int, num_labels: int):
        super().__init__()
        feature_side = int(math.sqrt(float(feature_dim)))
        if feature_side * feature_side != feature_dim:
            raise ValueError(f"feature_dim must be a square number for conv export: {feature_dim}")
        self.feature_side = feature_side
        self.conv = torch.nn.Conv2d(1, num_labels, kernel_size=feature_side)

    def forward(self, features):
        return self.conv(features)


class FieldHashDataset(Dataset):
    def __init__(self, rows, label2id, feature_dim):
        self.rows = rows
        self.label2id = label2id
        self.feature_dim = feature_dim
        self.features = torch.tensor([text_features(row["text"], feature_dim) for row in rows], dtype=torch.float32)
        self.labels = torch.tensor([label2id[row["label"]] for row in rows], dtype=torch.long)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def parse_args():
    parser = argparse.ArgumentParser(description="Export a Harmony-friendly hash field classifier to ONNX.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.12)
    return parser.parse_args()


def load_labels(model_dir: Path):
    for path in [model_dir / "label_map.json", model_dir / "slm" / "label_map.json"]:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        labels = data.get("labels")
        if isinstance(labels, list) and labels:
            return [str(label) for label in labels]
    raise FileNotFoundError(f"label_map.json not found under {model_dir}")


def load_rows(path: Path, labels):
    label_set = set(labels)
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw = line.strip()
            if raw == "":
                continue
            row = json.loads(raw)
            text = str(row.get("text", "")).strip()
            label = str(row.get("label", "")).strip()
            if text == "" or label not in label_set:
                raise ValueError(f"Invalid row at {line_no}: label={label}")
            rows.append({"text": text, "label": label})
    if not rows:
        raise ValueError(f"No rows found: {path}")
    return rows


def split_rows(rows, valid_ratio, seed):
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    valid_size = max(1, int(len(shuffled) * valid_ratio)) if len(shuffled) >= 10 else 0
    return shuffled[valid_size:], shuffled[:valid_size]


def stable_hash(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 2147483647
    return value


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().replace("\r", "\n").split())


def is_separator(ch: str) -> bool:
    return ch.isspace() or ch in ";:,.，。；：/\\|()（）[]【】{}<>《》=+-*_'\""


def feature_tokens(text: str):
    value = normalize_text(text)
    tokens = []
    word = ""
    prev = ""
    for ch in value:
        if not ch.isspace():
            tokens.append("u:" + ch)
            if prev != "":
                tokens.append("b:" + prev + ch)
        if is_separator(ch):
            if word != "":
                push_word_tokens(tokens, word)
                word = ""
        else:
            word += ch
        prev = "" if ch.isspace() else ch
    if word != "":
        push_word_tokens(tokens, word)
    return tokens


def push_word_tokens(tokens, word: str):
    if word == "":
        return
    if len(word) <= 32:
        tokens.append("w:" + word)
    if len(word) > 3:
        tokens.append("p:" + word[:3])
        tokens.append("s:" + word[-3:])


def text_features(text: str, feature_dim: int):
    features = [0.0] * feature_dim
    tokens = feature_tokens(text)
    if not tokens:
        return features
    for token in tokens:
        features[stable_hash(token) % feature_dim] += 1.0
    scale = 1.0 / math.sqrt(float(len(tokens)))
    for i, value in enumerate(features):
        if value != 0.0:
            features[i] = value * scale
    return features


def collate(batch):
    feature_dim = int(batch[0][0].numel())
    feature_side = int(math.sqrt(float(feature_dim)))
    if feature_side * feature_side != feature_dim:
        raise ValueError(f"feature_dim must be a square number for conv export: {feature_dim}")
    xs = torch.stack([item[0] for item in batch], dim=0).reshape(len(batch), 1, feature_side, feature_side)
    ys = torch.stack([item[1] for item in batch], dim=0)
    return xs, ys


def class_weights(rows, label2id):
    counts = [0 for _ in label2id]
    for row in rows:
        counts[label2id[row["label"]]] += 1
    total = sum(counts)
    return torch.tensor([total / max(1, len(counts) * count) for count in counts], dtype=torch.float32)


def evaluate(model, loader, weights):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x).reshape(x.shape[0], -1)
            loss = F.cross_entropy(logits, y, weight=weights)
            loss_sum += float(loss.item())
            pred = logits.argmax(dim=-1)
            total += int(y.numel())
            correct += int((pred == y).sum().item())
    return {"loss": loss_sum / max(1, len(loader)), "accuracy": correct / max(1, total)}


def rounded_values(values):
    return [round(float(value), 7) for value in values]


def write_label_map(model_dir: Path, labels, model: HashLinearClassifier, feature_dim: int):
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    weight = model.conv.weight.detach().cpu().reshape(len(labels), feature_dim)
    bias = model.conv.bias.detach().cpu()
    payload = {
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "hash_feature_dim": feature_dim,
        "hash_weights": rounded_values(weight.reshape(-1).tolist()),
        "hash_bias": rounded_values(bias.tolist()),
    }
    json_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    (model_dir / "label_map.json").write_text(json_text, encoding="utf-8")
    slm_dir = model_dir / "slm"
    slm_dir.mkdir(parents=True, exist_ok=True)
    (slm_dir / "label_map.json").write_text(json_text, encoding="utf-8")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_dir = Path(args.model_dir)
    train_file = Path(args.train_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = load_labels(model_dir)
    label2id = {label: index for index, label in enumerate(labels)}
    rows = load_rows(train_file, labels)
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)

    train_loader = DataLoader(
        FieldHashDataset(train_rows, label2id, args.feature_dim),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        FieldHashDataset(valid_rows, label2id, args.feature_dim),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    model = HashLinearClassifier(args.feature_dim, len(labels))
    weights = class_weights(train_rows, label2id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x).reshape(x.shape[0], -1)
            loss = F.cross_entropy(logits, y, weight=weights)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            metrics = evaluate(model, valid_loader, weights)
            print(
                f"epoch={epoch} train_loss={loss_sum / max(1, len(train_loader)):.4f} "
                f"valid_loss={metrics['loss']:.4f} valid_acc={metrics['accuracy']:.4f}"
            )

    model.eval()
    feature_side = int(math.sqrt(float(args.feature_dim)))
    if feature_side * feature_side != args.feature_dim:
        raise ValueError(f"feature_dim must be a square number for conv export: {args.feature_dim}")
    dummy = torch.zeros((1, 1, feature_side, feature_side), dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy)
    print(f"Dry run logits shape: {tuple(logits.shape)}")
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
    )
    write_label_map(model_dir, labels, model, args.feature_dim)
    (model_dir / "finance_field_cls_hash_config.json").write_text(
        json.dumps({"feature_dim": args.feature_dim, "feature_shape": [1, feature_side, feature_side], "labels": len(labels)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"ONNX exported: {output_path}")
    print(f"feature_dim={args.feature_dim}, labels={len(labels)}")


if __name__ == "__main__":
    main()
