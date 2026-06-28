import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


# 使用预训练好的 Layout Base Encoder 微调金融字段分类器。
#
# 上一步：
#   pretrain-layout-aware-base-encoder-scratch.py
#
# 本脚本：
#   1. 加载 base_encoder.pt、char_vocab.json、model_config.json。
#   2. 在 base encoder 后面接一个字段分类头。
#   3. 用有标签 layout JSONL 微调。
#
# 输出文件名保持和 scratch 分类模型一致：
#   layout_field_encoder_scratch.pt
#   char_vocab.json
#   label_map.json
#   model_config.json
#
# 因此可以继续复用：
#   probe-layout-aware-field-encoder-scratch.py
#   export-layout-aware-field-encoder-scratch-onnx.py

DEFAULT_BASE_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_layout_base_encoder_scratch"
DEFAULT_TRAIN_FILE = r"E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl"
DEFAULT_OUTPUT_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base"

DEFAULT_LABELS = (
    "merchant,counterparty,amount,income_amount,expense_amount,balance,date_time,"
    "payment_method,order_id,bank_card,transaction_type,status,asset_name,asset_code,"
    "market_value,profit,profit_rate,holding,available,quantity,price,net_value,shares,other"
)

PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune finance field classifier from a pretrained layout base encoder.")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="Example: cuda, cpu. Empty means auto.")
    parser.add_argument("--freeze-base", action="store_true", help="Train only the classification head.")
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_pages(path, label2id):
    pages = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw_text = line.strip()
            if raw_text == "":
                continue
            row = json.loads(raw_text)
            raw_items = row.get("items")
            if not isinstance(raw_items, list) or len(raw_items) == 0:
                raise ValueError(f"Line {line_no}: missing items")
            items = []
            for raw_item in raw_items:
                text = str(raw_item.get("text", "")).strip()
                bbox = raw_item.get("bbox")
                label = str(raw_item.get("label", "")).strip()
                if text == "":
                    continue
                if label not in label2id:
                    raise ValueError(f"Line {line_no}: unknown label={label}, text={text}")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Line {line_no}: bad bbox, text={text}")
                items.append({"text": text, "bbox": [float(v) for v in bbox], "label": label})
            if items:
                pages.append({
                    "page_type": str(row.get("page_type", "unknown")),
                    "width": int(row.get("width", 1080)),
                    "height": int(row.get("height", 2400)),
                    "items": items,
                })
    if not pages:
        raise ValueError(f"No pages found: {path}")
    return pages


def split_pages(pages, valid_ratio, seed):
    shuffled = pages[:]
    random.Random(seed).shuffle(shuffled)
    valid_size = int(len(shuffled) * valid_ratio)
    if valid_size <= 0 and len(shuffled) >= 10:
        valid_size = 1
    valid_pages = shuffled[:valid_size]
    train_pages = shuffled[valid_size:]
    if not train_pages:
        return shuffled, []
    return train_pages, valid_pages


def clamp_int(value, low, high):
    return max(low, min(high, int(round(value))))


def normalize_bbox_1000(bbox, width, height):
    width = max(1, int(width))
    height = max(1, int(height))
    x0, y0, x1, y1 = bbox
    return [
        clamp_int(x0 * 1000.0 / width, 0, 1000),
        clamp_int(y0 * 1000.0 / height, 0, 1000),
        clamp_int(x1 * 1000.0 / width, 0, 1000),
        clamp_int(y1 * 1000.0 / height, 0, 1000),
    ]


def cluster_axis(items, axis, max_id):
    if not items:
        return []
    centers = []
    sizes = []
    for item in items:
        x0, y0, x1, y1 = item["bbox"]
        if axis == 0:
            centers.append((x0 + x1) * 0.5)
            sizes.append(max(1.0, x1 - x0))
        else:
            centers.append((y0 + y1) * 0.5)
            sizes.append(max(1.0, y1 - y0))
    sorted_sizes = sorted(sizes)
    median_size = sorted_sizes[len(sorted_sizes) // 2]
    threshold = max(20.0, median_size * (0.75 if axis == 0 else 1.35))
    order = sorted(range(len(items)), key=lambda index: centers[index])
    ids = [0 for _ in items]
    current_id = 0
    last_center = centers[order[0]]
    for index in order:
        if abs(centers[index] - last_center) > threshold:
            current_id += 1
            last_center = centers[index]
        ids[index] = min(current_id, max_id - 1)
    return ids


def encode_text(text, vocab, max_len):
    pad_id = vocab[PAD]
    unk_id = vocab[UNK]
    ids = [vocab[CLS]]
    for ch in text:
        if len(ids) >= max_len:
            break
        ids.append(vocab.get(ch, unk_id))
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(pad_id)
        mask.append(0)
    return ids, mask


class LayoutFieldDataset(Dataset):
    def __init__(self, pages, vocab, label2id, max_items, max_item_len):
        self.pages = pages
        self.vocab = vocab
        self.label2id = label2id
        self.max_items = max_items
        self.max_item_len = max_item_len

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        page = self.pages[index]
        width = int(page["width"])
        height = int(page["height"])
        items = sorted(page["items"], key=lambda item: (item["bbox"][1], item["bbox"][0]))[: self.max_items]
        row_ids = cluster_axis(items, axis=1, max_id=self.max_items)
        col_ids = cluster_axis(items, axis=0, max_id=self.max_items)

        input_ids = torch.zeros((self.max_items, self.max_item_len), dtype=torch.long)
        text_mask = torch.zeros((self.max_items, self.max_item_len), dtype=torch.bool)
        bbox = torch.zeros((self.max_items, 4), dtype=torch.long)
        rows = torch.zeros((self.max_items,), dtype=torch.long)
        cols = torch.zeros((self.max_items,), dtype=torch.long)
        labels = torch.full((self.max_items,), -100, dtype=torch.long)
        item_mask = torch.zeros((self.max_items,), dtype=torch.bool)

        for i, item in enumerate(items):
            ids, mask = encode_text(item["text"], self.vocab, self.max_item_len)
            input_ids[i] = torch.tensor(ids, dtype=torch.long)
            text_mask[i] = torch.tensor(mask, dtype=torch.bool)
            bbox[i] = torch.tensor(normalize_bbox_1000(item["bbox"], width, height), dtype=torch.long)
            rows[i] = row_ids[i]
            cols[i] = col_ids[i]
            labels[i] = self.label2id[item["label"]]
            item_mask[i] = True

        return {
            "input_ids": input_ids,
            "text_mask": text_mask,
            "bbox": bbox,
            "row_ids": rows,
            "col_ids": cols,
            "labels": labels,
            "item_mask": item_mask,
        }


def make_transformer_encoder(layer, num_layers):
    try:
        return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


class LayoutFieldClassifier(nn.Module):
    def __init__(self, config, vocab_size, num_labels):
        super().__init__()
        hidden_size = int(config["hidden_size"])
        max_items = int(config["max_items"])
        max_item_len = int(config["max_item_len"])
        heads = int(config["heads"])
        dropout = float(config.get("dropout", 0.1))
        pad_id = int(config.get("pad_id", 0))

        self.token_embed = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.text_pos_embed = nn.Embedding(max_item_len, hidden_size)
        text_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.text_encoder = make_transformer_encoder(text_layer, int(config["text_layers"]))

        self.x0_embed = nn.Embedding(1001, hidden_size)
        self.y0_embed = nn.Embedding(1001, hidden_size)
        self.x1_embed = nn.Embedding(1001, hidden_size)
        self.y1_embed = nn.Embedding(1001, hidden_size)
        self.row_embed = nn.Embedding(max_items, hidden_size)
        self.col_embed = nn.Embedding(max_items, hidden_size)
        self.item_type_embed = nn.Embedding(2, hidden_size)

        page_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.page_encoder = make_transformer_encoder(page_layer, int(config["page_layers"]))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, text_mask, bbox, row_ids, col_ids, item_mask):
        batch_size, max_items, max_item_len = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * max_items, max_item_len)
        flat_text_mask = text_mask.reshape(batch_size * max_items, max_item_len).clone()
        empty_text = flat_text_mask.sum(dim=1) == 0
        flat_text_mask[:, 0] = flat_text_mask[:, 0] | empty_text

        positions = torch.arange(max_item_len, device=input_ids.device).unsqueeze(0)
        text_vec = self.token_embed(flat_ids) + self.text_pos_embed(positions)
        encoded_text = self.text_encoder(text_vec, src_key_padding_mask=~flat_text_mask)
        cls_vec = encoded_text[:, 0, :].reshape(batch_size, max_items, -1)

        bbox = bbox.clamp(min=0, max=1000)
        layout_vec = (
            self.x0_embed(bbox[:, :, 0])
            + self.y0_embed(bbox[:, :, 1])
            + self.x1_embed(bbox[:, :, 2])
            + self.y1_embed(bbox[:, :, 3])
            + self.row_embed(row_ids)
            + self.col_embed(col_ids)
            + self.item_type_embed(item_mask.long())
        )
        page_vec = self.layer_norm(cls_vec + layout_vec)
        page_vec = self.dropout(page_vec)
        encoded_page = self.page_encoder(page_vec, src_key_padding_mask=~item_mask)
        return self.classifier(encoded_page)


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def class_weights(pages, label2id, device):
    counts = [0 for _ in label2id]
    for page in pages:
        for item in page["items"]:
            counts[label2id[item["label"]]] += 1
    total = sum(counts)
    weights = [total / max(1, len(counts) * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def masked_loss(logits, labels, weights):
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        weight=weights,
        ignore_index=-100,
    )


def evaluate(model, loader, device, weights):
    if loader is None:
        return None
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(
                input_ids=batch["input_ids"],
                text_mask=batch["text_mask"],
                bbox=batch["bbox"],
                row_ids=batch["row_ids"],
                col_ids=batch["col_ids"],
                item_mask=batch["item_mask"],
            )
            labels = batch["labels"]
            loss = masked_loss(logits, labels, weights)
            loss_sum += float(loss.item())
            active = labels != -100
            pred = logits.argmax(dim=-1)
            correct += int((pred[active] == labels[active]).sum().item())
            total += int(active.sum().item())
    if total == 0:
        return None
    return {"loss": loss_sum / max(1, len(loader)), "accuracy": correct / total}


def save_artifacts(output_dir, model, vocab, labels, base_config, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "layout_field_encoder_scratch.pt")
    (output_dir / "char_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    (output_dir / "label_map.json").write_text(
        json.dumps({"labels": labels, "label2id": label2id, "id2label": id2label}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = dict(base_config)
    config["model_type"] = "finance_layout_aware_field_encoder_from_base"
    config["labels"] = labels
    config["base_dir"] = str(args.base_dir)
    (output_dir / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = Path(args.base_dir)
    train_file = Path(args.train_file)
    if not (base_dir / "base_encoder.pt").exists():
        raise FileNotFoundError(f"Base encoder not found: {base_dir / 'base_encoder.pt'}")
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    label2id = {label: index for index, label in enumerate(labels)}
    base_config = read_json(base_dir / "model_config.json")
    vocab = read_json(base_dir / "char_vocab.json")
    pages = load_pages(train_file, label2id)
    train_pages, valid_pages = split_pages(pages, args.valid_ratio, args.seed)

    train_dataset = LayoutFieldDataset(train_pages, vocab, label2id, int(base_config["max_items"]), int(base_config["max_item_len"]))
    valid_dataset = LayoutFieldDataset(valid_pages, vocab, label2id, int(base_config["max_items"]), int(base_config["max_item_len"])) if valid_pages else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False) if valid_dataset else None

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LayoutFieldClassifier(base_config, len(vocab), len(labels)).to(device)
    missing, unexpected = model.load_state_dict(load_state_dict(base_dir / "base_encoder.pt", device), strict=False)
    if args.freeze_base:
        for name, param in model.named_parameters():
            if not name.startswith("classifier."):
                param.requires_grad = False

    weights = class_weights(train_pages, label2id, device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    print(
        f"finetune_pages={len(pages)} train={len(train_pages)} valid={len(valid_pages)} "
        f"labels={len(labels)} vocab={len(vocab)} device={device} freeze_base={args.freeze_base}"
    )
    print(f"loaded_base missing={len(missing)} unexpected={len(unexpected)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids=batch["input_ids"],
                text_mask=batch["text_mask"],
                bbox=batch["bbox"],
                row_ids=batch["row_ids"],
                col_ids=batch["col_ids"],
                item_mask=batch["item_mask"],
            )
            loss = masked_loss(logits, batch["labels"], weights)
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

    save_artifacts(args.output_dir, model, vocab, labels, base_config, args)
    print(f"Saved fine-tuned field classifier from base: {args.output_dir}")


if __name__ == "__main__":
    main()
