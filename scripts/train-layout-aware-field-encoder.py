import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


# Layout-Aware Encoder 的核心思路：
# 1. 每个 OCR 框先用中文 RoBERTa 编成“文本向量”。
# 2. 再把 bbox 坐标、行号、列号编成“布局向量”。
# 3. 文本向量 + 布局向量 输入页级 Transformer Encoder。
# 4. 最后对每个 OCR 框分类：金额、基金名、代码、时间、状态等。
#
# 这和当前项目里的“把坐标拼进一句文本再分类”不同：
# 当前做法：模型只看到字符串。
# 本脚本：模型显式看到整页 OCR 框的二维布局。

DEFAULT_BASE_MODEL_DIR = r"E:\CamXAll\ZEMO\Data\model\chinese_roberta_L-4_H-256"
DEFAULT_TRAIN_FILE = r"E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl"
DEFAULT_OUTPUT_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_model"

DEFAULT_LABELS = (
    "merchant,counterparty,amount,income_amount,expense_amount,balance,date_time,"
    "payment_method,order_id,bank_card,transaction_type,status,asset_name,asset_code,"
    "market_value,profit,profit_rate,holding,available,quantity,price,net_value,shares,other"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a page-level Layout-Aware Encoder for finance OCR fields.")
    parser.add_argument("--base-model-dir", default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--max-items", type=int, default=96)
    parser.add_argument("--max-item-len", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--page-layers", type=int, default=2)
    parser.add_argument("--page-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="Example: cuda, cpu. Empty means auto.")
    parser.add_argument("--unfreeze-text-encoder", action="store_true", help="Fine-tune RoBERTa too; needs more data/GPU.")
    return parser.parse_args()


def load_pages(path, label2id):
    pages = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            raw_text = line.strip()
            if raw_text == "":
                continue
            row = json.loads(raw_text)
            items = row.get("items")
            if not isinstance(items, list) or len(items) == 0:
                raise ValueError(f"Line {line_no}: missing items")
            normalized_items = []
            for item in items:
                text = str(item.get("text", "")).strip()
                bbox = item.get("bbox")
                label = str(item.get("label", "")).strip()
                if text == "":
                    continue
                if label not in label2id:
                    raise ValueError(f"Line {line_no}: unknown label={label}, text={text}")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Line {line_no}: bad bbox, text={text}")
                normalized_items.append({
                    "text": text,
                    "bbox": [float(v) for v in bbox],
                    "label": label,
                })
            if normalized_items:
                pages.append({
                    "page_type": str(row.get("page_type", "unknown")),
                    "width": int(row.get("width", 1080)),
                    "height": int(row.get("height", 2400)),
                    "items": normalized_items,
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
    # LayoutLM 系列模型常把页面坐标归一化到 0..1000。
    # 这样不同手机分辨率的截图可以落在同一坐标空间。
    width = max(1, width)
    height = max(1, height)
    x0, y0, x1, y1 = bbox
    return [
        clamp_int(x0 * 1000.0 / width, 0, 1000),
        clamp_int(y0 * 1000.0 / height, 0, 1000),
        clamp_int(x1 * 1000.0 / width, 0, 1000),
        clamp_int(y1 * 1000.0 / height, 0, 1000),
    ]


def cluster_axis(items, axis, max_id):
    # 根据 OCR 框中心点粗略聚类行/列。
    # axis=0 表示按 x 聚类列，axis=1 表示按 y 聚类行。
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


class LayoutPageDataset(Dataset):
    def __init__(self, pages, tokenizer, label2id, max_items, max_item_len):
        self.pages = pages
        self.tokenizer = tokenizer
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
        attention_mask = torch.zeros((self.max_items, self.max_item_len), dtype=torch.long)
        bbox = torch.zeros((self.max_items, 4), dtype=torch.long)
        rows = torch.zeros((self.max_items,), dtype=torch.long)
        cols = torch.zeros((self.max_items,), dtype=torch.long)
        labels = torch.full((self.max_items,), -100, dtype=torch.long)
        item_mask = torch.zeros((self.max_items,), dtype=torch.bool)

        for i, item in enumerate(items):
            encoded = self.tokenizer(
                item["text"],
                max_length=self.max_item_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids[i] = encoded["input_ids"].squeeze(0)
            attention_mask[i] = encoded["attention_mask"].squeeze(0)
            bbox[i] = torch.tensor(normalize_bbox_1000(item["bbox"], width, height), dtype=torch.long)
            rows[i] = row_ids[i]
            cols[i] = col_ids[i]
            labels[i] = self.label2id[item["label"]]
            item_mask[i] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "row_ids": rows,
            "col_ids": cols,
            "labels": labels,
            "item_mask": item_mask,
        }


class LayoutAwareFieldEncoder(nn.Module):
    def __init__(
        self,
        base_model_dir,
        num_labels,
        hidden_size,
        max_items,
        page_layers,
        page_heads,
        dropout,
        freeze_text_encoder,
    ):
        super().__init__()
        try:
            # 分类只取每个 OCR 文本的 CLS 向量，不需要 BERT pooler。
            self.text_encoder = AutoModel.from_pretrained(str(base_model_dir), add_pooling_layer=False)
        except TypeError:
            self.text_encoder = AutoModel.from_pretrained(str(base_model_dir))
        text_hidden = int(getattr(self.text_encoder.config, "hidden_size", hidden_size))
        self.text_proj = nn.Linear(text_hidden, hidden_size)

        # bbox 四个坐标分别 embedding，再求和。bucket 数是 1001，对应 0..1000。
        self.x0_embed = nn.Embedding(1001, hidden_size)
        self.y0_embed = nn.Embedding(1001, hidden_size)
        self.x1_embed = nn.Embedding(1001, hidden_size)
        self.y1_embed = nn.Embedding(1001, hidden_size)
        self.row_embed = nn.Embedding(max_items, hidden_size)
        self.col_embed = nn.Embedding(max_items, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=page_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.page_encoder = nn.TransformerEncoder(encoder_layer, num_layers=page_layers)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, bbox, row_ids, col_ids, item_mask):
        batch_size, max_items, max_item_len = input_ids.shape

        # 把 [B, N, L] 拉平成 [B*N, L]，让 RoBERTa 一次性编码所有 OCR 框文本。
        flat_input_ids = input_ids.reshape(batch_size * max_items, max_item_len)
        flat_attention_mask = attention_mask.reshape(batch_size * max_items, max_item_len).clone()

        # padding item 的 attention_mask 全 0，部分 Transformer 实现会不稳定。
        # 这里给它临时打开第 1 个 token，后面 item_mask 会把它屏蔽掉。
        empty_text = flat_attention_mask.sum(dim=1) == 0
        if empty_text.any():
            flat_attention_mask[empty_text, 0] = 1

        text_output = self.text_encoder(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        text_vec = text_output.last_hidden_state[:, 0, :].reshape(batch_size, max_items, -1)
        text_vec = self.text_proj(text_vec)

        bbox = bbox.clamp(min=0, max=1000)
        layout_vec = (
            self.x0_embed(bbox[:, :, 0])
            + self.y0_embed(bbox[:, :, 1])
            + self.x1_embed(bbox[:, :, 2])
            + self.y1_embed(bbox[:, :, 3])
            + self.row_embed(row_ids)
            + self.col_embed(col_ids)
        )

        page_vec = self.layer_norm(text_vec + layout_vec)
        page_vec = self.dropout(page_vec)

        # src_key_padding_mask=True 表示这个 OCR item 是 padding，不参与页级注意力。
        encoded = self.page_encoder(page_vec, src_key_padding_mask=~item_mask)
        return self.classifier(encoded)


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
            labels = batch["labels"]
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                bbox=batch["bbox"],
                row_ids=batch["row_ids"],
                col_ids=batch["col_ids"],
                item_mask=batch["item_mask"],
            )
            loss = masked_loss(logits, labels, weights)
            loss_sum += float(loss.item())
            active = labels != -100
            pred = logits.argmax(dim=-1)
            correct += int((pred[active] == labels[active]).sum().item())
            total += int(active.sum().item())
    if total == 0:
        return None
    return {"loss": loss_sum / max(1, len(loader)), "accuracy": correct / total}


def save_model(output_dir, model, tokenizer, labels, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))
    torch.save(model.state_dict(), output_dir / "layout_field_encoder.pt")
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as handle:
        json.dump({"labels": labels, "label2id": label2id, "id2label": id2label}, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "model_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_type": "finance_layout_aware_field_encoder",
                "base_model_dir": str(args.base_model_dir),
                "max_items": args.max_items,
                "max_item_len": args.max_item_len,
                "hidden_size": args.hidden_size,
                "page_layers": args.page_layers,
                "page_heads": args.page_heads,
                "dropout": args.dropout,
                "labels": labels,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_model_dir = Path(args.base_model_dir)
    train_file = Path(args.train_file)
    if not base_model_dir.exists():
        raise FileNotFoundError(f"Base model dir not found: {base_model_dir}")
    if not train_file.exists():
        raise FileNotFoundError(
            f"Train file not found: {train_file}. Run scripts/prepare-finance-layout-data.py first."
        )

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    label2id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir))
    pages = load_pages(train_file, label2id)
    train_pages, valid_pages = split_pages(pages, args.valid_ratio, args.seed)

    train_dataset = LayoutPageDataset(train_pages, tokenizer, label2id, args.max_items, args.max_item_len)
    valid_dataset = LayoutPageDataset(valid_pages, tokenizer, label2id, args.max_items, args.max_item_len) if valid_pages else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False) if valid_dataset else None

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LayoutAwareFieldEncoder(
        base_model_dir=base_model_dir,
        num_labels=len(labels),
        hidden_size=args.hidden_size,
        max_items=args.max_items,
        page_layers=args.page_layers,
        page_heads=args.page_heads,
        dropout=args.dropout,
        freeze_text_encoder=not args.unfreeze_text_encoder,
    ).to(device)

    weights = class_weights(train_pages, label2id, device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(
        f"pages={len(pages)} train={len(train_pages)} valid={len(valid_pages)} "
        f"labels={len(labels)} device={device} text_encoder={'trainable' if args.unfreeze_text_encoder else 'frozen'}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
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

    save_model(args.output_dir, model, tokenizer, labels, args)
    print(f"Saved layout-aware encoder: {args.output_dir}")


if __name__ == "__main__":
    main()
