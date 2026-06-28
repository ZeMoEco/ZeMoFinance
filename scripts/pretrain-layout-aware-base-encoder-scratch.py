import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


# 从零制作一个可复用的 Layout-Aware Base Encoder。
#
# 这个脚本不使用 RoBERTa/BERT/HuggingFace 本地模型。
# 它使用自监督 MLM 任务预训练：
# 1. 从 OCR 文本中随机 mask 一部分字符。
# 2. 模型看到剩余字符 + bbox/行/列布局。
# 3. 模型预测被 mask 的原字符。
#
# 预训练完成后得到 base_encoder.pt。
# 下游任务可以加载它，再接字段分类头、金额抽取头、页面分类头等。

DEFAULT_TRAIN_FILE = r"E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl"
DEFAULT_OUTPUT_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_layout_base_encoder_scratch"

PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"
MASK = "[MASK]"
SPECIAL_TOKENS = [PAD, UNK, CLS, MASK]


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain a from-scratch Layout-Aware Base Encoder with MLM.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=96)
    parser.add_argument("--max-item-len", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--text-layers", type=int, default=2)
    parser.add_argument("--page-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--min-char-freq", type=int, default=1)
    parser.add_argument("--max-vocab-size", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="", help="Example: cuda, cpu. Empty means auto.")
    return parser.parse_args()


def load_pages(path):
    pages = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if text == "":
                continue
            row = json.loads(text)
            raw_items = row.get("items")
            if not isinstance(raw_items, list) or len(raw_items) == 0:
                raise ValueError(f"Line {line_no}: missing items")
            items = []
            for raw_item in raw_items:
                item_text = str(raw_item.get("text", "")).strip()
                bbox = raw_item.get("bbox")
                if item_text == "":
                    continue
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Line {line_no}: bad bbox, text={item_text}")
                items.append({"text": item_text, "bbox": [float(v) for v in bbox]})
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


def build_char_vocab(pages, min_freq, max_vocab_size):
    counter = Counter()
    for page in pages:
        for item in page["items"]:
            counter.update(item["text"])
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for ch, count in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0])):
        if count < min_freq:
            continue
        if ch in vocab:
            continue
        if len(vocab) >= max_vocab_size:
            break
        vocab[ch] = len(vocab)
    return vocab


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


def encode_mlm_text(text, vocab, max_len, mask_prob):
    pad_id = vocab[PAD]
    unk_id = vocab[UNK]
    cls_id = vocab[CLS]
    mask_id = vocab[MASK]
    ids = [cls_id]
    labels = [-100]
    candidate_positions = []

    for ch in text:
        if len(ids) >= max_len:
            break
        token_id = vocab.get(ch, unk_id)
        candidate_positions.append(len(ids))
        labels.append(-100)
        if random.random() < mask_prob:
            labels[-1] = token_id
            choice = random.random()
            if choice < 0.8:
                ids.append(mask_id)
            elif choice < 0.9:
                ids.append(random.randint(0, len(vocab) - 1))
            else:
                ids.append(token_id)
        else:
            ids.append(token_id)

    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(pad_id)
        labels.append(-100)
        mask.append(0)
    return ids, mask, labels, candidate_positions


class LayoutMlmDataset(Dataset):
    def __init__(self, pages, vocab, max_items, max_item_len, mask_prob):
        self.pages = pages
        self.vocab = vocab
        self.max_items = max_items
        self.max_item_len = max_item_len
        self.mask_prob = mask_prob

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
        mlm_labels = torch.full((self.max_items, self.max_item_len), -100, dtype=torch.long)
        bbox = torch.zeros((self.max_items, 4), dtype=torch.long)
        rows = torch.zeros((self.max_items,), dtype=torch.long)
        cols = torch.zeros((self.max_items,), dtype=torch.long)
        item_mask = torch.zeros((self.max_items,), dtype=torch.bool)

        first_candidate = None
        masked_count = 0
        for i, item in enumerate(items):
            ids, mask, labels, candidates = encode_mlm_text(item["text"], self.vocab, self.max_item_len, self.mask_prob)
            if first_candidate is None and candidates:
                first_candidate = (i, candidates[0])
            masked_count += sum(1 for label in labels if label != -100)
            input_ids[i] = torch.tensor(ids, dtype=torch.long)
            text_mask[i] = torch.tensor(mask, dtype=torch.bool)
            mlm_labels[i] = torch.tensor(labels, dtype=torch.long)
            bbox[i] = torch.tensor(normalize_bbox_1000(item["bbox"], width, height), dtype=torch.long)
            rows[i] = row_ids[i]
            cols[i] = col_ids[i]
            item_mask[i] = True

        # CrossEntropy 全 ignore_index 会得到 nan；确保每页至少有一个 MLM 目标。
        if masked_count == 0 and first_candidate is not None:
            item_index, token_index = first_candidate
            mlm_labels[item_index, token_index] = input_ids[item_index, token_index]
            input_ids[item_index, token_index] = self.vocab[MASK]

        return {
            "input_ids": input_ids,
            "text_mask": text_mask,
            "mlm_labels": mlm_labels,
            "bbox": bbox,
            "row_ids": rows,
            "col_ids": cols,
            "item_mask": item_mask,
        }


def make_transformer_encoder(layer, num_layers):
    try:
        return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


class LayoutBaseEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_items, max_item_len, text_layers, page_layers, heads, dropout, pad_id):
        super().__init__()
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
        self.text_encoder = make_transformer_encoder(text_layer, text_layers)

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
        self.page_encoder = make_transformer_encoder(page_layer, page_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, text_mask, bbox, row_ids, col_ids, item_mask, return_token_hidden=False):
        batch_size, max_items, max_item_len = input_ids.shape
        flat_ids = input_ids.reshape(batch_size * max_items, max_item_len)
        flat_text_mask = text_mask.reshape(batch_size * max_items, max_item_len).clone()
        empty_text = flat_text_mask.sum(dim=1) == 0
        flat_text_mask[:, 0] = flat_text_mask[:, 0] | empty_text

        positions = torch.arange(max_item_len, device=input_ids.device).unsqueeze(0)
        text_vec = self.token_embed(flat_ids) + self.text_pos_embed(positions)
        token_hidden = self.text_encoder(text_vec, src_key_padding_mask=~flat_text_mask)
        cls_vec = token_hidden[:, 0, :].reshape(batch_size, max_items, -1)

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
        page_hidden = self.page_encoder(page_vec, src_key_padding_mask=~item_mask)

        if not return_token_hidden:
            return page_hidden
        token_hidden = token_hidden.reshape(batch_size, max_items, max_item_len, -1)
        token_hidden = token_hidden + page_hidden.unsqueeze(2)
        return page_hidden, token_hidden


class LayoutMlmModel(nn.Module):
    def __init__(self, base_encoder, hidden_size, vocab_size):
        super().__init__()
        self.base = base_encoder
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, input_ids, text_mask, bbox, row_ids, col_ids, item_mask):
        _page_hidden, token_hidden = self.base(
            input_ids=input_ids,
            text_mask=text_mask,
            bbox=bbox,
            row_ids=row_ids,
            col_ids=col_ids,
            item_mask=item_mask,
            return_token_hidden=True,
        )
        return self.mlm_head(token_hidden)


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def mlm_loss(logits, labels):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def evaluate(model, loader, device):
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
            labels = batch["mlm_labels"]
            loss = mlm_loss(logits, labels)
            loss_sum += float(loss.item())
            active = labels != -100
            pred = logits.argmax(dim=-1)
            correct += int((pred[active] == labels[active]).sum().item())
            total += int(active.sum().item())
    if total == 0:
        return None
    return {"loss": loss_sum / max(1, len(loader)), "accuracy": correct / total}


def save_artifacts(output_dir, model, vocab, args):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.base.state_dict(), output_dir / "base_encoder.pt")
    torch.save(model.state_dict(), output_dir / "pretrain_mlm_model.pt")
    (output_dir / "char_vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "model_type": "finance_layout_base_encoder_scratch",
        "max_items": args.max_items,
        "max_item_len": args.max_item_len,
        "hidden_size": args.hidden_size,
        "text_layers": args.text_layers,
        "page_layers": args.page_layers,
        "heads": args.heads,
        "dropout": args.dropout,
        "vocab_size": len(vocab),
        "pad_id": vocab[PAD],
        "unk_id": vocab[UNK],
        "cls_id": vocab[CLS],
        "mask_id": vocab[MASK],
        "pretrain_task": "masked_character_modeling",
    }
    (output_dir / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_file = Path(args.train_file)
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}. Run scripts/prepare-finance-layout-data.py first.")

    pages = load_pages(train_file)
    train_pages, valid_pages = split_pages(pages, args.valid_ratio, args.seed)
    vocab = build_char_vocab(train_pages, args.min_char_freq, args.max_vocab_size)

    train_dataset = LayoutMlmDataset(train_pages, vocab, args.max_items, args.max_item_len, args.mask_prob)
    valid_dataset = LayoutMlmDataset(valid_pages, vocab, args.max_items, args.max_item_len, args.mask_prob) if valid_pages else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False) if valid_dataset else None

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    base = LayoutBaseEncoder(
        vocab_size=len(vocab),
        hidden_size=args.hidden_size,
        max_items=args.max_items,
        max_item_len=args.max_item_len,
        text_layers=args.text_layers,
        page_layers=args.page_layers,
        heads=args.heads,
        dropout=args.dropout,
        pad_id=vocab[PAD],
    )
    model = LayoutMlmModel(base, args.hidden_size, len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(
        f"pretrain_pages={len(pages)} train={len(train_pages)} valid={len(valid_pages)} "
        f"vocab={len(vocab)} device={device}"
    )

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
            loss = mlm_loss(logits, batch["mlm_labels"])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())

        train_loss = loss_sum / max(1, len(train_loader))
        metrics = evaluate(model, valid_loader, device)
        if metrics is None:
            print(f"epoch={epoch} train_loss={train_loss:.4f}")
        else:
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"valid_loss={metrics['loss']:.4f} valid_mlm_acc={metrics['accuracy']:.4f}"
            )

    save_artifacts(args.output_dir, model, vocab, args)
    print(f"Saved layout base encoder: {args.output_dir}")


if __name__ == "__main__":
    main()
