import argparse
import json
from pathlib import Path

import torch
from torch import nn


# 本脚本用于检查“从零训练版”模型的预测效果。
# 它不训练，只加载训练输出目录，然后对 layout JSONL 里的页面逐个 OCR 框预测字段。

DEFAULT_MODEL_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_scratch"
DEFAULT_LAYOUT_FILE = r"E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl"

PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"


def parse_args():
    parser = argparse.ArgumentParser(description="Probe scratch Layout-Aware Encoder predictions.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--layout-file", default=DEFAULT_LAYOUT_FILE)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="", help="Example: cuda, cpu. Empty means auto.")
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_pages(path):
    pages = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                pages.append(json.loads(text))
    if not pages:
        raise ValueError(f"No pages found: {path}")
    return pages


def clamp_int(value, low, high):
    return max(low, min(high, int(round(value))))


def normalize_bbox_1000(bbox, width, height):
    width = max(1, int(width))
    height = max(1, int(height))
    x0, y0, x1, y1 = [float(v) for v in bbox]
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
        x0, y0, x1, y1 = [float(v) for v in item["bbox"]]
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
    for ch in str(text):
        if len(ids) >= max_len:
            break
        ids.append(vocab.get(ch, unk_id))
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(pad_id)
        mask.append(0)
    return ids, mask


def make_transformer_encoder(layer, num_layers):
    try:
        return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


def load_state_dict(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class ScratchLayoutAwareFieldEncoder(nn.Module):
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


def page_to_tensors(page, vocab, config, device):
    max_items = int(config["max_items"])
    max_item_len = int(config["max_item_len"])
    width = int(page.get("width", 1080))
    height = int(page.get("height", 2400))
    items = sorted(page.get("items", []), key=lambda item: (item["bbox"][1], item["bbox"][0]))[:max_items]
    row_ids = cluster_axis(items, axis=1, max_id=max_items)
    col_ids = cluster_axis(items, axis=0, max_id=max_items)

    input_ids = torch.zeros((1, max_items, max_item_len), dtype=torch.long)
    text_mask = torch.zeros((1, max_items, max_item_len), dtype=torch.bool)
    bbox = torch.zeros((1, max_items, 4), dtype=torch.long)
    rows = torch.zeros((1, max_items), dtype=torch.long)
    cols = torch.zeros((1, max_items), dtype=torch.long)
    item_mask = torch.zeros((1, max_items), dtype=torch.bool)

    for i, item in enumerate(items):
        ids, mask = encode_text(item.get("text", ""), vocab, max_item_len)
        input_ids[0, i] = torch.tensor(ids, dtype=torch.long)
        text_mask[0, i] = torch.tensor(mask, dtype=torch.bool)
        bbox[0, i] = torch.tensor(normalize_bbox_1000(item["bbox"], width, height), dtype=torch.long)
        rows[0, i] = row_ids[i]
        cols[0, i] = col_ids[i]
        item_mask[0, i] = True

    return {
        "items": items,
        "input_ids": input_ids.to(device),
        "text_mask": text_mask.to(device),
        "bbox": bbox.to(device),
        "row_ids": rows.to(device),
        "col_ids": cols.to(device),
        "item_mask": item_mask.to(device),
    }


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    config = read_json(model_dir / "model_config.json")
    vocab = read_json(model_dir / "char_vocab.json")
    labels = read_json(model_dir / "label_map.json")["labels"]

    model = ScratchLayoutAwareFieldEncoder(config, len(vocab), len(labels)).to(device)
    state = load_state_dict(model_dir / "layout_field_encoder_scratch.pt", device)
    model.load_state_dict(state)
    model.eval()

    pages = load_pages(args.layout_file)
    end = min(len(pages), args.page_index + args.max_pages)
    with torch.no_grad():
        for page_index in range(args.page_index, end):
            batch = page_to_tensors(pages[page_index], vocab, config, device)
            logits = model(
                input_ids=batch["input_ids"],
                text_mask=batch["text_mask"],
                bbox=batch["bbox"],
                row_ids=batch["row_ids"],
                col_ids=batch["col_ids"],
                item_mask=batch["item_mask"],
            )
            probs = torch.softmax(logits[0], dim=-1).cpu()
            print(json.dumps({"page_index": page_index, "page_type": pages[page_index].get("page_type", "")}, ensure_ascii=False))
            for i, item in enumerate(batch["items"]):
                top = torch.topk(probs[i], min(args.top_k, len(labels)))
                top_items = [
                    {"label": labels[int(label_id)], "score": round(float(score), 4)}
                    for score, label_id in zip(top.values, top.indices)
                ]
                print(json.dumps({
                    "index": i,
                    "text": item.get("text", ""),
                    "gold": item.get("label", ""),
                    "pred": top_items[0]["label"],
                    "top": top_items,
                }, ensure_ascii=False))


if __name__ == "__main__":
    main()
