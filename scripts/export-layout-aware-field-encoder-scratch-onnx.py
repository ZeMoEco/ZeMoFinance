import argparse
import json
from pathlib import Path

import torch
from torch import nn


# 把“从零训练版” Layout-Aware Encoder 导出为 ONNX。
# 这个 ONNX 还不是最终手机可直接用的 .ms；它是中间格式：
# PyTorch .pt -> ONNX -> MindSpore Lite .ms / 其它端侧推理格式。

DEFAULT_MODEL_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_scratch"
DEFAULT_OUTPUT = r"E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_scratch\finance_layout_field_encoder_scratch.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Export scratch Layout-Aware Encoder to ONNX.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-batch", action="store_true", help="Allow dynamic batch size. Keep off for simpler mobile conversion.")
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_transformer_encoder(layer, num_layers):
    try:
        return nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
    except TypeError:
        return nn.TransformerEncoder(layer, num_layers=num_layers)


def load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def load_model(model_dir):
    config = read_json(model_dir / "model_config.json")
    vocab = read_json(model_dir / "char_vocab.json")
    labels = read_json(model_dir / "label_map.json")["labels"]
    model = ScratchLayoutAwareFieldEncoder(config, len(vocab), len(labels))
    state = load_state_dict(model_dir / "layout_field_encoder_scratch.pt")
    model.load_state_dict(state)
    model.eval()
    return model, config, labels


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, config, labels = load_model(model_dir)

    max_items = int(config["max_items"])
    max_item_len = int(config["max_item_len"])
    input_ids = torch.zeros((1, max_items, max_item_len), dtype=torch.long)
    text_mask = torch.zeros((1, max_items, max_item_len), dtype=torch.bool)
    bbox = torch.zeros((1, max_items, 4), dtype=torch.long)
    row_ids = torch.zeros((1, max_items), dtype=torch.long)
    col_ids = torch.zeros((1, max_items), dtype=torch.long)
    item_mask = torch.zeros((1, max_items), dtype=torch.bool)

    # 给第一个 OCR 框一个非空示例，避免导出时全 padding 掩码触发特殊路径。
    input_ids[0, 0, 0] = 2
    text_mask[0, 0, 0] = True
    item_mask[0, 0] = True

    with torch.no_grad():
        logits = model(input_ids, text_mask, bbox, row_ids, col_ids, item_mask)
    print(f"Dry run logits shape: {tuple(logits.shape)}, labels={len(labels)}")

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "input_ids": {0: "batch"},
            "text_mask": {0: "batch"},
            "bbox": {0: "batch"},
            "row_ids": {0: "batch"},
            "col_ids": {0: "batch"},
            "item_mask": {0: "batch"},
            "logits": {0: "batch"},
        }

    torch.onnx.export(
        model,
        (input_ids, text_mask, bbox, row_ids, col_ids, item_mask),
        str(output),
        input_names=["input_ids", "text_mask", "bbox", "row_ids", "col_ids", "item_mask"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print(f"ONNX exported: {output}")


if __name__ == "__main__":
    main()
