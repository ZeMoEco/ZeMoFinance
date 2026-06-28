import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


# Lite-friendly ONNX exporter.
#
# 普通 export-layout-aware-field-encoder-scratch-onnx.py 使用 PyTorch 内置
# TransformerEncoder。它能导出 ONNX，也能被 converter_lite 转成 .ms，
# 但 MindSpore Lite runtime 可能把 Shape 算子转换成 Custom，导致 benchmark
# 加载失败。
#
# 本脚本只用于导出和端侧转换：
# - 复用训练得到的 state_dict。
# - 手写固定 shape self-attention。
# - 不使用动态 batch/seq shape。
# - 不依赖 PyTorch 内置 MultiheadAttention 的 ONNX 导出路径。

DEFAULT_MODEL_DIR = r"E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base"
DEFAULT_OUTPUT = r"E:\CamXAll\ZEMO\Data\model\finance_layout_field_encoder_from_base\finance_layout_field_encoder_from_base_lite.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description="Export scratch/from-base Layout-Aware Encoder to Lite-friendly ONNX.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class LiteSelfAttention(nn.Module):
    def __init__(self, hidden_size, heads, batch_size, seq_len):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by heads")
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.in_proj_weight = nn.Parameter(torch.empty(hidden_size * 3, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, key_padding_mask):
        # x: [fixed_batch, fixed_seq, hidden]
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(self.batch_size, self.seq_len, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0, :, :].transpose(1, 2)
        k = qkv[:, :, 1, :, :].transpose(1, 2)
        v = qkv[:, :, 2, :, :].transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(float(self.head_dim)))
        # key_padding_mask=True means ignored key. Use a finite negative value
        # instead of -inf so all-padding rows stay numeric.
        mask = key_padding_mask.to(dtype=scores.dtype).unsqueeze(1).unsqueeze(2)
        scores = scores + mask * -10000.0
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).reshape(self.batch_size, self.seq_len, self.hidden_size)
        return self.out_proj(context)


class LiteTransformerLayer(nn.Module):
    def __init__(self, hidden_size, heads, batch_size, seq_len):
        super().__init__()
        self.self_attn = LiteSelfAttention(hidden_size, heads, batch_size, seq_len)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, key_padding_mask):
        attn = self.self_attn(x, key_padding_mask)
        x = self.norm1(x + attn)
        ff = self.linear2(F.gelu(self.linear1(x)))
        x = self.norm2(x + ff)
        return x


class LiteTransformerEncoder(nn.Module):
    def __init__(self, hidden_size, heads, batch_size, seq_len, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            LiteTransformerLayer(hidden_size, heads, batch_size, seq_len)
            for _ in range(layers)
        ])

    def forward(self, x, key_padding_mask):
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return x


class LiteLayoutFieldClassifier(nn.Module):
    def __init__(self, config, vocab_size, num_labels):
        super().__init__()
        self.hidden_size = int(config["hidden_size"])
        self.max_items = int(config["max_items"])
        self.max_item_len = int(config["max_item_len"])
        heads = int(config["heads"])
        pad_id = int(config.get("pad_id", 0))

        self.token_embed = nn.Embedding(vocab_size, self.hidden_size, padding_idx=pad_id)
        self.text_pos_embed = nn.Embedding(self.max_item_len, self.hidden_size)
        self.text_encoder = LiteTransformerEncoder(
            hidden_size=self.hidden_size,
            heads=heads,
            batch_size=self.max_items,
            seq_len=self.max_item_len,
            layers=int(config["text_layers"]),
        )

        self.x0_embed = nn.Embedding(1001, self.hidden_size)
        self.y0_embed = nn.Embedding(1001, self.hidden_size)
        self.x1_embed = nn.Embedding(1001, self.hidden_size)
        self.y1_embed = nn.Embedding(1001, self.hidden_size)
        self.row_embed = nn.Embedding(self.max_items, self.hidden_size)
        self.col_embed = nn.Embedding(self.max_items, self.hidden_size)
        self.item_type_embed = nn.Embedding(2, self.hidden_size)

        self.page_encoder = LiteTransformerEncoder(
            hidden_size=self.hidden_size,
            heads=heads,
            batch_size=1,
            seq_len=self.max_items,
            layers=int(config["page_layers"]),
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        self.register_buffer("position_ids", torch.arange(self.max_item_len, dtype=torch.long), persistent=False)

    def forward(self, input_ids, text_mask, bbox, row_ids, col_ids, item_mask):
        flat_ids = input_ids.reshape(self.max_items, self.max_item_len)
        flat_text_mask = text_mask.reshape(self.max_items, self.max_item_len)
        token_vec = self.token_embed(flat_ids) + self.text_pos_embed(self.position_ids).unsqueeze(0)
        token_hidden = self.text_encoder(token_vec, ~flat_text_mask)
        cls_vec = token_hidden[:, 0, :].reshape(1, self.max_items, self.hidden_size)

        # bbox is already clamped to 0..1000 by preprocessing.
        layout_vec = (
            self.x0_embed(bbox[:, :, 0])
            + self.y0_embed(bbox[:, :, 1])
            + self.x1_embed(bbox[:, :, 2])
            + self.y1_embed(bbox[:, :, 3])
            + self.row_embed(row_ids)
            + self.col_embed(col_ids)
            + self.item_type_embed(item_mask.to(torch.long))
        )
        page_vec = self.layer_norm(cls_vec + layout_vec)
        page_hidden = self.page_encoder(page_vec, ~item_mask)
        return self.classifier(page_hidden)


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = read_json(model_dir / "model_config.json")
    vocab = read_json(model_dir / "char_vocab.json")
    labels = read_json(model_dir / "label_map.json")["labels"]
    model = LiteLayoutFieldClassifier(config, len(vocab), len(labels))
    state = load_state_dict(model_dir / "layout_field_encoder_scratch.pt")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    max_items = int(config["max_items"])
    max_item_len = int(config["max_item_len"])
    input_ids = torch.zeros((1, max_items, max_item_len), dtype=torch.long)
    text_mask = torch.zeros((1, max_items, max_item_len), dtype=torch.bool)
    bbox = torch.zeros((1, max_items, 4), dtype=torch.long)
    row_ids = torch.zeros((1, max_items), dtype=torch.long)
    col_ids = torch.zeros((1, max_items), dtype=torch.long)
    item_mask = torch.zeros((1, max_items), dtype=torch.bool)
    input_ids[0, 0, 0] = int(config.get("cls_id", 2))
    text_mask[0, 0, 0] = True
    item_mask[0, 0] = True

    with torch.no_grad():
        logits = model(input_ids, text_mask, bbox, row_ids, col_ids, item_mask)
    print(f"Dry run logits shape: {tuple(logits.shape)}, labels={len(labels)}")

    torch.onnx.export(
        model,
        (input_ids, text_mask, bbox, row_ids, col_ids, item_mask),
        str(output),
        input_names=["input_ids", "text_mask", "bbox", "row_ids", "col_ids", "item_mask"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Lite-friendly ONNX exported: {output}")


if __name__ == "__main__":
    main()
