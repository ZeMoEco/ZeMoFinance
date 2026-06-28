import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = ROOT / "model" / "data" / "processed"
OUT_ROOT = ROOT / "model" / "screen_omni_vision_ms"
STATIC_ROOT = ROOT / "static" / "models" / "screen_omni_vision_ms"

LABELS = [
    "chat_todo",
    "bill_record",
    "investment_import",
    "shopping_activity",
    "entertainment_activity",
    "phone_action",
    "screen_summary",
]
LABEL_META = {
    "chat_todo": {"kind": "chat", "action": "create_todos", "title": "聊天待办"},
    "bill_record": {"kind": "bill", "action": "create_transaction", "title": "账单记账"},
    "investment_import": {"kind": "investment", "action": "import_investment", "title": "基金持仓导入"},
    "shopping_activity": {"kind": "activity", "action": "record_activity", "title": "购物浏览"},
    "entertainment_activity": {"kind": "activity", "action": "record_activity", "title": "娱乐活动"},
    "phone_action": {"kind": "phone_action", "action": "suggest_phone_action", "title": "手机操作建议"},
    "screen_summary": {"kind": "general", "action": "summarize_screen", "title": "屏幕摘要"},
}
GRID = 32
FEATURE_DIM = GRID * GRID * 3
HIDDEN_DIM = 96
SEED = 20260622


def rgb(hex_value: str):
    value = hex_value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


PALETTE = {
    "bg": [rgb("#f8f8f8"), rgb("#f6f1e8"), rgb("#ffffff"), rgb("#f3f5f8")],
    "line": [rgb("#e5e7eb"), rgb("#d8dee9"), rgb("#ead8c4")],
    "dark": [rgb("#222222"), rgb("#344054"), rgb("#4c2d12")],
    "green": [rgb("#95ec69"), rgb("#22c55e"), rgb("#dcfce7")],
    "blue": [rgb("#dbeafe"), rgb("#3b82f6"), rgb("#e0f2fe")],
    "red": [rgb("#ef4444"), rgb("#fee2e2"), rgb("#dc2626")],
    "orange": [rgb("#f59e0b"), rgb("#ffedd5"), rgb("#fb923c")],
}


def rand_color(group: str):
    return random.choice(PALETTE[group])


def draw_text_bar(draw: ImageDraw.ImageDraw, xy, w, h, fill=None):
    x, y = xy
    fill = fill or rand_color("line")
    radius = max(1, h // 3)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill)


def base_canvas():
    w = random.randint(180, 240)
    h = random.randint(340, 460)
    image = Image.new("RGB", (w, h), rand_color("bg"))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, w, random.randint(20, 36)], fill=rgb("#ffffff"))
    for i in range(3):
        draw_text_bar(draw, (12 + i * 22, 8), random.randint(8, 16), 4, rand_color("line"))
    return image, draw, w, h


def draw_chat():
    image, draw, w, h = base_canvas()
    draw.rectangle([0, 0, w, 44], fill=rgb("#f7f7f7"))
    draw_text_bar(draw, (w // 2 - 34, 17), 68, 8, rand_color("dark"))
    y = 58
    for i in range(random.randint(7, 11)):
        left = i % 3 != 1
        avatar_x = 12 if left else w - 28
        draw.ellipse([avatar_x, y, avatar_x + 18, y + 18], fill=rand_color("blue" if left else "green"))
        bw = random.randint(52, min(138, w - 64))
        bh = random.randint(20, 42)
        bx = 36 if left else w - 36 - bw
        bubble = random.choice(PALETTE["green"] if not left else [rgb("#ffffff"), rgb("#e8f1ff")])
        draw.rounded_rectangle([bx, y - 2, bx + bw, y + bh], radius=10, fill=bubble)
        for k in range(random.randint(1, 3)):
            draw_text_bar(draw, (bx + 8, y + 6 + k * 10), random.randint(28, max(30, bw - 18)), 4, rand_color("line"))
        y += bh + random.randint(8, 18)
        if y > h - 70:
            break
    draw.rectangle([0, h - 44, w, h], fill=rgb("#ffffff"))
    draw.rounded_rectangle([14, h - 34, w - 48, h - 10], radius=12, fill=rgb("#f1f5f9"))
    return image


def draw_bill():
    image, draw, w, h = base_canvas()
    accent = random.choice([rgb("#22c55e"), rgb("#1677ff"), rgb("#fa8c16"), rgb("#00b578")])
    draw.rectangle([0, 0, w, random.randint(54, 86)], fill=accent)
    draw.ellipse([w // 2 - 22, 54, w // 2 + 22, 98], fill=rgb("#ffffff"))
    draw_text_bar(draw, (w // 2 - 50, 112), 100, 10, rand_color("dark"))
    draw_text_bar(draw, (w // 2 - 70, 138), 140, 18, rand_color("dark"))
    y = 182
    for i in range(random.randint(6, 10)):
        draw.rectangle([12, y, w - 12, y + 1], fill=rand_color("line"))
        draw_text_bar(draw, (20, y + 14), random.randint(34, 70), 5, rand_color("line"))
        draw_text_bar(draw, (w - random.randint(88, 120), y + 12), random.randint(54, 92), 6, rand_color("dark" if i < 3 else "line"))
        y += random.randint(28, 38)
        if y > h - 20:
            break
    return image


def draw_investment():
    image, draw, w, h = base_canvas()
    draw.rectangle([0, 0, w, 48], fill=random.choice([rgb("#ffffff"), rgb("#fef3c7")]))
    draw_text_bar(draw, (16, 18), 52, 8, rand_color("dark"))
    draw.rounded_rectangle([12, 62, w - 12, 130], radius=12, fill=rgb("#ffffff"))
    for i in range(3):
        x = 26 + i * ((w - 52) // 3)
        draw_text_bar(draw, (x, 78), random.randint(36, 58), 5, rand_color("line"))
        draw_text_bar(draw, (x, 96), random.randint(42, 68), 8, random.choice(PALETTE["red"] + PALETTE["green"]))
    y = 150
    for r in range(random.randint(5, 8)):
        draw.rounded_rectangle([12, y, w - 12, y + 42], radius=8, fill=rgb("#ffffff"))
        draw_text_bar(draw, (22, y + 9), random.randint(52, 96), 6, rand_color("dark"))
        draw_text_bar(draw, (22, y + 25), random.randint(34, 62), 5, rand_color("line"))
        draw_text_bar(draw, (w - 104, y + 9), random.randint(44, 82), 6, random.choice(PALETTE["red"] + PALETTE["green"]))
        draw_text_bar(draw, (w - 92, y + 25), random.randint(36, 70), 5, rand_color("line"))
        y += random.randint(48, 56)
        if y > h - 44:
            break
    return image


def draw_shopping():
    image, draw, w, h = base_canvas()
    draw.rectangle([0, 0, w, 48], fill=random.choice([rgb("#ff5000"), rgb("#e1251b"), rgb("#f97316")]))
    draw.rounded_rectangle([14, 14, w - 14, 34], radius=10, fill=rgb("#ffffff"))
    draw_text_bar(draw, (28, 22), random.randint(70, w - 70), 5, rand_color("line"))
    card_w = (w - 42) // 2
    y = 62
    for row in range(3):
        for col in range(2):
            x = 14 + col * (card_w + 14)
            draw.rounded_rectangle([x, y, x + card_w, y + 94], radius=8, fill=rgb("#ffffff"))
            draw.rectangle([x + 8, y + 8, x + card_w - 8, y + 54], fill=random.choice(PALETTE["orange"] + PALETTE["blue"]))
            draw_text_bar(draw, (x + 8, y + 63), random.randint(34, max(35, card_w - 18)), 5, rand_color("dark"))
            draw_text_bar(draw, (x + 8, y + 78), random.randint(24, max(28, card_w - 28)), 6, random.choice(PALETTE["red"]))
        y += 108
        if y > h - 72:
            break
    return image


def draw_entertainment():
    image, draw, w, h = base_canvas()
    mode = random.choice(["video", "music", "short_video"])
    if mode == "music":
        draw.rectangle([0, 0, w, h], fill=random.choice([rgb("#111827"), rgb("#0f172a"), rgb("#18181b")]))
        draw.ellipse([w // 2 - 52, 70, w // 2 + 52, 174], fill=random.choice(PALETTE["blue"] + PALETTE["orange"]))
        draw_text_bar(draw, (w // 2 - 58, 200), 116, 8, rgb("#ffffff"))
        draw_text_bar(draw, (w // 2 - 42, 222), 84, 5, rand_color("line"))
        y = 270
        for _ in range(4):
            draw_text_bar(draw, (24, y), random.randint(76, w - 52), 5, rand_color("line"))
            y += 26
    else:
        draw.rectangle([0, 0, w, h], fill=rgb("#101010"))
        draw.rectangle([0, 42, w, min(h - 86, h // 2 + 70)], fill=random.choice([rgb("#1f2937"), rgb("#111827"), rgb("#2d1b0e")]))
        draw.polygon([(w // 2 - 16, h // 3 - 24), (w // 2 - 16, h // 3 + 24), (w // 2 + 24, h // 3)], fill=rgb("#ffffff"))
        y = min(h - 150, h // 2 + 88)
        for _ in range(5):
            draw_text_bar(draw, (18, y), random.randint(84, w - 34), 6, rand_color("line"))
            y += 24
            if y > h - 42:
                break
    return image


def draw_phone_action():
    image, draw, w, h = base_canvas()
    mode = random.choice(["settings", "map", "form"])
    if mode == "settings":
        y = 58
        for _ in range(9):
            draw.rectangle([14, y, w - 14, y + 34], fill=rgb("#ffffff"))
            draw_text_bar(draw, (26, y + 13), random.randint(54, 112), 6, rand_color("dark"))
            draw.ellipse([w - 48, y + 10, w - 28, y + 30], fill=rand_color("line"))
            y += 42
    elif mode == "map":
        draw.rectangle([0, 48, w, h - 54], fill=rgb("#dbeafe"))
        for _ in range(9):
            x = random.randint(0, w - 20)
            y = random.randint(54, h - 80)
            draw.line([x, y, min(w, x + random.randint(40, 120)), min(h, y + random.randint(-30, 40))], fill=rgb("#ffffff"), width=4)
        draw.ellipse([w // 2 - 12, h // 2 - 12, w // 2 + 12, h // 2 + 12], fill=rgb("#ef4444"))
    else:
        y = 58
        for _ in range(6):
            draw_text_bar(draw, (18, y), random.randint(80, w - 42), 6, rand_color("dark"))
            draw.rounded_rectangle([18, y + 16, w - 18, y + 44], radius=8, fill=rgb("#ffffff"))
            y += 62
    return image


def draw_summary():
    image, draw, w, h = base_canvas()
    y = 54
    for _ in range(random.randint(5, 9)):
        kind = random.choice(["feed", "card", "list"])
        if kind == "feed":
            draw.rounded_rectangle([12, y, w - 12, y + 72], radius=10, fill=rgb("#ffffff"))
            draw.rectangle([22, y + 12, 72, y + 60], fill=random.choice(PALETTE["blue"] + PALETTE["orange"]))
            draw_text_bar(draw, (84, y + 16), random.randint(56, w - 108), 6, rand_color("dark"))
            draw_text_bar(draw, (84, y + 34), random.randint(70, w - 100), 5, rand_color("line"))
            y += 86
        elif kind == "card":
            draw.rounded_rectangle([12, y, w - 12, y + 56], radius=10, fill=rgb("#ffffff"))
            draw_text_bar(draw, (26, y + 17), random.randint(80, w - 52), 8, rand_color("line"))
            y += 70
        else:
            draw_text_bar(draw, (18, y + 8), random.randint(80, w - 36), 6, rand_color("dark"))
            draw.rectangle([14, y + 28, w - 14, y + 29], fill=rand_color("line"))
            y += 42
        if y > h - 50:
            break
    return image


GENERATORS = {
    "chat_todo": draw_chat,
    "bill_record": draw_bill,
    "investment_import": draw_investment,
    "shopping_activity": draw_shopping,
    "entertainment_activity": draw_entertainment,
    "phone_action": draw_phone_action,
    "screen_summary": draw_summary,
}


def augment(image: Image.Image) -> Image.Image:
    if random.random() < 0.15:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.55:
        arr = np.asarray(image).astype(np.float32)
        arr = arr * random.uniform(0.84, 1.12) + random.uniform(-10.0, 10.0)
        arr += np.random.normal(0.0, random.uniform(0.0, 3.0), arr.shape)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")
    return image


def image_features(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((GRID, GRID), Image.Resampling.NEAREST)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr.reshape(-1)


def load_real_rows():
    path = PROCESSED_ROOT / "minimind_o_zemo_screen_i2t.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = (item.get("intent") or "").strip()
            image_path = Path(item.get("image_path") or "")
            if label in LABELS and image_path.exists():
                rows.append({"label": label, "image_path": image_path, "source": "zemo_phone_scene"})
    return rows


def build_dataset(per_label=200, real_repeat=30):
    xs = []
    ys = []
    rows = []
    real_rows = load_real_rows()
    for row in real_rows:
        with Image.open(row["image_path"]) as src:
            base_features = image_features(src.convert("RGB"))
        for _ in range(real_repeat):
            features = base_features.copy()
            if _ > 0:
                features = features * random.uniform(0.88, 1.12)
                features += np.random.normal(0.0, 0.015, features.shape).astype(np.float32)
                features = np.clip(features, 0.0, 1.0).astype(np.float32)
            xs.append(features)
            ys.append(LABELS.index(row["label"]))
            rows.append({"label": row["label"], "source": row["source"], "image_path": str(row["image_path"])})
    for label in LABELS:
        for i in range(per_label):
            image = augment(GENERATORS[label]())
            xs.append(image_features(image))
            ys.append(LABELS.index(label))
            rows.append({"label": label, "source": "synthetic_ui_vision"})
    x = np.stack(xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)
    order = np.arange(y.shape[0])
    rng = np.random.default_rng(SEED)
    rng.shuffle(order)
    return x[order], y[order], [rows[i] for i in order]


def one_hot(y, classes):
    out = np.zeros((y.shape[0], classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def train_mlp(x_train, y_train, x_val, y_val):
    rng = np.random.default_rng(SEED)
    n, d = x_train.shape
    h = HIDDEN_DIM
    c = len(LABELS)
    w1 = (rng.normal(0, 1.0 / np.sqrt(d), (d, h))).astype(np.float32)
    b1 = np.zeros((h,), dtype=np.float32)
    w2 = (rng.normal(0, 1.0 / np.sqrt(h), (h, c))).astype(np.float32)
    b2 = np.zeros((c,), dtype=np.float32)
    lr = 0.018
    batch = 128
    best = None
    best_acc = -1.0
    y_train_oh = one_hot(y_train, c)
    for epoch in range(1, 81):
        order = rng.permutation(n)
        total_loss = 0.0
        for start in range(0, n, batch):
            idx = order[start : start + batch]
            xb = x_train[idx]
            yb = y_train_oh[idx]
            z1 = xb @ w1 + b1
            a1 = np.maximum(z1, 0)
            logits = a1 @ w2 + b2
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            prob = exp / exp.sum(axis=1, keepdims=True)
            total_loss += float((-yb * np.log(prob + 1e-7)).sum())
            grad = (prob - yb) / xb.shape[0]
            gw2 = a1.T @ grad
            gb2 = grad.sum(axis=0)
            ga1 = grad @ w2.T
            gz1 = ga1 * (z1 > 0)
            gw1 = xb.T @ gz1
            gb1 = gz1.sum(axis=0)
            w1 -= lr * gw1.astype(np.float32)
            b1 -= lr * gb1.astype(np.float32)
            w2 -= lr * gw2.astype(np.float32)
            b2 -= lr * gb2.astype(np.float32)
        acc, val_loss, _ = evaluate((w1, b1, w2, b2), x_val, y_val)
        if acc > best_acc:
            best_acc = acc
            best = tuple(v.copy() for v in (w1, b1, w2, b2))
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch={epoch:03d} loss={total_loss / n:.4f} val_loss={val_loss:.4f} val_acc={acc:.4f}")
    return best


def evaluate(params, x, y):
    w1, b1, w2, b2 = params
    logits = np.maximum(x @ w1 + b1, 0) @ w2 + b2
    pred = logits.argmax(axis=1)
    acc = float((pred == y).mean())
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    prob = exp / exp.sum(axis=1, keepdims=True)
    loss = float((-np.log(prob[np.arange(y.shape[0]), y] + 1e-7)).mean())
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for t, p in zip(y.tolist(), pred.tolist()):
        cm[t, p] += 1
    return acc, loss, cm


def build_real_eval_dataset():
    xs = []
    ys = []
    for row in load_real_rows():
        with Image.open(row["image_path"]) as src:
            xs.append(image_features(src.convert("RGB")))
        ys.append(LABELS.index(row["label"]))
    if len(xs) == 0:
        return None, None
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def export_onnx(params, path: Path):
    w1, b1, w2, b2 = params
    input_info = helper.make_tensor_value_info("image_features", TensorProto.FLOAT, [1, 1, 1, FEATURE_DIM])
    output_info = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, len(LABELS)])
    nodes = [
        helper.make_node("Flatten", ["image_features"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "w1", "b1"], ["hidden_pre"], alpha=1.0, beta=1.0, transB=0),
        helper.make_node("Relu", ["hidden_pre"], ["hidden"]),
        helper.make_node("Gemm", ["hidden", "w2", "b2"], ["logits"], alpha=1.0, beta=1.0, transB=0),
    ]
    graph = helper.make_graph(
        nodes,
        "zemo_screen_omni_vision_intent",
        [input_info],
        [output_info],
        [
            numpy_helper.from_array(w1.astype(np.float32), "w1"),
            numpy_helper.from_array(b1.astype(np.float32), "b1"),
            numpy_helper.from_array(w2.astype(np.float32), "w2"),
            numpy_helper.from_array(b2.astype(np.float32), "b2"),
        ],
    )
    model = helper.make_model(graph, producer_name="zemo-screen-omni-vision")
    model.opset_import[0].version = 13
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATIC_ROOT.mkdir(parents=True, exist_ok=True)
    x, y, rows = build_dataset()
    split = int(x.shape[0] * 0.9)
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    params = train_mlp(x_train, y_train, x_val, y_val)
    acc, loss, cm = evaluate(params, x_val, y_val)
    real_x, real_y = build_real_eval_dataset()
    real_acc = None
    real_cm = None
    if real_x is not None and real_y is not None:
        real_acc, _real_loss, real_cm = evaluate(params, real_x, real_y)
    onnx_path = OUT_ROOT / "screen_omni_vision_intent.onnx"
    export_onnx(params, onnx_path)
    label_map = {
        "format": "zemo_screen_omni_vision_intent_ms_v1",
        "labels": LABELS,
        "label_meta": LABEL_META,
        "image_feature_dim": FEATURE_DIM,
        "image_grid": [GRID, GRID],
        "input_name": "image_features",
        "output_name": "logits",
    }
    config = {
        "format": "zemo_screen_omni_vision_intent_config_v1",
        "input_shape": [1, 1, 1, FEATURE_DIM],
        "input_data_format": "NCHW",
        "feature_type": "pixel_grid_rgb_nearest_32x32",
        "labels": LABELS,
    }
    report = {
        "format": "zemo_screen_omni_vision_intent_train_report_v1",
        "samples": int(x.shape[0]),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "real_image_files": len(load_real_rows()),
        "eval_accuracy": acc,
        "eval_loss": loss,
        "confusion_matrix": cm.tolist(),
        "real_eval_accuracy": real_acc,
        "real_confusion_matrix": None if real_cm is None else real_cm.tolist(),
        "image_feature_dim": FEATURE_DIM,
        "image_grid": [GRID, GRID],
        "onnx": str(onnx_path),
    }
    (OUT_ROOT / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "README.md").write_text(
        "# ZeMo Screen Omni Vision Intent MS\n\n"
        "截图图片 32x32 RGB 采样特征 -> 屏幕意图分类。用于在 OCR 关闭或文本不可用时判断账单、聊天、基金/资产、普通屏幕等页面类型。\n",
        encoding="utf-8",
    )
    for name in ["screen_omni_vision_intent.onnx", "label_map.json", "model_config.json", "train_report.json", "README.md"]:
        shutil.copy2(OUT_ROOT / name, STATIC_ROOT / name)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
