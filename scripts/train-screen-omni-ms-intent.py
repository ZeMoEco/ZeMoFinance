import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "model" / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
OUT_ROOT = ROOT / "model" / "screen_omni_ms"
STATIC_ROOT = ROOT / "static" / "models" / "screen_omni_ms"

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
FEATURE_DIM = 1024
SEED = 20260621


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\[\[ocr:[^\]]+\]\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_hash(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) % 2147483647
    return value


def is_feature_separator(ch: str) -> bool:
    if ch.isspace():
        return True
    return ch in ";:,.，。；：/\\|()（）[]【】{}<>《》=+-*_'\""


def push_feature(vec: np.ndarray, token: str) -> int:
    vec[stable_hash(token) % vec.shape[0]] += 1.0
    return 1


def push_word(vec: np.ndarray, word: str) -> int:
    if not word:
        return 0
    count = 0
    if len(word) <= 32:
        count += push_feature(vec, "w:" + word)
    if len(word) > 3:
        count += push_feature(vec, "p:" + word[:3])
        count += push_feature(vec, "s:" + word[-3:])
    return count


def hash_features(text: str, dim: int = FEATURE_DIM) -> np.ndarray:
    value = normalize_text(text)
    vec = np.zeros(dim, dtype=np.float32)
    previous = ""
    word = ""
    count = 0
    for ch in value:
        if not ch.isspace():
            count += push_feature(vec, "u:" + ch)
            if previous:
                count += push_feature(vec, "b:" + previous + ch)
        if is_feature_separator(ch):
            count += push_word(vec, word)
            word = ""
        else:
            word += ch
        previous = "" if ch.isspace() else ch
    count += push_word(vec, word)
    if count > 0:
        vec *= 1.0 / math.sqrt(count)
    return vec.reshape(1, 1, dim)


def add(rows, label: str, text: str, source: str):
    rows.append({"label": label, "text": text, "source": source})


def build_business_samples():
    rows = []
    names = ["张三", "李四", "王敏", "产品群", "财务", "客户A", "同事", "老板"]
    todo_verbs = ["提交报销材料", "确认基金持仓截图", "处理付款申请", "跟进客户回款", "整理会议纪要", "缴费", "还款", "转账给供应商", "更新待办"]
    todo_times = ["今天 18:00", "明天 9:30", "后天上午", "今晚", "周五下午", "下周一 10:00", "6月28日"]
    for name in names:
        for verb in todo_verbs:
            for due in todo_times:
                add(rows, "chat_todo", f"群聊 项目组\n{name}: {due} 记得{verb}\n收到请回复", "synthetic_chat")
                add(rows, "chat_todo", f"微信聊天\n{name}: 麻烦{due}前{verb}\n我: 好的", "synthetic_chat")

    apps = ["微信支付", "支付宝", "招商银行", "云闪付", "美团", "京东", "工商银行"]
    merchants = ["便利店", "星巴克", "滴滴出行", "盒马", "房租", "工资", "退款", "基金申购", "餐饮"]
    amounts = ["12.80", "35.00", "128.45", "500.00", "1200.00", "6888.88", "9.90"]
    modes = ["支付成功", "付款", "交易详情", "账单", "已收款", "退款到账", "扣款成功"]
    for app in apps:
        for merchant in merchants:
            for amount in amounts:
                for mode in modes:
                    add(rows, "bill_record", f"{app}\n{mode}\n金额 ¥{amount}\n商户 {merchant}\n付款方式 零钱/银行卡\n交易时间 2026-06-21 12:30", "synthetic_bill")
                    add(rows, "bill_record", f"{app}\n{mode}\n金额 ¥{amount}\n商户 {merchant}\n付款方式 余额", "synthetic_bill_short")
                    add(rows, "bill_record", f"{app} {mode} ¥{amount} {merchant} 付款方式 银行卡", "synthetic_bill_short")
                    add(rows, "bill_record", f"账单详情\n{merchant}\n实付 {amount} 元\n支付方式 {app}", "synthetic_bill_short")

    funds = ["沪深300ETF", "中证红利基金", "易方达蓝筹精选", "华夏能源革新", "纳斯达克100ETF", "招商中证白酒", "天弘余额宝"]
    codes = ["510300", "090010", "005827", "003834", "513100", "161725", "000198"]
    providers = ["支付宝基金", "天天基金", "招商证券", "华泰证券", "蛋卷基金", "证券持仓"]
    for provider in providers:
        for fund, code in zip(funds, codes):
            add(rows, "investment_import", f"{provider}\n持仓\n{fund} {code}\n持有市值 12345.67\n持有收益 +234.56\n收益率 +3.21%\n持有份额 1000.00\n最新净值 1.2345", "synthetic_investment")
            add(rows, "investment_import", f"基金列表\n名称 {fund}\n代码 {code}\n市值 8888.00\n盈亏 -56.78\n净值 0.9876", "synthetic_investment")

    shops = ["淘宝", "天猫", "京东", "拼多多", "闲鱼", "小米商城", "1688", "千牛"]
    goods = ["手机壳", "运动鞋", "充电器", "牛奶", "耳机", "电脑桌", "衣服", "家用纸巾"]
    shopping_scenes = ["商品详情", "购物车", "店铺首页", "搜索结果", "订单列表", "优惠券", "猜你喜欢"]
    for app in shops:
        for scene in shopping_scenes:
            for good in goods:
                add(rows, "shopping_activity", f"{app}\n{scene}\n{good}\n加入购物车\n领券下单\n店铺 评价 物流", "synthetic_shopping")
                add(rows, "shopping_activity", f"{app} 浏览商品 {good} 价格 优惠券 收藏 店铺客服", "synthetic_shopping_short")

    video_apps = ["哔哩哔哩", "抖音", "腾讯视频", "爱奇艺", "短剧", "优酷"]
    music_apps = ["QQ音乐", "汽水音乐", "网易云音乐", "华为音乐"]
    shows = ["电影", "电视剧", "综艺", "番剧", "直播", "短视频", "课程视频"]
    songs = ["歌单", "专辑", "歌曲", "播放列表", "歌词", "电台"]
    for app in video_apps:
        for show in shows:
            add(rows, "entertainment_activity", f"{app}\n正在播放 {show}\n暂停 倍速 弹幕 评论 收藏 分享", "synthetic_entertainment_video")
            add(rows, "entertainment_activity", f"{app} 推荐视频 {show} 第12集 继续播放 点赞 投币 收藏", "synthetic_entertainment_video")
    for app in music_apps:
        for song in songs:
            add(rows, "entertainment_activity", f"{app}\n正在播放 {song}\n上一首 下一首 暂停 循环 歌词", "synthetic_entertainment_music")
            add(rows, "entertainment_activity", f"{app} {song} 推荐 每日30首 播放列表 收藏", "synthetic_entertainment_music")

    general = [
        "设置\n账号与安全\n隐私\n通用",
        "首页\n搜索\n推荐内容\n消息",
        "天气\n今天多云\n空气质量良好",
        "浏览器\n搜索结果\n新闻",
        "日历\n本周安排\n会议提醒",
        "相册\n照片\n最近项目",
        "备忘录\n文件夹\n编辑",
    ]
    for item in general:
        for _ in range(80):
            add(rows, "screen_summary", item, "synthetic_general")
    return rows


def build_public_samples(max_summary=1800, max_phone=1400):
    rows = []
    for item in load_jsonl(PROCESSED_ROOT / "screen_summary_text_only.jsonl")[:max_summary]:
        add(rows, "screen_summary", f"{item.get('instruction', '')}\n{item.get('output', '')}", item.get("source", "screen2words"))
    for name in ["screen_action_sft.jsonl", "mobile_actions_tool_sft.jsonl"]:
        for item in load_jsonl(PROCESSED_ROOT / name)[:max_phone]:
            add(rows, "phone_action", f"{item.get('instruction', '')}\n{item.get('output', '')}", item.get("source", name))
    return rows


class ScreenIntentNet(nn.Module):
    def __init__(self, feature_dim: int, label_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(96, label_count),
        )

    def forward(self, features):
        return self.net(features)


def split_rows(rows):
    random.Random(SEED).shuffle(rows)
    by_label = {label: [] for label in LABELS}
    for row in rows:
        if row["label"] in by_label:
            by_label[row["label"]].append(row)
    train, val = [], []
    for label, items in by_label.items():
        split = max(1, int(len(items) * 0.9))
        train.extend(items[:split])
        val.extend(items[split:])
    random.Random(SEED + 1).shuffle(train)
    random.Random(SEED + 2).shuffle(val)
    return train, val


def tensorize(rows):
    x = np.stack([hash_features(row["text"])[0] for row in rows]).astype(np.float32)
    y = np.array([LABELS.index(row["label"]) for row in rows], dtype=np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def evaluate(model, x, y, loss_fn=None):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        loss = loss_fn(logits, y).item() if loss_fn is not None else 0.0
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
        for t, p in zip(y.cpu().numpy().tolist(), pred.cpu().numpy().tolist()):
            cm[t, p] += 1
    return acc, loss, cm


def export_onnx(model, path: Path):
    model.eval()
    dummy = torch.zeros(1, 1, 1, FEATURE_DIM, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATIC_ROOT.mkdir(parents=True, exist_ok=True)

    rows = build_business_samples() + build_public_samples()
    rows = [row for row in rows if row["label"] in LABELS and normalize_text(row["text"])]
    train_rows, val_rows = split_rows(rows)
    x_train, y_train = tensorize(train_rows)
    x_val, y_val = tensorize(val_rows)

    model = ScreenIntentNet(FEATURE_DIM, len(LABELS))
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    label_counts = Counter(row["label"] for row in train_rows)
    weights = []
    for label in LABELS:
        weights.append(len(train_rows) / max(1, label_counts[label]))
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    best_acc = -1.0
    best_loss = 1e9
    best_state = None
    for epoch in range(1, 26):
        model.train()
        total = 0.0
        for bx, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()
            total += loss.item() * bx.shape[0]
        acc, val_loss, _ = evaluate(model, x_val, y_val, loss_fn)
        if acc > best_acc or (acc >= best_acc and val_loss < best_loss):
            best_acc = acc
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"epoch={epoch:02d} loss={total / len(train_rows):.4f} val_loss={val_loss:.4f} val_acc={acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    acc, val_loss, cm = evaluate(model, x_val, y_val, loss_fn)

    onnx_path = OUT_ROOT / "screen_omni_intent.onnx"
    export_onnx(model, onnx_path)

    torch.save(model.state_dict(), OUT_ROOT / "screen_omni_intent.pt")
    label_map = {
        "format": "zemo_screen_omni_intent_ms_v1",
        "labels": LABELS,
        "label_meta": LABEL_META,
        "hash_feature_dim": FEATURE_DIM,
        "input_name": "features",
        "output_name": "logits",
    }
    config = {
        "format": "zemo_screen_omni_intent_config_v1",
        "input_shape": [1, 1, 1, FEATURE_DIM],
        "input_data_format": "NCHW",
        "feature_type": "stable_hash_text",
        "labels": LABELS,
    }
    report = {
        "format": "zemo_screen_omni_intent_train_report_v1",
        "samples": len(rows),
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "eval_accuracy": acc,
        "eval_loss": val_loss,
        "confusion_matrix": cm.tolist(),
        "feature_dim": FEATURE_DIM,
        "onnx": str(onnx_path),
    }
    (OUT_ROOT / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "model_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "README.md").write_text(
        "# ZeMo Screen Omni Intent MS\n\n"
        "OCR 文本 hash 特征 -> 屏幕意图分类，用于账单记账、基金持仓导入、聊天待办和普通屏幕摘要分流。\n",
        encoding="utf-8",
    )

    for name in ["screen_omni_intent.onnx", "label_map.json", "model_config.json", "train_report.json", "README.md"]:
        shutil.copy2(OUT_ROOT / name, STATIC_ROOT / name)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
