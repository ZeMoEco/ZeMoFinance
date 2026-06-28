import argparse
import json
import random
import re
from pathlib import Path


# 这个脚本负责把“手机金融页面 OCR 结果”整理成 Layout-Aware Encoder 需要的整页数据。
# 输出格式是一行一页 JSON：
# {
#   "page_type": "investment_holding",
#   "width": 1080,
#   "height": 2400,
#   "items": [
#     {"text": "纳斯达克", "bbox": [80, 420, 250, 456], "label": "asset_name"}
#   ]
# }
#
# 训练脚本会把 text 当作语义输入，把 bbox 当作布局输入。

DEFAULT_OUTPUT = r"E:\CamXAll\ZEMO\Data\model\finance_layout_train.jsonl"

LABELS = [
    "merchant",
    "counterparty",
    "amount",
    "income_amount",
    "expense_amount",
    "balance",
    "date_time",
    "payment_method",
    "order_id",
    "bank_card",
    "transaction_type",
    "status",
    "asset_name",
    "asset_code",
    "market_value",
    "profit",
    "profit_rate",
    "holding",
    "available",
    "quantity",
    "price",
    "net_value",
    "shares",
    "other",
]

MERCHANTS = ["星巴克", "美团外卖", "滴滴出行", "盒马鲜生", "京东商城", "中国移动", "中石化", "喜茶", "全家便利店"]
COUNTERPARTIES = ["张三", "李四", "王小明", "招商银行", "支付宝", "微信支付", "中国银河证券", "天天基金", "华泰证券"]
PAYMENT_METHODS = ["余额宝", "微信零钱", "招商银行储蓄卡(1234)", "工商银行信用卡(8899)", "支付宝余额"]
BANKS = ["招商银行", "工商银行", "建设银行", "农业银行", "交通银行", "中国银行"]
ASSETS = [
    ("纳斯达克", "513300"),
    ("纳100ETF", "159696"),
    ("恒指科技", "513180"),
    ("沪深300ETF", "510300"),
    ("创业板ETF", "159915"),
    ("中证红利ETF", "515080"),
    ("天弘余额宝货币", "000198"),
    ("易方达蓝筹精选", "005827"),
    ("贵州茅台", "600519"),
    ("宁德时代", "300750"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare page-level layout data for finance OCR field classification.")
    parser.add_argument("--input", nargs="*", default=[], help="OCR txt/jsonl files. Empty means generate synthetic seed pages.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-scene", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weak-label", action="store_true", help="Use simple rules to label unlabeled OCR txt items.")
    return parser.parse_args()


def money(min_value=1, max_value=50000):
    return f"{random.uniform(min_value, max_value):.2f}"


def percent():
    return f"{random.uniform(-8, 8):.3f}%"


def date_text():
    return f"2026-{random.randint(1, 12):02d}-{random.randint(1, 28):02d} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}"


def order_id():
    prefix = random.choice(["20260621", "420000", "P202606", "T202606", "ALI2026", "WX2026"])
    return prefix + "".join(str(random.randint(0, 9)) for _ in range(random.randint(8, 16)))


def jitter(value, radius=8):
    return int(value + random.randint(-radius, radius))


def text_width(text):
    # 粗略模拟 OCR 框宽度：中文字符更宽，数字和英文略窄。
    width = 0
    for ch in text:
        width += 28 if "\u4e00" <= ch <= "\u9fff" else 17
    return max(42, min(420, width + random.randint(10, 26)))


def add_item(items, text, x, y, label, h=38):
    x0 = jitter(x)
    y0 = jitter(y, 5)
    x1 = x0 + text_width(text)
    y1 = y0 + h + random.randint(-3, 4)
    items.append({"text": text, "bbox": [x0, y0, x1, y1], "label": label})


def page(page_type, items, width=1080, height=2400):
    # 统一按从上到下、从左到右排序，训练时模型仍会看到 bbox。
    items.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return {"page_type": page_type, "width": width, "height": height, "items": items}


def seed_payment_detail():
    merchant = random.choice(MERCHANTS)
    counterparty = random.choice(COUNTERPARTIES)
    amount = money(1, 3000)
    method = random.choice(PAYMENT_METHODS)
    oid = order_id()
    dt = date_text()
    status = random.choice(["支付成功", "交易成功", "已完成", "扣款成功", "退款成功"])
    tx_type = random.choice(["消费", "转账", "付款", "退款", "充值"])
    items = []
    add_item(items, "账单详情", 46, 90, "other")
    add_item(items, status, 420, 170, "status")
    add_item(items, f"-{amount}", 420, 240, "expense_amount")
    rows = [
        ("商户", merchant, "merchant"),
        ("对方账户", counterparty, "counterparty"),
        ("交易类型", tx_type, "transaction_type"),
        ("支付方式", method, "payment_method"),
        ("付款时间", dt, "date_time"),
        ("订单号", oid, "order_id"),
    ]
    y = 390
    for key, value, label in rows:
        add_item(items, key, 70, y, "other")
        add_item(items, value, 420, y, label)
        y += 86
    add_item(items, "对此订单有疑问", 365, y + 80, "other")
    return page("payment_detail", items)


def seed_payment_list():
    items = []
    add_item(items, "账单", 46, 86, "other")
    for i, tab in enumerate(["全部", "支出", "收入", "筛选"]):
        add_item(items, tab, 70 + i * 210, 170, "other")
    y = 300
    for _ in range(random.randint(4, 8)):
        merchant = random.choice(MERCHANTS)
        method = random.choice(PAYMENT_METHODS)
        amount = money(1, 1200)
        dt = date_text()
        tx_type = random.choice(["餐饮", "交通", "购物", "生活缴费", "娱乐", "退款"])
        sign = random.choice(["-", "+"])
        add_item(items, merchant, 70, y, "merchant")
        add_item(items, tx_type, 70, y + 42, "transaction_type")
        add_item(items, method, 280, y + 42, "payment_method")
        add_item(items, dt, 70, y + 82, "date_time")
        add_item(items, f"{sign}{amount}", 760, y + 22, "income_amount" if sign == "+" else "expense_amount")
        y += 168
    return page("payment_list", items)


def seed_bank_bill():
    bank = random.choice(BANKS)
    items = []
    add_item(items, bank, 46, 80, "counterparty")
    add_item(items, "交易明细", 430, 160, "other")
    y = 290
    for _ in range(random.randint(4, 7)):
        counterparty = random.choice(COUNTERPARTIES)
        amount = money(1, 20000)
        balance = money(100, 100000)
        dt = date_text()
        tx_type = random.choice(["快捷支付", "转账汇款", "工资收入", "基金赎回", "信用卡还款", "消费"])
        card = f"尾号{random.randint(1000, 9999)}"
        sign = random.choice(["收入", "支出"])
        prefix = "+" if sign == "收入" else "-"
        add_item(items, dt, 65, y, "date_time")
        add_item(items, tx_type, 65, y + 48, "transaction_type")
        add_item(items, counterparty, 270, y + 48, "counterparty")
        add_item(items, card, 65, y + 92, "bank_card")
        add_item(items, f"{prefix}{amount}", 770, y + 28, "income_amount" if sign == "收入" else "expense_amount")
        add_item(items, f"余额 {balance}", 760, y + 82, "balance")
        y += 178
    return page("bank_bill", items)


def seed_investment_holding():
    items = []
    add_item(items, "我的资产", 46, 82, "other")
    headers = [("名称", 70), ("市值", 330), ("盈亏", 500), ("持仓/份额", 650), ("代码/净值", 840)]
    for text, x in headers:
        add_item(items, text, x, 210, "other")
    y = 310
    for _ in range(random.randint(3, 7)):
        name, code = random.choice(ASSETS)
        market_value = money(100, 80000)
        profit_text = f"{random.uniform(-3000, 3000):.2f}"
        rate = percent()
        holding = str(random.choice([100, 200, 500, 1000, 1500, 2200, 5000]))
        available = holding if random.random() > 0.25 else str(max(0, int(holding) - random.choice([100, 200, 500])))
        price = f"{random.uniform(0.5, 2800):.3f}"
        net_value = f"{random.uniform(0.6, 8.0):.4f}"
        shares = f"{random.uniform(10, 10000):.2f}"
        add_item(items, name, 70, y, "asset_name")
        add_item(items, code, 70, y + 44, "asset_code")
        add_item(items, market_value, 330, y, "market_value")
        add_item(items, profit_text, 500, y, "profit")
        add_item(items, rate, 500, y + 44, "profit_rate")
        add_item(items, holding, 650, y, "holding")
        add_item(items, available, 650, y + 44, "available")
        add_item(items, shares, 650, y + 86, "shares")
        add_item(items, price, 840, y, "price")
        add_item(items, net_value, 840, y + 44, "net_value")
        y += 156
    return page("investment_holding", items)


def seed_investment_trade():
    items = []
    add_item(items, "交易记录", 46, 82, "other")
    headers = [("名称", 70), ("类型", 310), ("金额", 470), ("数量", 640), ("状态", 800)]
    for text, x in headers:
        add_item(items, text, x, 205, "other")
    y = 305
    for _ in range(random.randint(4, 8)):
        name, code = random.choice(ASSETS)
        amount = money(100, 30000)
        qty = str(random.choice([100, 200, 500, 1000, 2000]))
        price = f"{random.uniform(0.5, 2800):.3f}"
        dt = date_text()
        tx_type = random.choice(["买入", "卖出", "申购", "赎回", "定投", "分红"])
        status = random.choice(["已成交", "已确认", "处理中", "已撤单"])
        add_item(items, name, 70, y, "asset_name")
        add_item(items, code, 70, y + 42, "asset_code")
        add_item(items, tx_type, 310, y, "transaction_type")
        add_item(items, amount, 470, y, "amount")
        add_item(items, qty, 640, y, "quantity")
        add_item(items, price, 640, y + 42, "price")
        add_item(items, status, 800, y, "status")
        add_item(items, dt, 70, y + 84, "date_time")
        add_item(items, order_id(), 520, y + 84, "order_id")
        y += 150
    return page("investment_trade", items)


def generate_seed_pages(samples_per_scene):
    generators = [
        seed_payment_detail,
        seed_payment_list,
        seed_bank_bill,
        seed_investment_holding,
        seed_investment_trade,
    ]
    pages = []
    for generator in generators:
        for _ in range(samples_per_scene):
            pages.append(generator())
    random.shuffle(pages)
    return pages


def normalize_bbox(item):
    if "bbox" in item and isinstance(item["bbox"], list) and len(item["bbox"]) == 4:
        return [float(v) for v in item["bbox"]]
    if all(key in item for key in ["x", "y", "w", "h"]):
        x = float(item["x"])
        y = float(item["y"])
        return [x, y, x + float(item["w"]), y + float(item["h"])]
    if all(key in item for key in ["left", "top", "right", "bottom"]):
        return [float(item["left"]), float(item["top"]), float(item["right"]), float(item["bottom"])]
    raise ValueError(f"Missing bbox: {item}")


def normalize_page(raw, fallback_page_type="unknown", weak_label=False):
    width = int(raw.get("width", 1080))
    height = int(raw.get("height", 2400))
    page_type = str(raw.get("page_type", fallback_page_type))
    items = []
    for item in raw.get("items", []):
        text = str(item.get("text", "")).strip()
        if text == "":
            continue
        label = str(item.get("label", "")).strip()
        if label == "" and weak_label:
            label = weak_label_item(text)
        if label == "":
            label = "other"
        if label not in LABELS:
            raise ValueError(f"Unknown label={label}, text={text}")
        items.append({"text": text, "bbox": normalize_bbox(item), "label": label})
    return page(page_type, items, width, height)


OCR_LINE_PATTERN = re.compile(r"^\[\[ocr:([0-9.\-]+),([0-9.\-]+),([0-9.\-]+),([0-9.\-]+)\]\](.*)$")


def parse_txt_pages(path, weak_label=False):
    # 支持当前工程 OcrService 输出的格式：
    # [[ocr:left,top,width,height]]文本
    # 如果手工标注，可在行尾追加：\tlabel=asset_name 或 \tasset_name
    pages = []
    items = []
    page_index = 0
    for raw_line in path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n"):
        line = raw_line.strip()
        if line in ["---page---", "---PAGE---", "\f"]:
            if items:
                pages.append(page("ocr_page", items))
                items = []
                page_index += 1
            continue
        matched = OCR_LINE_PATTERN.match(line)
        if matched is None:
            continue
        x = float(matched.group(1))
        y = float(matched.group(2))
        w = float(matched.group(3))
        h = float(matched.group(4))
        text, label = split_inline_label(matched.group(5).strip())
        if label == "" and weak_label:
            label = weak_label_item(text)
        if label == "":
            label = "other"
        if label not in LABELS:
            raise ValueError(f"{path}: unknown label={label}, text={text}")
        items.append({"text": text, "bbox": [x, y, x + w, y + h], "label": label})
    if items:
        pages.append(page(f"ocr_page_{page_index}", items))
    return pages


def split_inline_label(value):
    parts = value.rsplit("\t", 1)
    if len(parts) == 2:
        text = parts[0].strip()
        suffix = parts[1].strip()
        if suffix.startswith("label="):
            return text, suffix[len("label="):].strip()
        if suffix in LABELS:
            return text, suffix
    return value, ""


def weak_label_item(text):
    # 弱标签只能用来冷启动，不能替代人工校对。
    value = text.strip()
    if value == "":
        return "other"
    if re.fullmatch(r"\d{6}", value):
        return "asset_code"
    if "%" in value and re.search(r"[-+]?\d", value):
        return "profit_rate"
    if re.search(r"\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}", value) or re.search(r"\d{1,2}:\d{2}", value):
        return "date_time"
    if re.fullmatch(r"[+]\s*[\d,]+(\.\d+)?", value):
        return "income_amount"
    if re.fullmatch(r"[-]\s*[\d,]+(\.\d+)?", value):
        return "expense_amount"
    if re.fullmatch(r"[\d,]+(\.\d+)?", value):
        return "amount"
    if any(key in value for key in ["成功", "完成", "已成交", "已确认", "处理中", "已撤单"]):
        return "status"
    if any(key in value for key in ["买入", "卖出", "申购", "赎回", "转账", "消费", "退款", "充值"]):
        return "transaction_type"
    if any(key in value for key in ["银行卡", "储蓄卡", "信用卡", "余额宝", "微信零钱", "支付宝余额"]):
        return "payment_method"
    if "尾号" in value:
        return "bank_card"
    if re.fullmatch(r"[A-Z0-9]{10,}", value):
        return "order_id"
    return "other"


def load_pages_from_inputs(paths, weak_label=False):
    pages = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    text = line.strip()
                    if text == "":
                        continue
                    raw = json.loads(text)
                    if "items" not in raw:
                        raise ValueError(f"{path}:{line_no} is not page-level layout JSON")
                    pages.append(normalize_page(raw, path.stem, weak_label))
        else:
            pages.extend(parse_txt_pages(path, weak_label))
    return pages


def write_jsonl(pages, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for item in pages:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def label_counts(pages):
    counts = {label: 0 for label in LABELS}
    for p in pages:
        for item in p["items"]:
            counts[item["label"]] = counts.get(item["label"], 0) + 1
    return counts


def main():
    args = parse_args()
    random.seed(args.seed)
    if args.input:
        pages = load_pages_from_inputs(args.input, args.weak_label)
    else:
        pages = generate_seed_pages(args.samples_per_scene)
    if not pages:
        raise ValueError("No layout pages generated")
    write_jsonl(pages, args.output)
    print(f"Wrote {len(pages)} pages: {args.output}")
    print(json.dumps(label_counts(pages), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
