import argparse
import json
import random
import re
from pathlib import Path


DEFAULT_OUTPUT = r"E:\CamXAll\ZEMO\Data\model\finance_field_cls_train.jsonl"

MERCHANTS = ["星巴克", "美团外卖", "滴滴出行", "盒马鲜生", "京东商城", "中国移动", "中石化", "喜茶", "全家便利店", "网易云音乐"]
COUNTERPARTIES = ["张三", "李四", "王小明", "招商银行", "支付宝", "微信支付", "中国银河证券", "天天基金", "华泰证券"]
PAYMENT_METHODS = ["余额宝", "微信零钱", "招商银行储蓄卡(1234)", "工商银行信用卡(8899)", "支付宝余额", "交通银行储蓄卡"]
ORDER_PREFIXES = ["20260621", "420000", "P202606", "T202606", "ALI2026", "WX2026"]
BANKS = ["招商银行", "工商银行", "建设银行", "农业银行", "交通银行", "中国银行", "邮储银行"]
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


def money(min_value=1, max_value=50000):
    return f"{random.uniform(min_value, max_value):.2f}"


def percent():
    value = random.uniform(-8, 8)
    return f"{value:.3f}%"


def date_text():
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"2026-{month:02d}-{day:02d} {hour:02d}:{minute:02d}"


def order_id():
    return random.choice(ORDER_PREFIXES) + "".join(str(random.randint(0, 9)) for _ in range(random.randint(8, 16)))


def card_no():
    return f"尾号{random.randint(1000, 9999)}"


def add(rows, text, label):
    rows.append({"text": text, "label": label})


def augment_compact_rows(rows):
    compact = []
    page_rx = re.compile(r"页面: ([^;]+)")
    header_rx = re.compile(r"表头: ([^;]+)")
    row_rx = re.compile(r"行: ([^;]+)")
    cell_rx = re.compile(r"单元格: ([^;]+)")
    hint_rx = re.compile(r"(?:位置|字段提示): ([^;]+)")
    for row in rows:
        text = row["text"]
        page = match_or_empty(page_rx, text)
        header = match_or_empty(header_rx, text)
        row_text = match_or_empty(row_rx, text)
        cell = match_or_empty(cell_rx, text)
        hint = match_or_empty(hint_rx, text)
        if cell == "":
            continue
        label = row["label"]
        compact.append({
            "text": f"页面: {page}; OCR单元格: {cell}; 区域/列: {hint}; 表头: {header}; 行文本: {row_text}",
            "label": label,
        })
        compact.append({
            "text": f"OCR识别: {cell}; 页面类型: {page}; 周围行: {row_text}; 位置描述: {hint}; 表头文字: {header}",
            "label": label,
        })
    rows.extend(compact)


def match_or_empty(pattern, text):
    matched = pattern.search(text)
    if matched is None:
        return ""
    return matched.group(1).strip()


def add_payment_detail(rows):
    merchant = random.choice(MERCHANTS)
    counterparty = random.choice(COUNTERPARTIES)
    amount = money(1, 3000)
    method = random.choice(PAYMENT_METHODS)
    oid = order_id()
    dt = date_text()
    status = random.choice(["支付成功", "交易成功", "已完成", "扣款成功", "退款成功"])
    tx_type = random.choice(["消费", "转账", "付款", "退款", "充值"])
    context = f"页面: 支付详情; 标题: {status}; 商户: {merchant}; 对方: {counterparty}; 金额: {amount}; 方式: {method}; 时间: {dt}; 订单号: {oid}; 类型: {tx_type}"
    add(rows, f"{context}; 单元格: {merchant}; 字段提示: 商户/收款方", "merchant")
    add(rows, f"{context}; 单元格: {counterparty}; 字段提示: 对方账户/付款方", "counterparty")
    add(rows, f"{context}; 单元格: {amount}; 字段提示: 金额", "amount")
    add(rows, f"{context}; 单元格: {method}; 字段提示: 支付方式", "payment_method")
    add(rows, f"{context}; 单元格: {oid}; 字段提示: 订单号/交易单号", "order_id")
    add(rows, f"{context}; 单元格: {dt}; 字段提示: 创建时间/付款时间", "date_time")
    add(rows, f"{context}; 单元格: {status}; 字段提示: 交易状态", "status")
    add(rows, f"{context}; 单元格: {tx_type}; 字段提示: 交易类型", "transaction_type")
    for value in ["完成", "返回", "账单详情", "对此订单有疑问", "查看往来记录"]:
        add(rows, f"{context}; 单元格: {value}; 字段提示: 页面控件/标题", "other")


def add_payment_list(rows):
    merchant = random.choice(MERCHANTS)
    method = random.choice(PAYMENT_METHODS)
    amount = money(1, 1200)
    dt = date_text()
    tx_type = random.choice(["餐饮", "交通", "购物", "生活缴费", "娱乐", "退款"])
    sign = random.choice(["-", "+"])
    label = "income_amount" if sign == "+" else "expense_amount"
    context = f"页面: 支付列表; 表头: 全部 支出 收入 筛选; 行: {dt} {merchant} {tx_type} {sign}{amount} {method}"
    add(rows, f"{context}; 单元格: {merchant}; 位置: 标题/商户", "merchant")
    add(rows, f"{context}; 单元格: {sign}{amount}; 位置: 金额列", label)
    add(rows, f"{context}; 单元格: {amount}; 位置: 金额列", "amount")
    add(rows, f"{context}; 单元格: {dt}; 位置: 时间列", "date_time")
    add(rows, f"{context}; 单元格: {method}; 位置: 支付方式", "payment_method")
    add(rows, f"{context}; 单元格: {tx_type}; 位置: 分类/交易类型", "transaction_type")
    for value in ["全部", "支出", "收入", "筛选", "账单", "本月"]:
        add(rows, f"{context}; 单元格: {value}; 位置: tab/筛选项", "other")


def add_bank_bill(rows):
    bank = random.choice(BANKS)
    counterparty = random.choice(COUNTERPARTIES)
    amount = money(1, 20000)
    balance = money(100, 100000)
    dt = date_text()
    tx_type = random.choice(["快捷支付", "转账汇款", "工资收入", "基金赎回", "信用卡还款", "消费"])
    card = card_no()
    sign = random.choice(["收入", "支出"])
    amount_label = "income_amount" if sign == "收入" else "expense_amount"
    display_amount = ("+" if sign == "收入" else "-") + amount
    context = f"页面: 银行账单; 银行: {bank}; 卡号: {card}; 行: {dt} {tx_type} {counterparty} {display_amount} 余额 {balance}"
    add(rows, f"{context}; 单元格: {bank}; 字段提示: 银行名称/机构", "counterparty")
    add(rows, f"{context}; 单元格: {card}; 字段提示: 银行卡尾号", "bank_card")
    add(rows, f"{context}; 单元格: {counterparty}; 字段提示: 对方户名/交易对手", "counterparty")
    add(rows, f"{context}; 单元格: {display_amount}; 字段提示: {sign}金额", amount_label)
    add(rows, f"{context}; 单元格: {balance}; 字段提示: 账户余额", "balance")
    add(rows, f"{context}; 单元格: {dt}; 字段提示: 交易时间", "date_time")
    add(rows, f"{context}; 单元格: {tx_type}; 字段提示: 交易类型/摘要", "transaction_type")
    for value in ["可用余额", "交易明细", "电子回单", "复制", "筛选"]:
        add(rows, f"{context}; 单元格: {value}; 字段提示: 页面控件/表头", "other")


def add_investment_holding(rows):
    name, code = random.choice(ASSETS)
    market_value = money(100, 80000)
    profit_value = random.uniform(-3000, 3000)
    profit_text = f"{profit_value:.2f}"
    rate = percent()
    holding = str(random.choice([100, 200, 500, 1000, 1500, 2200, 5000]))
    available = holding if random.random() > 0.25 else str(max(0, int(holding) - random.choice([100, 200, 500])))
    price = f"{random.uniform(0.5, 2800):.3f}"
    net_value = f"{random.uniform(0.6, 8.0):.4f}"
    shares = f"{random.uniform(10, 10000):.2f}"
    context = f"页面: 理财持仓; 表头: 名称 代码 市值 盈亏 盈亏率 持仓 可用 现价 净值 份额; 行: {name} {code} {market_value} {profit_text} {rate} {holding} {available} {price} {net_value} {shares}"
    add(rows, f"{context}; 单元格: {name}; 位置: 名称列", "asset_name")
    add(rows, f"{context}; 单元格: {code}; 位置: 代码列", "asset_code")
    add(rows, f"{context}; 单元格: {market_value}; 位置: 市值/持有金额列", "market_value")
    add(rows, f"{context}; 单元格: {profit_text}; 位置: 盈亏列", "profit")
    add(rows, f"{context}; 单元格: {rate}; 位置: 盈亏率/收益率列", "profit_rate")
    add(rows, f"{context}; 单元格: {holding}; 位置: 持仓列", "holding")
    add(rows, f"{context}; 单元格: {available}; 位置: 可用列", "available")
    add(rows, f"{context}; 单元格: {holding}; 位置: 数量/持仓数量", "quantity")
    add(rows, f"{context}; 单元格: {price}; 位置: 现价/成本价", "price")
    add(rows, f"{context}; 单元格: {net_value}; 位置: 单位净值", "net_value")
    add(rows, f"{context}; 单元格: {shares}; 位置: 持有份额", "shares")
    for value in ["买入", "卖出", "撤单", "持仓", "查询", "我的资产", "净资产总览"]:
        add(rows, f"{context}; 单元格: {value}; 位置: 按钮/tab/标题", "other")


def add_investment_trade_list(rows):
    name, code = random.choice(ASSETS)
    amount = money(100, 30000)
    qty = str(random.choice([100, 200, 500, 1000, 2000]))
    price = f"{random.uniform(0.5, 2800):.3f}"
    dt = date_text()
    tx_type = random.choice(["买入", "卖出", "申购", "赎回", "定投", "分红"])
    status = random.choice(["已成交", "已确认", "处理中", "已撤单"])
    oid = order_id()
    context = f"页面: 理财交易列表; 表头: 名称 代码 类型 金额 数量 价格 时间 状态 委托编号; 行: {name} {code} {tx_type} {amount} {qty} {price} {dt} {status} {oid}"
    add(rows, f"{context}; 单元格: {name}; 位置: 名称列", "asset_name")
    add(rows, f"{context}; 单元格: {code}; 位置: 代码列", "asset_code")
    add(rows, f"{context}; 单元格: {tx_type}; 位置: 交易类型", "transaction_type")
    add(rows, f"{context}; 单元格: {amount}; 位置: 交易金额", "amount")
    add(rows, f"{context}; 单元格: {qty}; 位置: 数量列", "quantity")
    add(rows, f"{context}; 单元格: {price}; 位置: 成交价格/委托价格", "price")
    add(rows, f"{context}; 单元格: {dt}; 位置: 时间列", "date_time")
    add(rows, f"{context}; 单元格: {status}; 位置: 状态列", "status")
    add(rows, f"{context}; 单元格: {oid}; 位置: 委托编号/订单号", "order_id")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate weak seed data for finance OCR field classification.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-scene", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    rows = []
    scene_generators = [
        add_payment_detail,
        add_payment_list,
        add_bank_bill,
        add_investment_holding,
        add_investment_trade_list,
    ]
    for generator in scene_generators:
        for _ in range(args.samples_per_scene):
            generator(rows)
    augment_compact_rows(rows)

    random.shuffle(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    print(f"Wrote {len(rows)} rows: {output}")
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
