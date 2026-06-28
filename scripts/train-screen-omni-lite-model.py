import gzip
import hashlib
import json
import math
import random
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "model" / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"
OUT_ROOT = ROOT / "model" / "screen_omni_lite"
STATIC_ROOT = ROOT / "static" / "models" / "screen_omni_lite"

LABELS = {
    "chat_todo": {
        "kind": "chat",
        "action": "create_todos",
        "title": "聊天待办",
        "description": "从聊天记录提取待办并创建任务",
    },
    "bill_record": {
        "kind": "bill",
        "action": "create_transaction",
        "title": "账单记账",
        "description": "从账单、付款、收款页面提取记账字段并创建交易",
    },
    "investment_import": {
        "kind": "investment",
        "action": "import_investment",
        "title": "基金持仓导入",
        "description": "从基金、证券、理财持仓列表导入或更新资产",
    },
    "phone_action": {
        "kind": "phone_action",
        "action": "suggest_phone_action",
        "title": "手机操作建议",
        "description": "根据手机截图和任务推断下一步操作",
    },
    "screen_summary": {
        "kind": "general",
        "action": "summarize_screen",
        "title": "屏幕摘要",
        "description": "总结普通屏幕内容",
    },
}

SLOT_PATTERNS = {
    "bill_record": {
        "amount": r"[+\-]?\s*[¥￥]?\s*[0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?\s*(?:元)?",
        "income_keywords": ["收入", "收款", "入账", "退款", "到账"],
        "expense_keywords": ["支出", "付款", "支付", "消费", "实付", "扣款"],
        "source_keywords": ["微信", "支付宝", "云闪付", "银行卡", "招商银行", "工商银行"],
    },
    "investment_import": {
        "code": r"\b(?:[0-9]{6}|[A-Z]{1,5})\b",
        "keywords": ["基金", "持仓", "市值", "收益率", "盈亏", "净值", "份额", "证券", "ETF"],
    },
    "chat_todo": {
        "time": r"(今天|明天|后天|今晚|下周[一二三四五六日天]?|周[一二三四五六日天]|\d{1,2}[:：]\d{2}|\d{1,2}[月/\-]\d{1,2})",
        "keywords": ["记得", "提醒", "待办", "安排", "开会", "提交", "确认", "处理", "跟进", "报销", "付款", "还款"],
    },
}


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


def tokenize(text: str):
    text = normalize_text(text)
    tokens = re.findall(r"[a-z_]+|[0-9]+(?:\.[0-9]+)?|[\u4e00-\u9fff]", text)
    bigrams = []
    chars = [t for t in tokens if len(t) == 1 and "\u4e00" <= t <= "\u9fff"]
    for i in range(len(chars) - 1):
        bigrams.append(chars[i] + chars[i + 1])
    return tokens + bigrams


def contains_any(text: str, keywords):
    return any(k.lower() in text for k in keywords)


def rule_predict(text: str):
    value = normalize_text(text)
    investment_hits = 0
    for keyword in SLOT_PATTERNS["investment_import"]["keywords"]:
        if keyword.lower() in value:
            investment_hits += 1
    if investment_hits >= 2:
        return "investment_import"
    has_amount = re.search(SLOT_PATTERNS["bill_record"]["amount"], value) is not None
    if has_amount and (
        contains_any(value, SLOT_PATTERNS["bill_record"]["income_keywords"])
        or contains_any(value, SLOT_PATTERNS["bill_record"]["expense_keywords"])
        or contains_any(value, SLOT_PATTERNS["bill_record"]["source_keywords"])
    ):
        return "bill_record"
    todo_hits = 0
    for keyword in SLOT_PATTERNS["chat_todo"]["keywords"]:
        if keyword.lower() in value:
            todo_hits += 1
    has_time = re.search(SLOT_PATTERNS["chat_todo"]["time"], value) is not None
    has_speaker = re.search(r"(^|\n).{1,12}[:：]\s*.{2,}", text or "") is not None
    has_chat_hint = contains_any(value, ["群聊", "聊天", "微信", "收到", "回复"])
    if todo_hits >= 1 and (has_time or has_speaker or has_chat_hint):
        return "chat_todo"
    return ""


def add_sample(rows, label: str, text: str, source: str):
    rows.append({"label": label, "text": text, "source": source})


def build_business_samples():
    rows = []
    names = ["张三", "李四", "王敏", "产品群", "财务", "客户A", "同事"]
    todo_verbs = ["提交报销材料", "确认基金持仓截图", "处理付款申请", "跟进客户回款", "整理会议纪要", "缴费", "还款", "转账给供应商"]
    todo_times = ["今天 18:00", "明天 9:30", "后天上午", "今晚", "周五下午", "下周一 10:00", "6月28日"]
    for name in names:
        for verb in todo_verbs:
            for due in todo_times:
                add_sample(rows, "chat_todo", f"群聊 项目组\n{name}: {due} 记得{verb}\n收到请回复", "synthetic_chat")
                add_sample(rows, "chat_todo", f"{name}: 麻烦{due}前{verb}\n我: 好的", "synthetic_chat")

    apps = ["微信支付", "支付宝", "招商银行", "云闪付", "美团", "京东"]
    merchants = ["便利店", "星巴克", "滴滴出行", "盒马", "房租", "工资", "退款", "基金申购"]
    amounts = ["12.80", "35.00", "128.45", "500.00", "1200.00", "6888.88"]
    modes = ["支付成功", "付款", "交易详情", "账单", "已收款", "退款到账"]
    for app in apps:
        for merchant in merchants:
            for amount in amounts:
                for mode in modes:
                    text = f"{app}\n{mode}\n金额 ¥{amount}\n商户 {merchant}\n付款方式 零钱/银行卡\n交易时间 2026-06-21 12:30"
                    add_sample(rows, "bill_record", text, "synthetic_bill")

    funds = ["沪深300ETF", "中证红利基金", "易方达蓝筹精选", "华夏能源革新", "纳斯达克100ETF", "招商中证白酒"]
    codes = ["510300", "090010", "005827", "003834", "513100", "161725"]
    providers = ["支付宝基金", "天天基金", "招商证券", "华泰证券", "蛋卷基金"]
    for provider in providers:
        for fund, code in zip(funds, codes):
            add_sample(
                rows,
                "investment_import",
                f"{provider}\n持仓\n{fund} {code}\n持有市值 12345.67\n持有收益 +234.56\n收益率 +3.21%\n持有份额 1000.00\n最新净值 1.2345",
                "synthetic_investment",
            )
            add_sample(
                rows,
                "investment_import",
                f"基金列表\n名称 {fund}\n代码 {code}\n市值 8888.00\n盈亏 -56.78\n净值 0.9876",
                "synthetic_investment",
            )

    general = [
        "设置\n账号与安全\n隐私\n通用",
        "首页\n搜索\n推荐内容\n消息",
        "天气\n今天多云\n空气质量良好",
        "视频播放\n第12集\n暂停\n倍速",
        "动漫详情\n播放列表\n收藏\n评论",
    ]
    for item in general:
        for _ in range(80):
            add_sample(rows, "screen_summary", item, "synthetic_general")
    return rows


def build_public_samples(max_summary=1800, max_phone=1400):
    rows = []
    for item in load_jsonl(PROCESSED_ROOT / "screen_summary_text_only.jsonl")[:max_summary]:
        text = f"{item.get('instruction', '')}\n{item.get('output', '')}"
        add_sample(rows, "screen_summary", text, item.get("source", "screen2words"))
    for item in load_jsonl(PROCESSED_ROOT / "screen_action_sft.jsonl")[:max_phone]:
        text = f"{item.get('instruction', '')}\n{item.get('output', '')}"
        add_sample(rows, "phone_action", text, item.get("source", "guiact"))
    for item in load_jsonl(PROCESSED_ROOT / "mobile_actions_tool_sft.jsonl")[:max_phone]:
        text = f"{item.get('instruction', '')}\n{item.get('output', '')}"
        add_sample(rows, "phone_action", text, item.get("source", "mobile_actions"))
    return rows


def train_nb(rows):
    label_counts = Counter()
    token_counts = defaultdict(Counter)
    vocab = set()
    for row in rows:
        label = row["label"]
        tokens = tokenize(row["text"])
        label_counts[label] += 1
        token_counts[label].update(tokens)
        vocab.update(tokens)
    return label_counts, token_counts, vocab


def predict(label_counts, token_counts, vocab, text):
    ruled = rule_predict(text)
    if ruled:
        return ruled
    tokens = tokenize(text)
    total = sum(label_counts.values())
    labels = list(label_counts.keys())
    vocab_size = max(1, len(vocab))
    best_label = ""
    best_score = -1e100
    for label in labels:
        counts = token_counts[label]
        denom = sum(counts.values()) + vocab_size
        score = math.log((label_counts[label] + 1.0) / (total + len(labels)))
        for token in tokens:
            score += math.log((counts.get(token, 0) + 1.0) / denom)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def evaluate(rows):
    random.seed(20260621)
    random.shuffle(rows)
    split = max(1, int(len(rows) * 0.9))
    train_rows = rows[:split]
    eval_rows = rows[split:]
    label_counts, token_counts, vocab = train_nb(train_rows)
    correct = 0
    confusion = defaultdict(Counter)
    for row in eval_rows:
        pred = predict(label_counts, token_counts, vocab, row["text"])
        if pred == row["label"]:
            correct += 1
        confusion[row["label"]][pred] += 1
    return train_rows, eval_rows, label_counts, token_counts, vocab, correct / max(1, len(eval_rows)), confusion


def compact_counts(token_counts, max_tokens_per_label=1800):
    compact = {}
    for label, counts in token_counts.items():
        compact[label] = dict(counts.most_common(max_tokens_per_label))
    return compact


def write_outputs(model, report):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    STATIC_ROOT.mkdir(parents=True, exist_ok=True)
    pretty = OUT_ROOT / "screen_omni_model.json"
    mini = OUT_ROOT / "screen_omni_model.min.json"
    gz = OUT_ROOT / "screen_omni_model.min.json.gz"
    zip_path = OUT_ROOT / "screen_omni_lite_model.zip"
    report_path = OUT_ROOT / "train_report.json"

    pretty.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    mini_text = json.dumps(model, ensure_ascii=False, separators=(",", ":"))
    mini.write_text(mini_text, encoding="utf-8")
    with gzip.open(gz, "wb") as f:
        f.write(mini_text.encode("utf-8"))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = OUT_ROOT / "README.md"
    readme.write_text(
        "# ZeMo Screen Omni Lite\n\n"
        "100MB 内的屏幕理解轻量模型包，输入为截图 OCR/布局文本，输出业务意图：聊天待办、账单记账、基金持仓导入、普通摘要、手机操作建议。\n\n"
        "决策层为业务优先规则 + 朴素贝叶斯轻量分类器。业务优先级：基金/持仓 > 账单 > 聊天待办 > NB 泛化分类。\n\n"
        "离线测试：`python scripts/predict-screen-omni-lite-model.py --text \"微信支付 金额 35.00 支付成功\"`。\n\n"
        "当前 App 执行层已用 OCR + 规则/字段模型完成闭环，本包用于后续替换或增强 `ScreenUnderstandingService` 的意图判定。\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(mini, "screen_omni_model.min.json")
        z.write(report_path, "train_report.json")
        z.write(readme, "README.md")

    for src in [mini, gz, report_path]:
        (STATIC_ROOT / src.name).write_bytes(src.read_bytes())
    return pretty, mini, gz, zip_path, report_path


def main():
    rows = build_business_samples() + build_public_samples()
    if not rows:
        raise SystemExit("没有训练数据，请先运行 scripts/prepare-screen-omni-data.py")
    train_rows, eval_rows, label_counts, token_counts, vocab, accuracy, confusion = evaluate(rows)
    model = {
        "format": "zemo_screen_omni_lite_nb_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decision": "business_priority_rules_then_naive_bayes",
        "labels": LABELS,
        "slot_patterns": SLOT_PATTERNS,
        "tokenizer": "lowercase + OCR-meta-strip + latin/digit/chinese-char + chinese-bigram",
        "vocab_size": len(vocab),
        "total_train_rows": len(train_rows),
        "label_counts": dict(label_counts),
        "token_counts": compact_counts(token_counts),
    }
    mini_text = json.dumps(model, ensure_ascii=False, separators=(",", ":"))
    model["sha256"] = hashlib.sha256(mini_text.encode("utf-8")).hexdigest()
    report = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "eval_accuracy": accuracy,
        "label_counts": dict(label_counts),
        "confusion": {label: dict(counts) for label, counts in confusion.items()},
        "size_limit_bytes": 100 * 1024 * 1024,
        "note": "这是 ZeMo 屏幕业务闭环的轻量意图模型，不是完整端到端视觉语言大模型；视觉输入由 OCR/布局模型提供。",
    }
    paths = write_outputs(model, report)
    sizes = {p.name: p.stat().st_size for p in paths}
    print(json.dumps({"out": str(OUT_ROOT), "static": str(STATIC_ROOT), "accuracy": accuracy, "sizes": sizes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
