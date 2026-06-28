# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OCR_DIR = Path(
    r"E:\CamXAll\ZEMO\Data\Model\Omni\zemo-o\dataset\img\_ocr_out_cpp_v2"
)
DEFAULT_MINIMIND_O_ROOT = Path(r"E:\CamXAll\ZEMO\uniappx\ZeMo-finance\model\minimind-o")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SCENES = ["娱乐", "记账", "理财", "待办", "记事", "购物", "出行", "政务", "健康", "通信", "工具", "其他"]
INTENTS = [
    "bill_record",
    "investment_import",
    "chat_todo",
    "entertainment_activity",
    "shopping_activity",
    "note_extract",
    "screen_summary",
]
ACTIONS = [
    "create_transaction",
    "import_investment",
    "create_todos",
    "record_activity",
    "create_note",
    "summarize_screen",
]

APP_NAMES = {
    "com.tencent.mm": "微信",
    "com.tencent.wechat": "微信",
    "com.ohos.mms": "短信",
    "com.alipay": "支付宝",
    "com.eg.android.alipaygphone": "支付宝",
    "cmb.pb": "招商银行",
    "cmbchina": "招商银行",
    "com.csg.palmhall": "南方电网",
    "mc10086": "中国移动",
    "com.hexin": "同花顺",
    "com.tdx": "通达信",
    "com.app.yangjibao": "养基宝",
    "bili": "哔哩哔哩",
    "danmaku": "哔哩哔哩",
    "videohm": "腾讯视频",
    "qiyi": "爱奇艺",
    "ugc.aweme": "抖音",
    "qqmusic": "QQ音乐",
    "luna.hm.music": "汽水音乐",
    "taobao": "淘宝",
    "idlefish": "闲鱼",
    "jingdong": "京东",
    "jd.hm.mall": "京东",
    "xunmeng": "拼多多",
    "pinduoduo": "拼多多",
    "alibaba.wireless": "1688",
    "xiaomi.shop": "小米商城",
    "microsoft.emmx": "Edge",
    "obsidian": "Obsidian",
    "zemo": "ZeMo",
}

FIELD_SCHEMA = {
    "bill_record": {
        "bill": {
            "amount": "string",
            "mode": "expense|income|transfer|refund",
            "merchant": "string",
            "pay_method": "string",
            "transaction_time": "string",
            "order_no": "string",
            "category": "string",
        }
    },
    "investment_import": {
        "investment": {
            "account": "string",
            "total_assets": "string",
            "daily_profit": "string",
            "total_profit": "string",
            "holdings": "array",
        }
    },
    "chat_todo": {
        "todos": [
            {
                "title": "string",
                "due_at": "string",
                "priority": "low|normal|high",
                "source_text": "string",
            }
        ]
    },
    "entertainment_activity": {
        "entertainment": {
            "media_type": "music|anime|video|short_video|live|other",
            "title": "string",
            "platform": "string",
            "activity": "string",
        }
    },
    "shopping_activity": {
        "shopping": {
            "platform": "string",
            "status": "string",
            "product": "string",
            "amount": "string",
            "order_no": "string",
        }
    },
    "note_extract": {
        "note": {
            "summary": "string",
            "key_points": "array",
            "key_values": "object",
        }
    },
}

NOISE_LINE_PATTERNS = [
    re.compile(r"^[0-9:：.\-\s]+$"),
    re.compile(r"^[<>×+·.。|/\\\s]+$"),
    re.compile(r"^[kK]?[0-9A-Za-z./\s]{1,12}$"),
]


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(compact_json(row) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            value = line.strip()
            if not value:
                continue
            try:
                rows.append(json.loads(value))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad jsonl at {path}:{line_no}: {exc}") from exc
    return rows


def path_has_unpackage(value: str) -> bool:
    if not value:
        return False
    return any(part.lower() == "unpackage" for part in Path(value).parts)


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def clean_text(value: str) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_line(value: str) -> str:
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def is_noise_line(value: str) -> bool:
    text = normalize_line(value)
    if not text:
        return True
    if re.match(r"^[+\-＋－]?\s*[¥￥]?\d+(?:,\d{3})*(?:\.\d{1,4})?%?$", text):
        return False
    if re.search(r"20\d{2}[-./年]\d{1,2}[-./月]\d{1,2}", text):
        return False
    if len(text) <= 1 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return True
    if text in {"复制", "查看", "更多", "展开", "筛选", "搜索", "返回", "完成", "我的", "首页"}:
        return True
    return any(pattern.match(text) for pattern in NOISE_LINE_PATTERNS)


def sorted_ocr_lines(row: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in row.get("lines", []) or []:
        text = normalize_line(str(item.get("t", "")))
        box = item.get("b", [0, 0, 0, 0]) or [0, 0, 0, 0]
        if len(box) < 4:
            box = [0, 0, 0, 0]
        try:
            conf = float(item.get("c", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        result.append(
            {
                "text": text,
                "conf": round(conf, 6),
                "bbox": [int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3]))],
            }
        )
    result.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return result


def plain_lines(lines: list[dict[str, Any]], keep_noise: bool = False) -> list[str]:
    values = [line["text"] for line in lines if line.get("text")]
    if keep_noise:
        return values
    return [value for value in values if not is_noise_line(value)]


def ocr_meta_text(lines: list[dict[str, Any]], limit_lines: int = 80) -> str:
    parts = []
    for item in lines[:limit_lines]:
        box = item["bbox"]
        parts.append(f"[[ocr:{box[0]},{box[1]},{box[2]},{box[3]}]]{item['text']}")
    return "\n".join(parts)


def strip_duplicate_suffix(value: str) -> str:
    return re.sub(r"\(\d+\)$", "", value)


def extract_package(image_path: str) -> str:
    stem = strip_duplicate_suffix(Path(image_path).stem)
    patterns = [
        r"(?i)^screenshot_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:-\d+)?_(?P<pkg>.+)$",
        r"(?i)^(?:screenshot|screen)_(?:\d{8}_\d{6})_(?P<pkg>.+)$",
        r"_(?P<pkg>[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9][A-Za-z0-9_-]*)+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return strip_duplicate_suffix(match.group("pkg")).lower()
    return ""


def extract_screen_time(image_path: str) -> str:
    stem = Path(image_path).stem
    patterns = [
        r"(?i)screenshot_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})",
        r"(?i)screenshot_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})",
        r"(?i)(?:img|mvimg)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})",
        r"ChatGPT Image (\d{4})年(\d{1,2})月(\d{1,2})日 (\d{1,2})_(\d{1,2})_(\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if not match:
            continue
        try:
            y, mo, d, h, mi, s = [int(v) for v in match.groups()]
            return datetime(y, mo, d, h, mi, s).isoformat()
        except ValueError:
            return ""
    return ""


def app_name(package: str) -> str:
    key = package.lower()
    for needle, name in APP_NAMES.items():
        if needle in key:
            return name
    return package


def contains_any(text: str, words: list[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)


def score_keywords(text: str, words: list[str]) -> int:
    lower = text.lower()
    return sum(1 for word in words if word.lower() in lower)


def text_blob(lines: list[str], package: str) -> str:
    return f"{package}\n" + "\n".join(lines)


def classify_scene(lines: list[str], package: str) -> dict[str, Any]:
    blob = text_blob(lines, package)
    pkg = package.lower()
    lower = blob.lower()

    investment_score = score_keywords(blob, ["基金", "持仓", "证券", "股票", "总资产", "市值", "盈亏", "收益率", "ETF", "净值", "份额"])
    if contains_any(pkg, ["hexin", "tdx", "yangjibao", "shenlan", "obc.cbn"]) or investment_score >= 3:
        return scene("理财", "investment_import", "investment", "fund_or_stock_holding", "import_investment", "理财/持仓识别", 0.88, ["理财包名或持仓关键词"])

    bill_score = score_keywords(blob, ["账单详情", "交易成功", "实付款", "支付金额", "订单金额", "付款方式", "支付方式", "支付时间", "扣款", "退款", "收支", "结息"])
    bill_pkg = contains_any(pkg, ["alipay", "cmb", "palmhall", "mc10086", "delivery.aggregator", "jingdong", "taobao", "xunmeng", "wechat", "tencent.mm"])
    if bill_score >= 2 or (bill_pkg and bill_score >= 1):
        return scene("记账", "bill_record", "bill", "bill_or_payment", "create_transaction", "账单/交易识别", 0.86, ["账单/支付关键词"])

    logistics_score = score_keywords(blob, ["取件码", "快递待取", "待取件", "运单号", "派送中", "寄件码"])
    todo_score = score_keywords(blob, ["记得", "提醒", "待办", "安排", "提交", "确认", "处理", "跟进", "缴费", "还款", "明天", "后天", "今晚", "截止"])
    chat_pkg = contains_any(pkg, ["wechat", "tencent.mm", "ohos.mms"])
    if logistics_score >= 1 or (chat_pkg and todo_score >= 1):
        sub_scene = "package_pickup" if logistics_score >= 1 else "chat_todo"
        return scene("待办", "chat_todo", "chat", sub_scene, "create_todos", "待办识别", 0.82, ["物流或聊天行动项"])

    entertainment_pkg = contains_any(pkg, ["qqmusic", "luna.hm.music", "bili", "danmaku", "videohm", "qiyi", "ugc.aweme", "sjz.ss"])
    entertainment_score = score_keywords(blob, ["正在播放", "歌词", "歌曲", "歌单", "弹幕", "动漫", "电视剧", "电影", "综艺", "直播", "短视频", "VIP", "倍速"])
    if entertainment_pkg or entertainment_score >= 3:
        return scene("娱乐", "entertainment_activity", "activity", infer_media_type(lower), "record_activity", "娱乐活动", 0.84, ["娱乐包名或播放关键词"])

    shopping_pkg = contains_any(pkg, ["taobao", "jingdong", "jd.hm.mall", "xunmeng", "pinduoduo", "idlefish", "xiaomi.shop", "alibaba.wireless", "qianniu"])
    shopping_score = score_keywords(blob, ["商品", "订单", "购物车", "待收货", "待付款", "店铺", "客服", "退款详情", "售后", "再次购买", "加入购物车"])
    if shopping_pkg or shopping_score >= 3:
        sub_scene = "shopping_order" if contains_any(blob, ["订单", "待收货", "退款", "售后"]) else "shopping_browse"
        return scene("购物", "shopping_activity", "activity", sub_scene, "record_activity", "购物活动", 0.78, ["购物包名或订单关键词"])

    if contains_any(blob, ["车票", "出票成功", "取票号", "车厢", "改签成功"]) or re.search(r"\bG\d{2,5}\b", blob, re.I):
        return scene("出行", "note_extract", "note", "travel_ticket", "create_note", "出行票据", 0.80, ["出行票务关键词"])
    if contains_any(blob, ["社保", "参保证明", "社保卡", "就业补贴", "政务"]):
        return scene("政务", "note_extract", "note", "government_service", "create_note", "政务/办事记录", 0.78, ["政务关键词"])
    if contains_any(blob, ["睡眠", "健康记录", "睡眠质量"]):
        return scene("健康", "note_extract", "note", "health_record", "create_note", "健康记录", 0.72, ["健康关键词"])
    if contains_any(blob, ["居民身份证", "公民身份号码", "student number", "graduation date", "school:", "毕业", "有效期限", "住址"]):
        return scene("记事", "note_extract", "note", "document_or_id", "create_note", "证件/资料摘录", 0.84, ["证件或资料文本"])
    if contains_any(blob, ["豆包", "内容由 ai 生成", "obsidian", "备忘", "笔记", "制图"]):
        return scene("记事", "note_extract", "note", "note_or_ai_chat", "create_note", "记事摘要", 0.72, ["笔记/AI/文档信息"])
    if not lines:
        return scene("其他", "screen_summary", "general", "empty_or_visual_only", "summarize_screen", "无文字截图", 0.45, ["无有效 OCR 文本"])
    return scene("记事", "note_extract", "note", "general_note", "create_note", "屏幕文字摘要", 0.62, ["默认文字摘要"])


def scene(scene_name: str, intent: str, kind: str, sub_scene: str, action: str, title: str, confidence: float, evidence: list[str]) -> dict[str, Any]:
    return {
        "scene": scene_name,
        "intent": intent,
        "kind": kind,
        "sub_scene": sub_scene,
        "action": action,
        "title": title,
        "confidence": confidence,
        "evidence": evidence,
    }


def infer_media_type(text: str) -> str:
    if contains_any(text, ["qqmusic", "luna.hm.music", "歌词", "歌曲", "歌单", "听歌"]):
        return "music"
    if contains_any(text, ["ugc.aweme", "抖音", "短视频"]):
        return "short_video"
    if contains_any(text, ["bili", "danmaku", "动漫", "番剧"]):
        return "anime"
    if contains_any(text, ["videohm", "qiyi", "电视剧", "电影", "综艺"]):
        return "video"
    if "直播" in text:
        return "live"
    return "other"


def normalize_amount(raw: str) -> str:
    return raw.replace("￥", "").replace("¥", "").replace("元", "").replace(",", "").replace(" ", "").replace("＋", "+").replace("－", "-").strip()


def amount_candidates(lines: list[str]) -> list[dict[str, Any]]:
    amount_re = re.compile(r"(?<!\d)([+\-＋－]?)\s*[¥￥]?\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)\s*(?:元)?")
    label_words = ["实付款", "支付金额", "订单金额", "金额", "实际应付", "账单", "合计", "支出", "收入", "扣款", "总资产", "市值", "已退款", "退款"]
    result = []
    for idx, line in enumerate(lines):
        if re.search(r"\d{8,}", line) and not re.search(r"[¥￥元+\-＋－]", line):
            continue
        near = "\n".join(lines[max(0, idx - 2) : min(len(lines), idx + 3)])
        for match in amount_re.finditer(line):
            raw = match.group(0)
            amount = normalize_amount(raw)
            if not amount:
                continue
            score = 0
            if re.search(r"[¥￥元+\-＋－]", raw):
                score += 3
            if contains_any(near, label_words):
                score += 4
            if "." in amount:
                score += 1
            try:
                if float(amount.replace("+", "").replace("-", "")) >= 0.01 and score > 0:
                    result.append({"amount": amount, "line": line, "index": idx, "score": score})
            except ValueError:
                pass
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


def next_value_after(lines: list[str], keys: list[str], max_lookahead: int = 2) -> str:
    for idx, line in enumerate(lines):
        if not contains_any(line, keys):
            continue
        tail = re.sub("|".join(re.escape(k) for k in keys), "", line).strip(" :：>")
        if tail:
            return tail
        for j in range(idx + 1, min(len(lines), idx + 1 + max_lookahead)):
            candidate = lines[j].strip(" >：:")
            if candidate and not is_noise_line(candidate):
                return candidate
    return ""


def extract_order_no(lines: list[str]) -> str:
    for idx, line in enumerate(lines):
        if contains_any(line, ["订单编号", "交易单号", "商户单号", "运单号", "快递单号"]):
            joined = " ".join(lines[idx : min(len(lines), idx + 2)])
            match = re.search(r"\b(?:SF|JD|YT|YD|ZTO|STO|EMS)?[A-Z0-9]{8,}\b", joined, re.I)
            if match:
                return match.group(0)
    return ""


def extract_first_datetime(lines: list[str], fallback_screen_time: str) -> str:
    text = "\n".join(lines)
    match = re.search(r"(20\d{2})[-./年](\d{1,2})[-./月](\d{1,2})[日\s-]*(\d{1,2})[:：](\d{2})(?::(\d{2}))?", text)
    if not match:
        return fallback_screen_time
    groups = match.groups(default="0")
    try:
        y, mo, d, h, mi, s = [int(v) for v in groups]
        return datetime(y, mo, d, h, mi, s).isoformat()
    except ValueError:
        return fallback_screen_time


def extract_entities(intent: str, lines: list[str], package: str, screen_time: str, sub_scene: str) -> dict[str, Any]:
    entities: dict[str, Any] = {
        "bill": {},
        "investment": {},
        "todos": [],
        "entertainment": {},
        "shopping": {},
        "note": {},
    }
    if intent == "bill_record":
        amounts = amount_candidates(lines)
        amount = amounts[0]["amount"] if amounts else ""
        mode = "refund" if contains_any("\n".join(lines), ["退款", "退回", "已退款"]) else "expense"
        if amount.startswith("+"):
            mode = "income"
        entities["bill"] = {
            "amount": amount,
            "mode": mode,
            "merchant": guess_title(lines, app_name(package) or "交易"),
            "pay_method": next_value_after(lines, ["付款方式", "支付方式"]),
            "transaction_time": extract_first_datetime(lines, screen_time),
            "order_no": extract_order_no(lines),
            "category": next_value_after(lines, ["账单分类", "消费项目", "账单类型"]),
        }
    elif intent == "investment_import":
        entities["investment"] = {
            "account": app_name(package),
            "total_assets": next_amount_near(lines, ["总资产", "账户资产", "人民币账户"]),
            "daily_profit": next_amount_near(lines, ["当日收益", "昨日收益", "当日盈亏"]),
            "total_profit": next_amount_near(lines, ["总盈亏", "持有收益", "盈亏"]),
            "holdings": extract_holdings(lines),
        }
    elif intent == "chat_todo":
        entities["todos"] = extract_todos(lines, sub_scene)
    elif intent == "entertainment_activity":
        entities["entertainment"] = {
            "media_type": sub_scene if sub_scene in {"music", "anime", "video", "short_video", "live", "other"} else "other",
            "title": guess_title(lines, ""),
            "platform": app_name(package),
            "activity": "watch_or_listen",
        }
    elif intent == "shopping_activity":
        entities["shopping"] = {
            "platform": app_name(package),
            "status": next_value_after(lines, ["订单状态", "交易状态", "物流状态"]),
            "product": guess_title(lines, ""),
            "amount": (amount_candidates(lines)[0]["amount"] if amount_candidates(lines) else ""),
            "order_no": extract_order_no(lines),
        }
    else:
        entities["note"] = extract_note(lines)
    return entities


def guess_title(lines: list[str], fallback: str) -> str:
    skip = ["账单详情", "全部账单", "交易成功", "订单金额", "支付时间", "付款方式", "更多", "金额"]
    for line in lines[:16]:
        if not is_noise_line(line) and not contains_any(line, skip) and not re.match(r"^[+\-]?[¥￥]?\d", line):
            return line[:80]
    return fallback


def next_amount_near(lines: list[str], keys: list[str]) -> str:
    for idx, line in enumerate(lines):
        if not contains_any(line, keys):
            continue
        values = amount_candidates(lines[idx : min(len(lines), idx + 5)])
        if values:
            return values[0]["amount"]
    return ""


def extract_holdings(lines: list[str], limit: int = 8) -> list[dict[str, str]]:
    holdings = []
    seen = set()
    for idx, line in enumerate(lines):
        code_match = re.search(r"\b\d{6}\b", line)
        is_name = contains_any(line, ["基金", "ETF", "etf", "纳斯达克", "标普", "股票", "证券", "混合"])
        if not code_match and not is_name:
            continue
        context = lines[max(0, idx - 2) : min(len(lines), idx + 5)]
        joined = " ".join(context)
        code = code_match.group(0) if code_match else ""
        if not code:
            code2 = re.search(r"\b\d{6}\b", joined)
            code = code2.group(0) if code2 else ""
        name = re.sub(r"\b\d{6}\b", "", line).strip(" -：:/")
        key = f"{name}|{code}"
        if key in seen or (not name and not code):
            continue
        seen.add(key)
        amounts = amount_candidates(context)
        percent = re.search(r"[+\-＋－]?\d+(?:\.\d+)?%", joined)
        holdings.append(
            {
                "name": name,
                "code": code,
                "market_value": amounts[0]["amount"] if amounts else "",
                "profit": amounts[1]["amount"] if len(amounts) > 1 else "",
                "profit_rate": percent.group(0).replace("＋", "+").replace("－", "-") if percent else "",
                "source_text": " / ".join(context[:5]),
            }
        )
        if len(holdings) >= limit:
            break
    return holdings


def extract_todos(lines: list[str], sub_scene: str, limit: int = 8) -> list[dict[str, Any]]:
    todos = []
    pickup_codes = []
    for line in lines:
        if contains_any(line, ["取件码", "寄件码"]):
            pickup_codes += re.findall(r"(?:取件码|寄件码)\s*[:：]?\s*([A-Z0-9\-]{3,20})", line, re.I)
    if pickup_codes:
        todos.append({"title": f"取快递 {pickup_codes[0]}", "due_at": "", "priority": "normal", "source_text": " / ".join(pickup_codes)})
    todo_words = ["记得", "提醒", "待办", "安排", "提交", "确认", "处理", "跟进", "缴费", "还款", "报销", "付款"]
    for line in lines:
        if not contains_any(line, todo_words):
            continue
        title = re.sub(r"^[^:：]{1,14}[:：]\s*", "", line).strip(" ，。,.!！?？")
        if title and title not in {item["title"] for item in todos}:
            todos.append({"title": title[:60], "due_at": extract_due_text(line), "priority": "normal", "source_text": line})
        if len(todos) >= limit:
            break
    if not todos and sub_scene == "package_pickup":
        todos.append({"title": "处理快递/物流信息", "due_at": "", "priority": "normal", "source_text": " / ".join(lines[:4])})
    return todos


def extract_due_text(line: str) -> str:
    for word in ["今天", "今晚", "明天", "后天", "本周", "周一", "周二", "周三", "周四", "周五", "周六", "周日"]:
        if word in line:
            return word
    match = re.search(r"\d{1,2}[月./-]\d{1,2}[日号]?", line)
    return match.group(0) if match else ""


def extract_note(lines: list[str]) -> dict[str, Any]:
    key_values = {}
    for line in lines[:40]:
        match = re.match(r"^(.{1,20})[:：]\s*(.{1,80})$", line)
        if match:
            key_values[match.group(1).strip()] = match.group(2).strip()
    return {
        "summary": " / ".join(lines[:4])[:180],
        "key_points": lines[:8],
        "key_values": key_values,
    }


def build_summary(meta: dict[str, Any], entities: dict[str, Any], lines: list[str]) -> str:
    intent = meta["intent"]
    if intent == "bill_record":
        bill = entities["bill"]
        return f"识别到账单/交易，金额 {bill.get('amount', '') or '未识别'}，商户 {bill.get('merchant', '') or '未识别'}。"
    if intent == "investment_import":
        inv = entities["investment"]
        return f"识别到理财/持仓页面，总资产 {inv.get('total_assets', '') or '未识别'}，持仓 {len(inv.get('holdings', []))} 项。"
    if intent == "chat_todo":
        todos = entities["todos"]
        return f"识别到待办：{todos[0].get('title', '')}。" if todos else "识别到可能需要跟进的信息。"
    if intent == "entertainment_activity":
        return f"识别到娱乐活动，平台 {entities['entertainment'].get('platform', '')}。"
    if intent == "shopping_activity":
        return f"识别到购物/订单页面，平台 {entities['shopping'].get('platform', '')}。"
    return f"识别到 {len(lines)} 行 OCR 文本。"


def mask_sensitive_text(value: str) -> str:
    def mask_id(match: re.Match[str]) -> str:
        text = match.group(0)
        return text[:6] + "*" * max(4, len(text) - 10) + text[-4:]

    def mask_phone(match: re.Match[str]) -> str:
        text = match.group(0)
        return text[:3] + "****" + text[-4:]

    def mask_long(match: re.Match[str]) -> str:
        text = match.group(0)
        if re.match(r"20\d{6,}", text):
            return text
        return text[:4] + "*" * max(4, len(text) - 8) + text[-4:]

    value = re.sub(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0\d|1[0-2])(?:[0-2]\d|3[01])\d{3}[0-9Xx]\b", mask_id, value)
    value = re.sub(r"\b1[3-9]\d{9}\b", mask_phone, value)
    value = re.sub(r"\b\d{13,19}\b", mask_long, value)
    return value


def mask_sensitive_obj(value: Any) -> Any:
    if isinstance(value, str):
        return mask_sensitive_text(value)
    if isinstance(value, list):
        return [mask_sensitive_obj(item) for item in value]
    if isinstance(value, dict):
        return {key: mask_sensitive_obj(item) for key, item in value.items()}
    return value


def sensitive_boxes(lines: list[dict[str, Any]]) -> list[list[int]]:
    boxes = []
    for idx, line in enumerate(lines):
        text = line["text"]
        context = "\n".join(item["text"] for item in lines[max(0, idx - 1) : idx + 2])
        if mask_sensitive_text(text) != text or contains_any(context, ["公民身份号码", "手机号", "电话", "身份证"]):
            boxes.append(line["bbox"])
    return boxes


def build_answer(row: dict[str, Any], mask_sensitive: bool) -> dict[str, Any]:
    image = str(row.get("image", ""))
    package = extract_package(image)
    screen_time = extract_screen_time(image)
    line_meta = sorted_ocr_lines(row)
    clean_lines = plain_lines(line_meta)
    extract_lines = plain_lines(line_meta, keep_noise=True)
    meta = classify_scene(clean_lines, package)
    entities = extract_entities(meta["intent"], extract_lines, package, screen_time, meta["sub_scene"])
    avg_conf = round(sum(float(x.get("conf", 0.0)) for x in line_meta) / len(line_meta), 4) if line_meta else 0.0
    answer = {
        "schema_version": "zemo_o_screen_sft_v1",
        "source": "real_ocr_cpp_v2",
        "scene": meta["scene"],
        "intent": meta["intent"],
        "kind": meta["kind"],
        "sub_scene": meta["sub_scene"],
        "action": meta["action"],
        "title": meta["title"],
        "summary": build_summary(meta, entities, clean_lines),
        "source_app_package": package,
        "source_app_name": app_name(package),
        "screen_time": screen_time,
        "needs_image": True,
        "needs_ocr_text": True,
        "confidence": round(float(meta["confidence"]), 3),
        "ocr_quality": {
            "line_count": len(line_meta),
            "text_line_count": len(clean_lines),
            "avg_conf": avg_conf,
            "width": int(row.get("w", 0) or 0),
            "height": int(row.get("h", 0) or 0),
        },
        "entities": entities,
        "field_schema": FIELD_SCHEMA.get(meta["intent"], {}),
        "raw_evidence": meta["evidence"] + clean_lines[:8],
        "requires_review": bool(meta["confidence"] < 0.7 or len(clean_lines) == 0),
    }
    return mask_sensitive_obj(answer) if mask_sensitive else answer


def build_prompt(row: dict[str, Any], variant: int, mask_sensitive: bool) -> str:
    image = str(row.get("image", ""))
    package = extract_package(image)
    screen_time = extract_screen_time(image)
    line_meta = sorted_ocr_lines(row)
    meta_text = ocr_meta_text(line_meta)
    visible_text = "\n".join(plain_lines(line_meta))[:4000]
    if mask_sensitive:
        meta_text = mask_sensitive_text(meta_text)
        visible_text = mask_sensitive_text(visible_text)
    contract = (
        "只输出严格 JSON，不要 Markdown。字段固定：schema_version, source, scene, intent, kind, sub_scene, action, title, "
        "summary, source_app_package, source_app_name, screen_time, needs_image, needs_ocr_text, confidence, ocr_quality, "
        "entities, field_schema, raw_evidence, requires_review。"
    )
    enums = (
        f"scene 只能取 {SCENES}；intent 只能取 {INTENTS}；action 只能取 {ACTIONS}。"
        "没有把握的字段留空，不要编造。"
    )
    if variant % 3 == 0:
        return "\n".join(
            [
                "<image>",
                "你是 ZeMo-O 手机截图理解模型。结合真实截图和 OCR 行，抽取 ZeMo 可执行事件。",
                contract,
                enums,
                f"source_app_package={package or 'unknown'}",
                f"screen_time={screen_time or 'unknown'}",
                "OCR 行（含 bbox）：",
                meta_text,
            ]
        )
    if variant % 3 == 1:
        return "\n".join(
            [
                "<image>",
                "任务：判断当前手机屏幕场景，并转换为 ZeMo 规范 JSON。",
                contract,
                "优先识别：娱乐、记账、理财、待办、记事、购物、出行、政务、健康、其他。",
                f"包名：{package or 'unknown'}；截图时间：{screen_time or 'unknown'}",
                "可见文字：",
                visible_text,
            ]
        )
    compact_lines = [
        {"t": item["text"], "b": item["bbox"], "c": item["conf"]}
        for item in line_meta[:80]
    ]
    if mask_sensitive:
        compact_lines = mask_sensitive_obj(compact_lines)
    return "\n".join(
        [
            "<image>",
            "根据截图和 OCR lines 判断当前场景并抽取结构化字段，输出统一 JSON。",
            enums,
            f"metadata={compact_json({'package': package, 'screen_time': screen_time, 'image': Path(image).name})}",
            f"ocr_lines={compact_json(compact_lines)}",
        ]
    )


def image_target_name(index: int, image_path: Path, row: dict[str, Any]) -> str:
    suffix = image_path.suffix.lower() if image_path.suffix.lower() in IMAGE_EXTS else ".jpg"
    digest = stable_hash(f"{index}:{image_path}:{row.get('w','')}:{row.get('h','')}", 12)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem)[:80].strip("._") or "image"
    return f"{index:05d}_{digest}_{stem}{suffix}"


def copy_image(source: Path, target: Path, redact_boxes: list[list[int]]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not redact_boxes:
        shutil.copy2(source, target)
        return
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("redact-sensitive-images 需要 pillow: python -m pip install pillow") from exc
    with Image.open(source) as img:
        out = img.convert("RGB")
        draw = ImageDraw.Draw(out)
        for left, top, width, height in redact_boxes:
            pad = max(4, int(min(width, height) * 0.08))
            draw.rectangle((left - pad, top - pad, left + width + pad, top + height + pad), fill=(0, 0, 0))
        out.save(target)


def build_dataset_rows(
    ocr_rows: list[dict[str, Any]],
    ocr_jsonl: Path,
    out_dir: Path,
    prompt_variants: int,
    dedupe: bool,
    mask_sensitive: bool,
    copy_images: bool,
    redact_sensitive_images: bool,
    limit: int,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen = set()
    counters = Counter()
    image_dir = out_dir / "images"
    max_items = limit if limit > 0 else len(ocr_rows)
    for source_index, row in enumerate(ocr_rows):
        if counters["accepted_sources"] >= max_items:
            break
        raw_image = str(row.get("image", ""))
        if path_has_unpackage(raw_image):
            counters["skipped_unpackage"] += 1
            continue
        source_image = Path(raw_image)
        if source_image.suffix.lower() not in IMAGE_EXTS or not source_image.exists():
            counters["skipped_missing_image"] += 1
            continue
        line_meta = sorted_ocr_lines(row)
        text_for_hash = "\n".join(plain_lines(line_meta, keep_noise=True))
        dedupe_key = stable_hash(f"{extract_package(raw_image)}\n{text_for_hash}")
        if dedupe and dedupe_key in seen:
            counters["skipped_duplicates"] += 1
            continue
        seen.add(dedupe_key)
        counters["accepted_sources"] += 1
        answer = build_answer(row, mask_sensitive=mask_sensitive)
        redact_boxes = sensitive_boxes(line_meta) if redact_sensitive_images else []
        target_image = image_dir / image_target_name(source_index, source_image, row)
        image_for_training = source_image
        image_rel = ""
        if copy_images or redact_sensitive_images:
            image_for_training = target_image
            image_rel = str(target_image.relative_to(out_dir)).replace("\\", "/")
            if not dry_run:
                copy_image(source_image, target_image, redact_boxes)
        for variant in range(max(1, prompt_variants)):
            item_id = stable_hash(f"{source_index}:{variant}:{raw_image}:{text_for_hash}", 20)
            prompt = build_prompt(row, variant=variant, mask_sensitive=mask_sensitive)
            conversations = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": compact_json(answer)},
            ]
            item = {
                "id": item_id,
                "schema_version": "zemo_o_minimind3o_sft_v1",
                "source": "zemo_real_ocr_cpp_v2",
                "task": "image_ocr_to_zemo_screen_json",
                "ocr_jsonl": str(ocr_jsonl),
                "source_index": source_index,
                "source_file": raw_image,
                "image_path": str(image_for_training),
                "image": image_rel,
                "scene": answer["scene"],
                "intent": answer["intent"],
                "action": answer["action"],
                "prompt_variant": variant,
                "conversations": conversations,
                "answer": answer,
                "ocr": {
                    "format": row.get("format", "zemo_ocr_lines_v2"),
                    "width": int(row.get("w", 0) or 0),
                    "height": int(row.get("h", 0) or 0),
                    "ms": int(row.get("ms", 0) or 0),
                    "lines": mask_sensitive_obj(line_meta) if mask_sensitive else line_meta,
                },
                "privacy": {
                    "text_masked": mask_sensitive,
                    "image_redacted": redact_sensitive_images,
                    "sensitive_box_count": len(redact_boxes),
                },
            }
            rows.append(item)
    report = {
        "raw_rows": len(ocr_rows),
        "accepted_source_rows": counters["accepted_sources"],
        "written_sft_rows": len(rows),
        "prompt_variants": max(1, prompt_variants),
        "dedupe": dedupe,
        "skipped_duplicates": counters["skipped_duplicates"],
        "skipped_missing_image": counters["skipped_missing_image"],
        "skipped_unpackage": counters["skipped_unpackage"],
        "mask_sensitive_text": mask_sensitive,
        "redact_sensitive_images": redact_sensitive_images,
        "copy_images": copy_images or redact_sensitive_images,
    }
    return rows, report


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> str:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        return f"skip parquet: {exc}"

    image_bytes = []
    conversations = []
    ids = []
    source_files = []
    images = []
    scenes = []
    intents = []
    actions = []
    for row in rows:
        image_path = Path(row["image_path"])
        image_bytes.append(image_path.read_bytes() if image_path.exists() and not path_has_unpackage(str(image_path)) else b"")
        conversations.append(compact_json(row["conversations"]))
        ids.append(row["id"])
        source_files.append(row["source_file"])
        images.append(row["image"])
        scenes.append(row["scene"])
        intents.append(row["intent"])
        actions.append(row["action"])
    table = pa.Table.from_pydict(
        {
            "id": ids,
            "image_bytes": image_bytes,
            "conversations": conversations,
            "source_file": source_files,
            "image": images,
            "scene": scenes,
            "intent": intents,
            "action": actions,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return str(path)


def copy_to_minimind_o(parquet_path: Path, minimind_o_root: Path) -> str:
    if not parquet_path.exists():
        return ""
    dataset_dir = minimind_o_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / parquet_path.name
    shutil.copy2(parquet_path, target)
    return str(target)


def build_report(base: dict[str, Any], rows: list[dict[str, Any]], out_jsonl: Path, out_parquet_status: str, out_dir: Path) -> dict[str, Any]:
    scene_counts = Counter(row["scene"] for row in rows)
    intent_counts = Counter(row["intent"] for row in rows)
    samples = [
        {
            "id": row["id"],
            "image": row["image"],
            "scene": row["scene"],
            "intent": row["intent"],
            "summary": row["answer"].get("summary", ""),
            "requires_review": row["answer"].get("requires_review", False),
        }
        for row in rows[:12]
    ]
    report = dict(base)
    report.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "out_dir": str(out_dir),
            "out_jsonl": str(out_jsonl),
            "out_parquet": out_parquet_status,
            "scene_counts": dict(scene_counts),
            "intent_counts": dict(intent_counts),
            "schema": {
                "parquet_required_columns": ["image_bytes", "conversations"],
                "image_marker": "<image>",
                "answer_schema_version": "zemo_o_screen_sft_v1",
            },
            "samples": samples,
        }
    )
    return report


def default_out_dir(ocr_dir: Path) -> Path:
    # Input is normally .../dataset/img/_ocr_out_cpp_v2; keep generated SFT outside img.
    if ocr_dir.name.startswith("_ocr_out"):
        dataset_dir = ocr_dir.parent.parent
        return dataset_dir / "sft" / "zemo_o_real_ocr_sft"
    return ocr_dir / "_zemo_o_real_ocr_sft"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ZeMo-O MiniMind3O SFT data from real OCR rows and images.")
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--ocr-jsonl", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    parser.add_argument("--out-parquet", type=Path, default=None)
    parser.add_argument("--out-report", type=Path, default=None)
    parser.add_argument("--prompt-variants", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--mask-sensitive", dest="mask_sensitive", action="store_true", default=True)
    parser.add_argument("--no-mask-sensitive", dest="mask_sensitive", action="store_false")
    parser.add_argument("--copy-images", dest="copy_images", action="store_true", default=True)
    parser.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    parser.add_argument("--redact-sensitive-images", action="store_true")
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--copy-to-minimind-o", action="store_true")
    parser.add_argument("--minimind-o-root", type=Path, default=DEFAULT_MINIMIND_O_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ocr_dir = args.ocr_dir.resolve()
    ocr_jsonl = (args.ocr_jsonl or (ocr_dir / "ocr_results.jsonl")).resolve()
    out_dir = (args.out_dir or default_out_dir(ocr_dir)).resolve()
    out_jsonl = (args.out_jsonl or (out_dir / "zemo_o_real_ocr_sft.jsonl")).resolve()
    out_parquet = (args.out_parquet or (out_dir / "zemo_o_real_ocr_sft.parquet")).resolve()
    out_report = (args.out_report or (out_dir / "zemo_o_real_ocr_sft.report.json")).resolve()

    if path_has_unpackage(str(ocr_jsonl)) or path_has_unpackage(str(out_dir)):
        raise ValueError("refuse to read or write unpackage path")
    if not ocr_jsonl.exists():
        raise FileNotFoundError(f"OCR jsonl not found: {ocr_jsonl}")

    ocr_rows = load_jsonl(ocr_jsonl)
    rows, base_report = build_dataset_rows(
        ocr_rows=ocr_rows,
        ocr_jsonl=ocr_jsonl,
        out_dir=out_dir,
        prompt_variants=args.prompt_variants,
        dedupe=not args.keep_duplicates,
        mask_sensitive=args.mask_sensitive,
        copy_images=args.copy_images,
        redact_sensitive_images=args.redact_sensitive_images,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    parquet_status = ""
    copied = ""
    if not args.dry_run:
        write_jsonl(out_jsonl, rows)
        if not args.no_parquet:
            parquet_status = write_parquet(out_parquet, rows)
        if args.copy_to_minimind_o and parquet_status == str(out_parquet):
            copied = copy_to_minimind_o(out_parquet, args.minimind_o_root.resolve())
    else:
        parquet_status = "dry-run"

    report = build_report(base_report, rows, out_jsonl, parquet_status, out_dir)
    if copied:
        report["copied_to_minimind_o"] = copied
    if not args.dry_run:
        write_json(out_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
