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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OCR_DIR = Path(
    r"E:\CamXAll\ZEMO\Data\Model\Tools\zemo_ppocrv6_windows_cpp_ocr\deploy\_smoke_out_v2"
)
PROCESSED_DIR = ROOT / "model" / "data" / "processed"
DEFAULT_OUT_JSONL = PROCESSED_DIR / "zemo_real_ocr_screen_pretrain.jsonl"
DEFAULT_OUT_REPORT = PROCESSED_DIR / "zemo_real_ocr_screen_pretrain.report.json"
DEFAULT_OUT_PARQUET = PROCESSED_DIR / "zemo_real_ocr_screen_pretrain.parquet"
DEFAULT_MINIMIND_O_ROOT = ROOT / "model" / "minimind-o"


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
KINDS = ["bill", "investment", "chat", "activity", "note", "general"]
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
    "com.alipay.mobile.client": "支付宝",
    "com.eg.android.alipaygphone": "支付宝",
    "com.cmbchina.harmony": "招商银行",
    "cmb.pb": "招商银行",
    "com.csg.palmhall": "南方电网",
    "com.greenpoint.android.mc10086.activity": "中国移动",
    "com.hexin.hmn.sjcg": "同花顺",
    "com.tdx.harmonypub": "通达信",
    "com.app.yangjibao": "养基宝",
    "yylx.danmaku.bili": "哔哩哔哩",
    "com.sjz.ss": "影视/动漫",
    "com.tencent.videohm": "腾讯视频",
    "com.qiyi.video.hmy": "爱奇艺",
    "com.ss.android.ugc.aweme": "抖音",
    "com.tencent.hm.qqmusic": "QQ音乐",
    "com.luna.hm.music": "汽水音乐",
    "com.taobao.taobao4hmos": "淘宝",
    "com.taobao.idlefish": "闲鱼",
    "com.taobao.qianniu": "千牛",
    "com.jd.hm.mall": "京东",
    "com.jingdong.app.mall": "京东",
    "com.xunmeng.pinduoduo": "拼多多",
    "com.alibaba.wireless": "1688",
    "com.xiaomi.shop": "小米商城",
    "com.microsoft.emmx": "Edge",
    "com.larus.nova.hm": "豆包",
    "md.obsidian": "Obsidian",
    "zenith.most.zemo": "ZeMo",
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
            "transactions": "array",
        }
    },
    "investment_import": {
        "investment": {
            "account": "string",
            "total_assets": "string",
            "daily_profit": "string",
            "total_profit": "string",
            "holdings": "array",
            "orders": "array",
        }
    },
    "chat_todo": {
        "todos": [
            {
                "title": "string",
                "due_at": "string",
                "priority": "low|normal|high",
                "source_text": "string",
                "entities": "object",
            }
        ]
    },
    "entertainment_activity": {
        "entertainment": {
            "media_type": "music|anime|video|short_video|live|other",
            "title": "string",
            "artist_or_author": "string",
            "episode": "string",
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
    re.compile(r"^[<>く×+·.。|/\\\s]+$"),
    re.compile(r"^[kK]?[0-9A-Za-z./\s]{1,14}$"),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad jsonl at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def strip_duplicate_suffix(value: str) -> str:
    return re.sub(r"\(\d+\)$", "", value)


def extract_package(image_path: str) -> str:
    stem = strip_duplicate_suffix(Path(image_path).stem)
    patterns = [
        r"(?i)^(?:screenshot|screen)_(?:\d{8}_\d{6})_(?P<pkg>.+)$",
        r"(?i)^screenshot_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:-\d+)?_(?P<pkg>.+)$",
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
        (r"(?i)screenshot_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", False),
        (r"(?i)screenshot_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", False),
        (r"(?i)(?:img|mvimg)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", False),
        (r"ChatGPT Image (\d{4})年(\d{1,2})月(\d{1,2})日 (\d{1,2})_(\d{1,2})_(\d{1,2})", False),
    ]
    for pattern, _ in patterns:
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
    if not key:
        return ""
    for needle, name in APP_NAMES.items():
        if needle in key:
            return name
    return package


def clean_text(value: str) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_line(value: str) -> str:
    value = clean_text(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def is_noise_line(value: str) -> bool:
    text = normalize_line(value)
    if not text:
        return True
    if re.match(r"^[+\-＋－]\s*[¥￥]?\d+(?:,\d{3})*(?:\.\d{1,4})?%?$", text):
        return False
    if re.match(r"^[¥￥]\s*\d+(?:,\d{3})*(?:\.\d{1,2})?$", text):
        return False
    if re.match(r"^\d{1,3}(?:,\d{3})+(?:\.\d{1,4})?$", text):
        return False
    if re.match(r"^\d+\.\d{1,4}%?$", text):
        return False
    if re.search(r"20\d{2}[-./年]\d{1,2}[-./月]\d{1,2}", text):
        return False
    if len(text) <= 1 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return True
    for pattern in NOISE_LINE_PATTERNS:
        if pattern.match(text):
            return True
    if text in {"复制", "查看", "更多", "展开", "筛选", "搜索", "返回", "完成", "我的", "首页"}:
        return True
    return False


def sorted_ocr_lines(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in row.get("lines", []) or []:
        text = normalize_line(str(item.get("t", "")))
        box = item.get("b", [0, 0, 0, 0]) or [0, 0, 0, 0]
        if len(box) < 4:
            box = [0, 0, 0, 0]
        try:
            conf = float(item.get("c", 0.0))
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
    return [v for v in values if not is_noise_line(v)]


def plain_text(lines: list[dict[str, Any]], keep_noise: bool = False, limit: int = 5000) -> str:
    text = "\n".join(plain_lines(lines, keep_noise=keep_noise))
    return text[:limit]


def ocr_meta_text(lines: list[dict[str, Any]], limit_lines: int = 80) -> str:
    parts = []
    for item in lines[:limit_lines]:
        box = item["bbox"]
        parts.append(f"[[ocr:{box[0]},{box[1]},{box[2]},{box[3]}]]{item['text']}")
    return "\n".join(parts)


def contains_any(text: str, words: list[str]) -> bool:
    return any(word.lower() in text for word in words)


def text_blob(lines: list[str], package: str) -> str:
    return (package + "\n" + "\n".join(lines)).lower()


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def score_keywords(blob: str, words: list[str], weight: int = 1) -> int:
    return sum(weight for word in words if word.lower() in blob)


def looks_like_chat(lines: list[str], package: str) -> bool:
    blob = text_blob(lines, package)
    if contains_any(blob, ["群聊", "聊天", "语音", "视频通话", "撤回了一条消息"]):
        return True
    if "wechat" in package or "tencent.mm" in package or "ohos.mms" in package:
        speaker_lines = 0
        for line in lines:
            if re.match(r"^.{1,14}[:：]\s*.{2,}$", line):
                speaker_lines += 1
        return speaker_lines >= 1 or len(lines) >= 8
    return False


def classify_scene(lines: list[str], package: str) -> dict[str, Any]:
    blob = text_blob(lines, package)
    pkg = package.lower()
    evidence: list[str] = []

    investment_score = score_keywords(
        blob,
        ["基金", "持仓", "证券", "股票", "总资产", "市值", "盈亏", "收益率", "ETF", "净值", "份额", "仓位", "买入", "卖出"],
    )
    if contains_any(pkg, ["hexin", "tdx", "yangjibao"]) or investment_score >= 4 or (
        "alipay" in pkg and contains_any(blob, ["基金", "持有品", "持有收益"])
    ):
        evidence.append("理财包名/持仓关键词")
        return {
            "scene": "理财",
            "intent": "investment_import",
            "kind": "investment",
            "sub_scene": "fund_or_stock_holding" if contains_any(blob, ["持仓", "持有", "市值"]) else "investment_screen",
            "action": "import_investment",
            "title": "理财/持仓识别",
            "confidence": 0.92 if investment_score >= 4 else 0.82,
            "evidence": evidence,
        }

    if contains_any(blob, ["社保", "参保证明", "社保卡", "销户", "申领", "工单", "就业补贴", "政务"]):
        evidence.append("政务/办事信息")
        return {
            "scene": "政务",
            "intent": "note_extract",
            "kind": "note",
            "sub_scene": "government_service",
            "action": "create_note",
            "title": "政务/办事记录",
            "confidence": 0.82,
            "evidence": evidence,
        }

    logistics_words = ["取件码", "快递待取", "待取件", "取快递", "待揽收", "运单号", "物流详情", "派送中", "寄件码"]
    if score_keywords(blob, logistics_words) >= 1 and contains_any(blob, ["快递", "顺丰", "京东", "韵达", "申通", "物流", "驿", "包裹", "取件"]):
        evidence.append("快递/物流待办关键词")
        return {
            "scene": "待办",
            "intent": "chat_todo",
            "kind": "chat",
            "sub_scene": "package_pickup" if contains_any(blob, ["取件码", "待取", "取快递"]) else "delivery_tracking",
            "action": "create_todos",
            "title": "快递待办",
            "confidence": 0.94,
            "evidence": evidence,
        }

    bill_score = score_keywords(
        blob,
        [
            "账单详情",
            "全部账单",
            "收支",
            "交易成功",
            "交易关闭",
            "实付款",
            "支付金额",
            "订单金额",
            "付款方式",
            "支付方式",
            "支付时间",
            "实际应付",
            "本期账单",
            "话费账单",
            "扣款",
            "退款成功",
            "已自动支付",
            "结息",
        ],
    )
    bill_pkg = contains_any(
        pkg,
        [
            "alipay",
            "cmbchina",
            "palmhall",
            "mc10086",
            "delivery.aggregator",
            "jd.hm.mall",
            "jingdong",
            "taobao",
            "xunmeng",
            "wechat",
            "tencent.mm",
            "ohos.mms",
        ],
    )
    if bill_score >= 2 or (bill_pkg and bill_score >= 1):
        evidence.append("账单/支付/收支关键词")
        sub_scene = "bill_detail"
        if contains_any(blob, ["收支", "结余", "储蓄卡", "银行卡"]):
            sub_scene = "bank_statement"
        elif contains_any(blob, ["话费账单", "10086", "应付"]):
            sub_scene = "telecom_bill"
        elif contains_any(blob, ["退款", "退回"]):
            sub_scene = "refund_bill"
        elif contains_any(blob, ["订单编号", "实付款", "支付方式"]):
            sub_scene = "order_payment"
        return {
            "scene": "记账",
            "intent": "bill_record",
            "kind": "bill",
            "sub_scene": sub_scene,
            "action": "create_transaction",
            "title": "账单/交易识别",
            "confidence": 0.90 if bill_score >= 2 else 0.78,
            "evidence": evidence,
        }

    chat_todo_score = score_keywords(
        blob,
        [
            "记得",
            "提醒",
            "待办",
            "安排",
            "提交",
            "确认",
            "处理",
            "跟进",
            "缴费",
            "还款",
            "转账",
            "报销",
            "付款",
            "明天",
            "后天",
            "今晚",
            "今天",
            "截止",
            "联系这个号码",
            "五分钟后",
        ],
    )
    if looks_like_chat(lines, package) and chat_todo_score >= 1:
        evidence.append("聊天/短信中的行动项")
        return {
            "scene": "待办",
            "intent": "chat_todo",
            "kind": "chat",
            "sub_scene": "chat_todo",
            "action": "create_todos",
            "title": "聊天待办",
            "confidence": 0.82,
            "evidence": evidence,
        }

    entertainment_pkg = contains_any(
        pkg,
        ["qqmusic", "luna.hm.music", "bili", "danmaku", "sjz.ss", "videohm", "qiyi", "ugc.aweme"],
    )
    entertainment_score = score_keywords(
        blob,
        ["正在播放", "歌词", "歌曲", "歌单", "每日30首", "听歌", "弹幕", "选集", "动漫", "电视剧", "电影", "综艺", "直播", "短视频", "VIP", "倍速"],
    )
    if entertainment_pkg or entertainment_score >= 3:
        media_type = "other"
        sub_scene = "entertainment"
        if contains_any(pkg + blob, ["qqmusic", "luna.hm.music", "歌词", "歌曲", "歌单", "听歌", "每日30首"]):
            media_type = "music"
            sub_scene = "music_listening"
        elif contains_any(pkg + blob, ["ugc.aweme", "抖音", "短视频"]):
            media_type = "short_video"
            sub_scene = "short_video"
        elif contains_any(pkg + blob, ["bili", "danmaku", "动漫", "番剧", "吞噬星空", "排期表"]):
            media_type = "anime"
            sub_scene = "anime_watching"
        elif contains_any(pkg + blob, ["videohm", "qiyi", "电视剧", "电影", "综艺", "选集"]):
            media_type = "video"
            sub_scene = "video_watching"
        if "直播" in blob:
            media_type = "live"
            sub_scene = "live_stream"
        evidence.append(f"娱乐包名/播放关键词:{media_type}")
        return {
            "scene": "娱乐",
            "intent": "entertainment_activity",
            "kind": "activity",
            "sub_scene": sub_scene,
            "action": "record_activity",
            "title": "娱乐活动",
            "confidence": 0.92 if entertainment_pkg else 0.78,
            "media_type": media_type,
            "evidence": evidence,
        }

    shopping_pkg = contains_any(pkg, ["taobao", "jingdong", "jd.hm.mall", "xunmeng", "pinduoduo", "idlefish", "xiaomi.shop", "alibaba.wireless", "qianniu"])
    shopping_score = score_keywords(blob, ["商品", "订单", "购物车", "待收货", "待付款", "店铺", "客服", "退款详情", "售后", "再次购买", "加入购物车"])
    if shopping_pkg or shopping_score >= 3:
        evidence.append("购物/订单浏览关键词")
        return {
            "scene": "购物",
            "intent": "shopping_activity",
            "kind": "activity",
            "sub_scene": "shopping_order" if contains_any(blob, ["订单", "待收货", "退款", "售后"]) else "shopping_browse",
            "action": "record_activity",
            "title": "购物活动",
            "confidence": 0.84 if shopping_pkg else 0.72,
            "evidence": evidence,
        }

    if contains_any(blob, ["车票", "出票成功", "取票号", "车厢", "改签成功", "订单行程服务"]) or re.search(r"\bG\d{2,5}\b", blob, re.I):
        evidence.append("出行票务关键词")
        return {
            "scene": "出行",
            "intent": "note_extract",
            "kind": "note",
            "sub_scene": "travel_ticket",
            "action": "create_note",
            "title": "出行票据",
            "confidence": 0.86,
            "evidence": evidence,
        }

    if contains_any(blob, ["睡眠", "健康记录", "睡眠质量", "同龄用户", "探索睡眠动物"]):
        evidence.append("健康记录关键词")
        return {
            "scene": "健康",
            "intent": "note_extract",
            "kind": "note",
            "sub_scene": "health_record",
            "action": "create_note",
            "title": "健康记录",
            "confidence": 0.76,
            "evidence": evidence,
        }

    if contains_any(blob, ["居民身份证", "公民身份号码", "学生 number", "student number", "school:", "毕业", "有效期限", "住址"]):
        evidence.append("证件/资料文本")
        return {
            "scene": "记事",
            "intent": "note_extract",
            "kind": "note",
            "sub_scene": "document_or_id",
            "action": "create_note",
            "title": "证件/资料摘录",
            "confidence": 0.86,
            "evidence": evidence,
        }

    if contains_any(blob, ["豆包", "内容由 ai 生成", "obsidian", "投诉", "公司名称", "备忘", "笔记", "制图", "写下明天"]):
        evidence.append("笔记/AI 对话/文档信息")
        return {
            "scene": "记事",
            "intent": "note_extract",
            "kind": "note",
            "sub_scene": "note_or_ai_chat",
            "action": "create_note",
            "title": "记事摘要",
            "confidence": 0.78,
            "evidence": evidence,
        }

    if not lines:
        evidence.append("无有效 OCR 文本")
        return {
            "scene": "其他",
            "intent": "screen_summary",
            "kind": "general",
            "sub_scene": "empty_or_visual_only",
            "action": "summarize_screen",
            "title": "无文字截图",
            "confidence": 0.45,
            "evidence": evidence,
        }

    evidence.append("默认摘要")
    return {
        "scene": "记事",
        "intent": "note_extract",
        "kind": "note",
        "sub_scene": "general_note",
        "action": "create_note",
        "title": "屏幕文字摘要",
        "confidence": 0.62,
        "evidence": evidence,
    }


def normalize_amount(raw: str) -> str:
    value = raw.replace("￥", "").replace("¥", "").replace("元", "").replace(",", "").replace(" ", "")
    value = value.replace("＋", "+").replace("－", "-")
    return value.strip()


def extract_amount_candidates(lines: list[str]) -> list[dict[str, Any]]:
    amount_re = re.compile(r"(?<!\d)(?P<sign>[+\-＋－]?)\s*[¥￥]?\s*(?P<num>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)\s*(?:元)?")
    label_words = [
        "实付款",
        "支付金额",
        "订单金额",
        "金额",
        "实际应付",
        "本期账单",
        "账单",
        "合计",
        "支出",
        "收入",
        "余额",
        "扣款",
        "总资产",
        "账户资产",
        "总市值",
        "可用资金",
        "已退款",
        "退款记录",
        "退款总金额",
    ]
    discount_words = ["红包", "立减", "优惠", "减免", "补贴", "原价", "券", "折扣"]
    candidates = []
    for idx, line in enumerate(lines):
        if re.search(r"\d{4}[-./年]\d{1,2}[-./月]\d{1,2}", line):
            continue
        if re.fullmatch(r"20\d{2}年?", line.strip()):
            continue
        if re.fullmatch(r"20\d{4}", line.strip()):
            continue
        if re.search(r"\d{8,}", line) and not re.search(r"[¥￥元+\-＋－]", line):
            continue
        if "****" in line and not re.search(r"[¥￥元+\-＋－]", line):
            continue
        if idx <= 1 and not re.search(r"[¥￥元+\-＋－]", line) and not contains_any(line, label_words):
            continue
        near = "\n".join(lines[max(0, idx - 3) : min(len(lines), idx + 3)])
        for match in amount_re.finditer(line):
            raw = match.group(0)
            amount = normalize_amount(raw)
            if not amount:
                continue
            score = 0
            if "¥" in raw or "￥" in raw or "元" in raw:
                score += 4
            if match.group("sign"):
                score += 3
            if contains_any(near, label_words):
                score += 5
            if contains_any(line, label_words):
                score += 3
            if re.match(r"^[+\-＋－]?\s*[¥￥]?\s*\d+(?:\.\d{1,2})?\s*$", line) and contains_any(near, ["交易成功", "支付成功", "账单详情", "全部账单"]):
                score += 7
            if contains_any(line, ["实付款", "合计", "实际应付", "支付金额"]):
                score += 6
            if contains_any(near, ["已退款", "退款记录", "退款总金额"]):
                score += 6
            if "." in amount:
                score += 2
            if contains_any(near, ["用电量"]) and "." not in amount and not re.search(r"[¥￥元+\-＋－]", raw):
                score -= 5
            if contains_any(line, discount_words):
                score -= 7
            if contains_any(near, discount_words) and not contains_any(line, ["实付款", "合计", "实际应付", "支付金额"]):
                score -= 2
            if re.search(r"^\d{1,2}[:：]\d{2}", line):
                score -= 6
            if score > 0 and float(amount.replace("+", "").replace("-", "") or "0") >= 0.01:
                score += 1
            if score > 1:
                candidates.append({"amount": amount, "line": line, "index": idx, "score": score})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates


def next_value_after(lines: list[str], keys: list[str], max_lookahead: int = 2) -> str:
    for idx, line in enumerate(lines):
        if not contains_any(line, keys):
            continue
        for j in range(idx + 1, min(len(lines), idx + 1 + max_lookahead)):
            candidate = lines[j].strip(" >：:")
            if candidate and not is_noise_line(candidate):
                return candidate
        tail = re.sub("|".join(re.escape(k) for k in keys), "", line).strip(" :：>")
        if tail:
            return tail
    return ""


def extract_first_datetime(lines: list[str], fallback_screen_time: str) -> str:
    text = "\n".join(lines)
    patterns = [
        r"(20\d{2})[-./年](\d{1,2})[-./月](\d{1,2})[日\s-]*(\d{1,2})[:：](\d{2})(?::(\d{2}))?",
        r"(\d{1,2})月(\d{1,2})日\s*(\d{1,2})[:：](\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        groups = match.groups(default="0")
        try:
            if len(groups) == 6:
                y, mo, d, h, mi, s = [int(v) for v in groups]
            else:
                if fallback_screen_time:
                    y = datetime.fromisoformat(fallback_screen_time).year
                else:
                    y = datetime.now().year
                mo, d, h, mi = [int(v) for v in groups[:4]]
                s = 0
            return datetime(y, mo, d, h, mi, s).isoformat()
        except ValueError:
            return match.group(0)
    return ""


def extract_order_no(lines: list[str]) -> str:
    for idx, line in enumerate(lines):
        if contains_any(line, ["订单编号", "交易单号", "商户单号", "运单号", "快递单号"]):
            joined = " ".join(lines[idx : min(len(lines), idx + 2)])
            match = re.search(r"\b(?:SF|JD|YT|YD|ZTO|STO|EMS)?[A-Z0-9]{8,}\b", joined, re.I)
            if match:
                return match.group(0)
    return ""


def guess_merchant(lines: list[str]) -> str:
    skip_words = ["账单详情", "全部账单", "交易成功", "订单金额", "支付时间", "付款方式", "支付方式", "更多", "账单管理", "金额", "实付款"]
    for idx, line in enumerate(lines):
        if contains_any(line, ["账单详情", "全部账单"]):
            for candidate in lines[idx + 1 : idx + 5]:
                if not is_noise_line(candidate) and not contains_any(candidate, skip_words) and not re.match(r"^[+\-]?[¥￥]?\d", candidate):
                    return candidate.strip(" >")
    for line in lines[:16]:
        if is_noise_line(line) or contains_any(line, skip_words):
            continue
        if contains_any(line, ["店", "公司", "超市", "旗舰", "商户", "外卖", "京东", "淘宝", "拼多多", "银行", "移动"]):
            return line.strip(" >")
    return ""


def extract_payment_method(lines: list[str]) -> str:
    method_words = ["花呗", "余额", "零钱", "银行卡", "储蓄卡", "信用卡", "白条", "支付宝", "微信", "现金", "寄付现结"]
    for idx, line in enumerate(lines):
        if not contains_any(line, ["付款方式", "支付方式"]):
            continue
        context = lines[max(0, idx - 2) : min(len(lines), idx + 3)]
        for value in context:
            if value == line:
                continue
            if contains_any(value, method_words):
                return value.strip(" >")
        return ""
    return ""


def extract_transactions(lines: list[str], limit: int = 8) -> list[dict[str, str]]:
    candidates = extract_amount_candidates(lines)
    result = []
    seen = set()
    for item in candidates:
        idx = item["index"]
        if idx in seen:
            continue
        context = lines[max(0, idx - 1) : min(len(lines), idx + 2)]
        title = ""
        for value in context:
            if value != item["line"] and not is_noise_line(value):
                title = value
                break
        result.append({"amount": item["amount"], "title": title, "source_text": " / ".join(context)})
        seen.add(idx)
        if len(result) >= limit:
            break
    return result


def extract_bill(lines: list[str], package: str, screen_time: str) -> dict[str, Any]:
    blob = text_blob(lines, package)
    amount_candidates = extract_amount_candidates(lines)
    amount = amount_candidates[0]["amount"] if amount_candidates else ""
    mode = "expense"
    if contains_any(blob, ["退款", "退回", "退款成功", "返还", "已退款"]):
        mode = "refund"
    elif amount.startswith("+"):
        mode = "income"
    elif amount.startswith("-"):
        mode = "expense"
    elif contains_any(blob, ["入账", "收入", "到账", "转入", "结息", "收款成功", "已收款"]):
        mode = "income"
    elif contains_any(blob, ["转账", "转来", "转给"]):
        mode = "transfer"
    pay_method = extract_payment_method(lines)
    category = next_value_after(lines, ["账单分类", "消费项目", "账单类型"])
    return {
        "amount": amount,
        "mode": mode,
        "merchant": guess_merchant(lines),
        "pay_method": pay_method,
        "transaction_time": extract_first_datetime(lines, screen_time),
        "order_no": extract_order_no(lines),
        "category": category,
        "source_app": app_name(package),
        "transactions": extract_transactions(lines),
    }


def extract_percent(line: str) -> str:
    match = re.search(r"[+\-＋－]?\d+(?:\.\d+)?%", line)
    return match.group(0).replace("＋", "+").replace("－", "-") if match else ""


def extract_investment(lines: list[str], package: str) -> dict[str, Any]:
    text = "\n".join(lines)
    amounts = extract_amount_candidates(lines)
    total_assets = ""
    daily_profit = ""
    total_profit = ""
    for idx, line in enumerate(lines):
        near = " ".join(lines[idx : min(len(lines), idx + 2)])
        if total_assets == "" and contains_any(line, ["总资产", "账户资产", "人民币账户"]):
            vals = extract_amount_candidates(lines[idx : min(len(lines), idx + 5)])
            vals = sorted(vals, key=lambda x: abs(float(x["amount"].replace("+", "").replace("-", "") or "0")), reverse=True)
            if vals:
                total_assets = vals[0]["amount"]
        if daily_profit == "" and contains_any(line, ["当日收益", "昨日收益", "当日盈亏", "昨日"]):
            vals = extract_amount_candidates([near])
            if vals:
                daily_profit = vals[0]["amount"]
        if total_profit == "" and contains_any(line, ["总盈亏", "持有收益", "盈亏"]):
            vals = extract_amount_candidates([near])
            if vals:
                total_profit = vals[0]["amount"]

    holdings = []
    seen = set()
    name_words = ["基金", "ETF", "etf", "纳斯达克", "标普", "中债", "全球", "混合", "股票", "证券", "纳100", "纳指"]
    for idx, line in enumerate(lines):
        code_match = re.search(r"\b\d{6}\b", line)
        is_name = contains_any(line, name_words)
        if not code_match and not is_name:
            continue
        context = lines[max(0, idx - 2) : min(len(lines), idx + 5)]
        joined = " ".join(context)
        code = code_match.group(0) if code_match else ""
        if not code:
            code_match2 = re.search(r"\b\d{6}\b", joined)
            code = code_match2.group(0) if code_match2 else ""
        name = line
        if re.match(r"^[+\-]?\d", name) and idx > 0:
            name = lines[idx - 1]
        name = re.sub(r"\b\d{6}\b", "", name).strip(" -：:/")
        key = f"{name}|{code}"
        if key in seen or (not name and not code):
            continue
        seen.add(key)
        values = extract_amount_candidates(context)
        percent = extract_percent(joined)
        holdings.append(
            {
                "name": name,
                "code": code,
                "market_value": values[0]["amount"] if values else "",
                "profit": values[1]["amount"] if len(values) > 1 else "",
                "profit_rate": percent,
                "units": "",
                "nav": "",
                "source_text": " / ".join(context[:5]),
            }
        )
        if len(holdings) >= 8:
            break
    return {
        "account": app_name(package),
        "total_assets": total_assets,
        "daily_profit": daily_profit,
        "total_profit": total_profit,
        "holdings": holdings,
        "orders": [],
        "raw_numbers": [x["amount"] for x in amounts[:12]],
        "raw_text_hint": text[:300],
    }


def extract_pickup_codes(lines: list[str]) -> list[str]:
    codes = []
    for idx, line in enumerate(lines):
        if not contains_any(line, ["取件码", "寄件码"]):
            continue
        joined = " ".join(lines[idx : min(len(lines), idx + 2)])
        for match in re.finditer(r"(?:取件码|寄件码)\s*[:：]?\s*([A-Z0-9\-]{3,20})", joined, re.I):
            codes.append(match.group(1))
        if not codes:
            for match in re.finditer(r"\b[A-Z]?\d[\d\-]{2,18}\b", joined, re.I):
                codes.append(match.group(0))
    return list(dict.fromkeys(codes))[:8]


def extract_express_numbers(lines: list[str]) -> list[str]:
    text = "\n".join(lines)
    result = []
    for match in re.finditer(r"\b(?:SF|JD|YT|YD|ZTO|STO|EMS)?[A-Z0-9]{10,22}\b", text, re.I):
        value = match.group(0)
        if re.match(r"20\d{10,}", value):
            continue
        result.append(value)
    return list(dict.fromkeys(result))[:10]


def clean_todo_title(line: str) -> str:
    value = re.sub(r"^[^:：]{1,14}[:：]\s*", "", line).strip()
    value = re.sub(r"^(麻烦|帮我|请|你|我们|大家|记得|提醒我|提醒一下|待办[:：]?)", "", value)
    value = re.sub(r"[，。,.!！?？]+$", "", value).strip()
    return value[:48]


def is_bad_todo_line(line: str) -> bool:
    value = normalize_line(line)
    if len(value) < 4:
        return True
    ui_only = [
        "筛选",
        "批量处理",
        "全部",
        "待付款",
        "待收货",
        "待使用",
        "已完成",
        "售后",
        "订单管理",
        "搜索我的订单",
        "充值",
        "账单详情",
    ]
    if all(word in value for word in ["筛选", "批量处理"]):
        return True
    if value.replace(" ", "") in {"全部待提货待送货已提货售后", "全部待付款待收货待使用已完成"}:
        return True
    if contains_any(value, ui_only) and len(value) <= 18:
        return True
    return False


def extract_due_text(line: str) -> str:
    patterns = [
        r"(今天|明天|后天|今晚|下周[一二三四五六日天]?|周[一二三四五六日天]|月底|截止)",
        r"\d{1,2}[月/\-]\d{1,2}[日号]?",
        r"\d{1,2}[:：]\d{2}",
        r"20\d{2}[-./年]\d{1,2}[-./月]\d{1,2}[日]?",
    ]
    values = []
    for pattern in patterns:
        values += re.findall(pattern, line)
    return " ".join([v if isinstance(v, str) else "".join(v) for v in values])


def extract_todos(lines: list[str], sub_scene: str) -> list[dict[str, Any]]:
    todos = []
    if sub_scene in {"package_pickup", "delivery_tracking"}:
        pickup_codes = extract_pickup_codes(lines)
        express_numbers = extract_express_numbers(lines)
        station = ""
        for line in lines:
            if contains_any(line, ["驿", "代收点", "菜鸟", "小区", "顺丰驿收发"]):
                station = line
                break
        title = "取快递"
        if pickup_codes:
            title += f"（取件码 {pickup_codes[0]}）"
        todos.append(
            {
                "title": title,
                "due_at": "",
                "priority": "normal",
                "source_text": " / ".join(lines[:8]),
                "entities": {
                    "pickup_codes": pickup_codes,
                    "express_numbers": express_numbers,
                    "pickup_place": station,
                },
            }
        )
        return todos

    todo_words = [
        "记得",
        "提醒",
        "安排",
        "提交",
        "确认",
        "处理",
        "跟进",
        "缴费",
        "还款",
        "转账",
        "报销",
        "付款",
        "联系",
        "五分钟后",
        "明天",
        "后天",
        "今晚",
        "截止",
    ]
    for line in lines:
        if is_bad_todo_line(line):
            continue
        if not contains_any(line, todo_words):
            continue
        title = clean_todo_title(line)
        if len(title) < 3:
            continue
        todos.append(
            {
                "title": title,
                "due_at": extract_due_text(line),
                "priority": "high" if contains_any(line, ["今天", "今晚", "马上", "截止"]) else "normal",
                "source_text": line,
                "entities": {
                    "phones": re.findall(r"\b1[3-9]\d{9}\b", line),
                    "amounts": [x["amount"] for x in extract_amount_candidates([line])],
                },
            }
        )
        if len(todos) >= 6:
            break
    return todos


def meaningful_title(lines: list[str], fallback: str = "") -> str:
    for line in lines:
        if is_noise_line(line):
            continue
        if len(line) <= 2:
            continue
        return line[:48]
    return fallback


def extract_entertainment(lines: list[str], package: str, media_type: str) -> dict[str, str]:
    title = meaningful_title(
        [
            line
            for line in lines
            if not contains_any(line, ["首页", "发现", "我的", "搜索", "推荐", "VIP", "下载", "投屏", "评论", "关注"])
        ],
        app_name(package),
    )
    activity = {
        "music": "在听歌/浏览音乐",
        "anime": "在看动漫/番剧",
        "video": "在看剧/视频",
        "short_video": "在刷短视频",
        "live": "在看直播",
    }.get(media_type, "娱乐浏览")
    episode = ""
    for line in lines:
        if contains_any(line, ["第", "集", "更新至", "选集"]):
            episode = line[:40]
            break
    return {
        "media_type": media_type,
        "title": title,
        "artist_or_author": "",
        "episode": episode,
        "activity": activity,
        "platform": app_name(package),
    }


def parse_key_values(lines: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        m = re.match(r"^([A-Za-z][A-Za-z ]{1,24}|[\u4e00-\u9fff]{2,10})[:：]\s*(.+)$", line)
        if m:
            result[m.group(1).strip()] = m.group(2).strip()
    joined = "\n".join(lines)
    patterns = {
        "姓名": r"姓名\s*([\u4e00-\u9fffA-Za-z*]{1,12})",
        "性别": r"性别\s*([\u4e00-\u9fffA-Za-z]{1,4})",
        "出生": r"出生\s*([0-9 年月日.-]{6,20})",
        "住址": r"住址\s*([^\n]{4,80})",
        "公民身份号码": r"公民身份号码\s*([0-9Xx*]{8,24})",
        "签发机关": r"签发机关\s*([^\n]{4,40})",
        "有效期限": r"有效期限\s*([0-9.\-年月日 至长期]{6,40})",
        "订单编号": r"订单编号\s*[:：]?\s*([A-Z0-9]{6,30})",
        "取票号": r"取票号\s*[:：]?\s*([A-Z0-9]{4,30})",
        "公司名称": r"公司名称[:：]?\s*([^\n]{4,60})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, joined, re.I)
        if match and key not in result:
            result[key] = match.group(1).strip()
    return result


def extract_note(lines: list[str], sub_scene: str) -> dict[str, Any]:
    key_points = [line for line in lines if not is_noise_line(line)][:8]
    if sub_scene == "empty_or_visual_only":
        return {"summary": "截图中没有可用 OCR 文本，需要结合图片内容判断。", "key_points": [], "key_values": {}}
    summary = "；".join(key_points[:3])
    if len(summary) > 160:
        summary = summary[:157] + "..."
    return {
        "summary": summary,
        "key_points": key_points,
        "key_values": parse_key_values(lines),
    }


def extract_shopping(lines: list[str], package: str) -> dict[str, str]:
    amount_candidates = extract_amount_candidates(lines)
    status = ""
    for word in ["待付款", "待收货", "已完成", "完成", "交易关闭", "退款成功", "正在出库", "派送中", "待使用"]:
        if any(word in line for line in lines):
            status = word
            break
    product = ""
    for line in lines:
        if contains_any(line, ["¥", "￥", "商品", "旗舰", "自营", "数量", "型号"]) and not contains_any(line, ["实付款", "支付方式"]):
            product = line[:60]
            break
    return {
        "platform": app_name(package),
        "status": status,
        "product": product,
        "amount": amount_candidates[0]["amount"] if amount_candidates else "",
        "order_no": extract_order_no(lines),
    }


def build_summary(meta: dict[str, Any], entities: dict[str, Any], lines: list[str]) -> str:
    scene = meta["scene"]
    if scene == "记账":
        bill = entities["bill"]
        amount = bill.get("amount") or "金额未识别"
        merchant = bill.get("merchant") or app_name(bill.get("source_app", "")) or "交易"
        return f"识别到{merchant}账单，金额 {amount}，类型 {bill.get('mode', '')}。"
    if scene == "理财":
        inv = entities["investment"]
        total = inv.get("total_assets") or "未识别"
        return f"识别到理财/持仓页面，总资产 {total}，持仓 {len(inv.get('holdings', []))} 项。"
    if scene == "待办":
        todos = entities["todos"]
        if todos:
            return f"识别到待办：{todos[0].get('title', '')}。"
        return "识别到可能需要跟进的聊天或物流信息。"
    if scene == "娱乐":
        ent = entities["entertainment"]
        return f"识别到娱乐活动：{ent.get('activity', '')}，平台 {ent.get('platform', '')}。"
    if scene == "购物":
        shop = entities.get("shopping", {})
        return f"识别到购物/订单页面，平台 {shop.get('platform', '')}，状态 {shop.get('status', '')}。"
    note = entities["note"]
    if note.get("summary"):
        return f"提取屏幕重点：{note['summary']}"
    return f"识别到 {len(lines)} 行 OCR 文本。"


def build_answer(row: dict[str, Any], mask_sensitive: bool = False) -> dict[str, Any]:
    image = row.get("image", "")
    package = extract_package(image)
    screen_time = extract_screen_time(image)
    lines_meta = sorted_ocr_lines(row)
    lines = plain_lines(lines_meta)
    extract_lines = plain_lines(lines_meta, keep_noise=True)
    meta = classify_scene(lines, package)
    entities: dict[str, Any] = {
        "bill": {},
        "investment": {},
        "todos": [],
        "entertainment": {},
        "shopping": {},
        "note": {},
    }
    if meta["intent"] == "bill_record":
        entities["bill"] = extract_bill(extract_lines, package, screen_time)
    elif meta["intent"] == "investment_import":
        entities["investment"] = extract_investment(extract_lines, package)
    elif meta["intent"] == "chat_todo":
        entities["todos"] = extract_todos(extract_lines, meta["sub_scene"])
    elif meta["intent"] == "entertainment_activity":
        entities["entertainment"] = extract_entertainment(lines, package, meta.get("media_type", "other"))
    elif meta["intent"] == "shopping_activity":
        entities["shopping"] = extract_shopping(extract_lines, package)
    if meta["intent"] in {"note_extract", "screen_summary"}:
        entities["note"] = extract_note(lines, meta["sub_scene"])
    elif meta["scene"] in {"政务", "健康", "记事", "出行"}:
        entities["note"] = extract_note(lines, meta["sub_scene"])

    avg_conf = 0.0
    if lines_meta:
        avg_conf = round(sum(float(x.get("conf", 0.0)) for x in lines_meta) / len(lines_meta), 4)
    answer = {
        "schema_version": "zemo_screen_ocr_understanding_v1",
        "scene": meta["scene"],
        "intent": meta["intent"],
        "kind": meta["kind"],
        "sub_scene": meta["sub_scene"],
        "action": meta["action"],
        "title": meta["title"],
        "summary": build_summary(meta, entities, lines),
        "source_app_package": package,
        "source_app_name": app_name(package),
        "screen_time": screen_time,
        "needs_ocr_text": True,
        "confidence": round(float(meta["confidence"]), 3),
        "ocr_quality": {
            "line_count": len(lines_meta),
            "text_line_count": len(lines),
            "avg_conf": avg_conf,
            "width": int(row.get("w", 0) or 0),
            "height": int(row.get("h", 0) or 0),
        },
        "entities": entities,
        "field_schema": FIELD_SCHEMA.get(meta["intent"], {}),
        "raw_evidence": meta["evidence"] + lines[:8],
        "requires_review": bool(meta["confidence"] < 0.7 or len(lines) == 0),
    }
    if mask_sensitive:
        answer = mask_sensitive_obj(answer)
    return answer


def mask_sensitive_text(value: str) -> str:
    def mask_id(m: re.Match[str]) -> str:
        s = m.group(0)
        return s[:6] + "*" * max(4, len(s) - 10) + s[-4:]

    def mask_phone(m: re.Match[str]) -> str:
        s = m.group(0)
        return s[:3] + "****" + s[-4:]

    def mask_long(m: re.Match[str]) -> str:
        s = m.group(0)
        if re.match(r"20\d{6,}", s):
            return s
        return s[:4] + "*" * max(4, len(s) - 8) + s[-4:]

    value = re.sub(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0\d|1[0-2])(?:[0-2]\d|3[01])\d{3}[0-9Xx]\b", mask_id, value)
    value = re.sub(r"\b1[3-9]\d{9}\b", mask_phone, value)
    value = re.sub(r"\b\d{13,19}\b", mask_long, value)
    return value


def mask_sensitive_obj(value: Any) -> Any:
    if isinstance(value, str):
        return mask_sensitive_text(value)
    if isinstance(value, list):
        return [mask_sensitive_obj(v) for v in value]
    if isinstance(value, dict):
        return {k: mask_sensitive_obj(v) for k, v in value.items()}
    return value


def build_prompt(row: dict[str, Any], answer: dict[str, Any], variant: int, mask_sensitive: bool = False) -> str:
    image = row.get("image", "")
    package = extract_package(image)
    screen_time = extract_screen_time(image)
    lines_meta = sorted_ocr_lines(row)
    ocr_text = ocr_meta_text(lines_meta)
    visible = plain_text(lines_meta, keep_noise=False, limit=4000)
    if mask_sensitive:
        ocr_text = mask_sensitive_text(ocr_text)
        visible = mask_sensitive_text(visible)

    contract = (
        "只输出严格 JSON，不要 Markdown。字段固定：schema_version, scene, intent, kind, sub_scene, action, title, "
        "summary, source_app_package, source_app_name, screen_time, needs_ocr_text, confidence, ocr_quality, entities, field_schema, raw_evidence, requires_review。"
    )
    enums = (
        f"scene 只能取 {SCENES}；intent 只能取 {INTENTS}；kind 只能取 {KINDS}；action 只能取 {ACTIONS}。"
        "金额保留正负号；账单要提金额/商户/方式/时间/类型；理财要提基金/股票持仓；待办要提快递单号、取件码、聊天行动项；"
        "娱乐要判断听歌、看动漫/视频、刷短视频；记事要提重点和摘要。没有把握的字段留空，不要编造。"
    )
    if variant % 3 == 0:
        return "\n".join(
            [
                "你是 ZeMo 手机截图 OCR 理解模型。",
                contract,
                enums,
                f"source_app_package={package}",
                f"screen_time={screen_time}",
                "OCR 行（含 bbox）：",
                ocr_text,
            ]
        )
    if variant % 3 == 1:
        return "\n".join(
            [
                "任务：把手机截屏 OCR 转成 ZeMo 规范屏幕事件 JSON。",
                contract,
                "优先判断：娱乐、记账、理财、待办、记事、购物、出行、政务、健康、其他。",
                f"包名：{package or 'unknown'}；截图时间：{screen_time or 'unknown'}",
                "可见文字：",
                visible,
            ]
        )
    compact_lines = [
        {"t": item["text"], "b": item["bbox"], "c": item["conf"]}
        for item in lines_meta[:80]
    ]
    if mask_sensitive:
        compact_lines = mask_sensitive_obj(compact_lines)
    return "\n".join(
        [
            "根据 OCR lines 判断当前场景并抽取结构化字段，输出统一 JSON。",
            enums,
            f"metadata={compact_json({'package': package, 'screen_time': screen_time, 'image': str(Path(image).name)})}",
            f"ocr_lines={compact_json(compact_lines)}",
        ]
    )


def should_skip_path(path_value: str) -> bool:
    return any(part.lower() == "unpackage" for part in Path(path_value).parts)


def build_rows(
    ocr_rows: list[dict[str, Any]],
    ocr_jsonl: Path,
    prompt_variants: int,
    dedupe: bool,
    mask_sensitive: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    seen = set()
    skipped_unpackage = 0
    duplicate_rows = 0
    for source_index, row in enumerate(ocr_rows):
        image = row.get("image", "")
        if should_skip_path(image):
            skipped_unpackage += 1
            continue
        lines_meta = sorted_ocr_lines(row)
        text_for_hash = plain_text(lines_meta, keep_noise=True, limit=12000)
        dedupe_key = stable_hash(f"{extract_package(image)}\n{text_for_hash}")
        if dedupe and dedupe_key in seen:
            duplicate_rows += 1
            continue
        seen.add(dedupe_key)
        answer = build_answer(row, mask_sensitive=mask_sensitive)
        for variant in range(max(1, prompt_variants)):
            prompt = build_prompt(row, answer, variant, mask_sensitive=mask_sensitive)
            if mask_sensitive:
                image_value = mask_sensitive_text(image)
            else:
                image_value = image
            item_id = stable_hash(f"{source_index}:{variant}:{image}:{text_for_hash}", 20)
            item = {
                "id": item_id,
                "source": "zemo_real_ppocrv6_ocr",
                "task": "screen_ocr_to_zemo_json",
                "ocr_jsonl": str(ocr_jsonl),
                "source_index": source_index,
                "source_file": image_value,
                "image_path": image_value,
                "source_app_package": answer["source_app_package"],
                "screen_time": answer["screen_time"],
                "scene": answer["scene"],
                "intent": answer["intent"],
                "kind": answer["kind"],
                "sub_scene": answer["sub_scene"],
                "prompt_variant": variant,
                "instruction": prompt,
                "output": compact_json(answer),
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": compact_json(answer)},
                ],
                "answer": answer,
                "ocr": {
                    "format": row.get("format", "zemo_ocr_lines_v2"),
                    "width": int(row.get("w", 0) or 0),
                    "height": int(row.get("h", 0) or 0),
                    "ms": int(row.get("ms", 0) or 0),
                    "lines": mask_sensitive_obj(lines_meta) if mask_sensitive else lines_meta,
                },
            }
            output_rows.append(item)
    report = {
        "raw_rows": len(ocr_rows),
        "unique_source_rows": len(seen),
        "written_rows": len(output_rows),
        "prompt_variants": max(1, prompt_variants),
        "dedupe": dedupe,
        "duplicate_source_rows_skipped": duplicate_rows,
        "skipped_unpackage_rows": skipped_unpackage,
        "mask_sensitive": mask_sensitive,
    }
    return output_rows, report


def write_parquet(rows: list[dict[str, Any]], path: Path) -> str:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        return f"skip parquet: {exc}"

    image_bytes = []
    conversations = []
    source_files = []
    scenes = []
    intents = []
    outputs = []
    for row in rows:
        raw_path = row.get("image_path", "")
        img = Path(raw_path)
        if raw_path and img.exists() and not should_skip_path(raw_path):
            try:
                image_bytes.append(img.read_bytes())
            except OSError:
                image_bytes.append(b"")
        else:
            image_bytes.append(b"")
        conversations.append(compact_json(row["conversations"]))
        source_files.append(row.get("source_file", ""))
        scenes.append(row.get("scene", ""))
        intents.append(row.get("intent", ""))
        outputs.append(row.get("output", ""))
    table = pa.Table.from_pydict(
        {
            "image_bytes": image_bytes,
            "conversations": conversations,
            "source_file": source_files,
            "scene": scenes,
            "intent": intents,
            "output": outputs,
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
    target = dataset_dir / "zemo_real_ocr_screen_pretrain.parquet"
    shutil.copy2(parquet_path, target)
    return str(target)


def build_report(rows: list[dict[str, Any]], base_report: dict[str, Any], out_jsonl: Path, out_parquet_status: str) -> dict[str, Any]:
    scene_counts = Counter(row["scene"] for row in rows)
    intent_counts = Counter(row["intent"] for row in rows)
    sub_scene_counts = Counter(row["sub_scene"] for row in rows)
    package_counts = Counter(row["source_app_package"] or "(no_pkg)" for row in rows)
    samples = []
    for row in rows[:12]:
        samples.append(
            {
                "id": row["id"],
                "source_file": Path(row["source_file"]).name,
                "scene": row["scene"],
                "intent": row["intent"],
                "sub_scene": row["sub_scene"],
                "summary": row["answer"].get("summary", ""),
            }
        )
    report = dict(base_report)
    report.update(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "out_jsonl": str(out_jsonl),
            "out_parquet": out_parquet_status,
            "scene_counts": dict(scene_counts),
            "intent_counts": dict(intent_counts),
            "sub_scene_counts": dict(sub_scene_counts),
            "top_packages": dict(package_counts.most_common(30)),
            "output_schema": {
                "scene": SCENES,
                "intent": INTENTS,
                "kind": KINDS,
                "action": ACTIONS,
            },
            "samples": samples,
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export real ZeMo OCR rows to pretrain/SFT JSONL.")
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR)
    parser.add_argument("--ocr-jsonl", type=Path, default=None)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--out-report", type=Path, default=DEFAULT_OUT_REPORT)
    parser.add_argument("--out-parquet", type=Path, default=DEFAULT_OUT_PARQUET)
    parser.add_argument("--minimind-o-root", type=Path, default=DEFAULT_MINIMIND_O_ROOT)
    parser.add_argument("--prompt-variants", type=int, default=3)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--mask-sensitive", action="store_true")
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--copy-to-minimind-o", action="store_true")
    args = parser.parse_args()

    ocr_jsonl = args.ocr_jsonl or (args.ocr_dir / "ocr_results.jsonl")
    ocr_jsonl = ocr_jsonl.resolve()
    out_jsonl = args.out_jsonl.resolve()
    out_report = args.out_report.resolve()
    out_parquet = args.out_parquet.resolve()

    if should_skip_path(str(ocr_jsonl)):
        raise ValueError("refuse to read unpackage path")
    if not ocr_jsonl.exists():
        raise FileNotFoundError(f"OCR jsonl not found: {ocr_jsonl}")

    ocr_rows = load_jsonl(ocr_jsonl)
    rows, base_report = build_rows(
        ocr_rows,
        ocr_jsonl=ocr_jsonl,
        prompt_variants=args.prompt_variants,
        dedupe=not args.keep_duplicates,
        mask_sensitive=args.mask_sensitive,
    )
    write_jsonl(rows, out_jsonl)

    parquet_status = ""
    if not args.no_parquet:
        parquet_status = write_parquet(rows, out_parquet)
    copied = ""
    if args.copy_to_minimind_o and parquet_status == str(out_parquet):
        copied = copy_to_minimind_o(out_parquet, args.minimind_o_root.resolve())

    report = build_report(rows, base_report, out_jsonl, parquet_status)
    if copied:
        report["copied_to_minimind_o"] = copied
    write_json(out_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
