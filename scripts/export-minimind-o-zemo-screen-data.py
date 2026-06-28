import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = ROOT / "model" / "data" / "phone"
PROCESSED_DIR = ROOT / "model" / "data" / "processed"
DEFAULT_JSONL = PROCESSED_DIR / "minimind_o_zemo_screen_i2t.jsonl"
DEFAULT_PARQUET = PROCESSED_DIR / "minimind_o_zemo_screen_i2t.parquet"
DEFAULT_MINIMIND_O_ROOT = ROOT / "model" / "minimind-o"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


APP_RULES = [
    (["com.tencent.mm", "com.tencent.wechat", "com.ohos.mms"], "chat_todo", "chat", "extract_todos", "聊天待办"),
    (["com.hexin", "com.tdx", "com.app.yangjibao", "com.shenlan.cdr", "com.ai.obc.cbn"], "investment_import", "investment", "import_investment", "基金/证券持仓"),
    (["com.alipay", "alipay", "cmbchina", "cmb.pb", "com.eg.android.alipaygphone", "palmhall", "mc10086", "delivery.aggregator"], "bill_record", "finance", "create_transaction", "支付/账单"),
    (["qiyi", "videohm", "danmaku", "bili", "ugc.aweme"], "entertainment_activity", "entertainment_video", "record_activity", "视频娱乐"),
    (["qqmusic", "luna.hm.music"], "entertainment_activity", "entertainment_music", "record_activity", "音乐娱乐"),
    (["taobao", "jingdong", "jd.hm.mall", "xunmeng", "pinduoduo", "idlefish", "xiaomi.shop", "alibaba.wireless", "qianniu"], "shopping_activity", "shopping_browse", "record_activity", "购物浏览"),
    (["tantan"], "entertainment_activity", "social_activity", "record_activity", "社交娱乐"),
    (["microsoft.emmx"], "screen_summary", "web_browse", "summarize_screen", "网页浏览"),
    (["gallery", "youavideo"], "screen_summary", "media_browse", "summarize_screen", "相册/媒体"),
    (["tongcheng"], "screen_summary", "travel", "summarize_screen", "出行服务"),
    (["zemo", "obsidian"], "screen_summary", "productivity", "summarize_screen", "效率工具"),
]


SCHEMAS = {
    "bill_record": {
        "amount": "",
        "mode": "expense|income|transfer|refund",
        "merchant": "",
        "pay_method": "",
        "transaction_time": "",
        "note": "",
    },
    "investment_import": {
        "holdings": [
            {
                "name": "",
                "code": "",
                "market_value": "",
                "profit": "",
                "profit_rate": "",
                "units": "",
                "nav": "",
            }
        ]
    },
    "chat_todo": {
        "todos": [
            {
                "title": "",
                "start_at": "",
                "end_at": "",
                "priority": "",
                "source_text": "",
            }
        ]
    },
}


def extract_package(path: Path) -> str:
    stem = re.sub(r"\(\d+\)$", "", path.stem)
    match = re.search(r"_([A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9][A-Za-z0-9_-]*)+)$", stem)
    return match.group(1) if match else ""


def extract_screen_time(path: Path) -> str:
    stem = path.stem
    match = re.search(r"Screenshot_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", stem, re.I)
    if match:
        parts = [int(v) for v in match.groups()]
        return datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]).isoformat()
    match = re.search(r"screenshot_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", stem, re.I)
    if match:
        parts = [int(v) for v in match.groups()]
        return datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]).isoformat()
    return ""


def classify_package(package: str) -> dict:
    key = package.lower()
    for needles, intent, activity_type, action, title in APP_RULES:
        if any(needle in key for needle in needles):
            return {
                "intent": intent,
                "activity_type": activity_type,
                "action": action,
                "title": title,
            }
    return {
        "intent": "screen_summary",
        "activity_type": "unknown_screen",
        "action": "summarize_screen",
        "title": "未知屏幕/场景",
    }


def build_prompt() -> str:
    intents = [
        "bill_record",
        "investment_import",
        "chat_todo",
        "entertainment_activity",
        "shopping_activity",
        "screen_summary",
        "phone_action",
    ]
    return (
        "<image>\n"
        "你是 ZeMo 手机屏幕意图模型。根据截图判断屏幕在做什么，并输出严格 JSON。"
        "需要同时判断是否要读取屏幕文字、要提取哪些字段，以及 ZeMo 应执行什么动作。"
        f"intent 只能取: {', '.join(intents)}。"
        "如果是账单/基金/聊天，请标记 needs_ocr_text=true 并给出字段 schema；"
        "如果是视频、音乐、购物、网页等活动，请输出 record_activity。"
    )


def build_answer(path: Path) -> dict:
    package = extract_package(path)
    rule = classify_package(package)
    intent = rule["intent"]
    needs_ocr = intent in {"bill_record", "investment_import", "chat_todo"}
    return {
        "intent": intent,
        "activity_type": rule["activity_type"],
        "action": rule["action"],
        "screen_title": rule["title"],
        "source_app_package": package,
        "screen_time": extract_screen_time(path),
        "needs_ocr_text": needs_ocr,
        "visible_text": "",
        "entities": {},
        "extraction_schema": SCHEMAS.get(intent, {}),
        "confidence_hint": "weak_label_from_filename_package",
    }


def iter_images(image_dir: Path, limit: int) -> list[Path]:
    images = [
        p
        for p in image_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
        and "unpackage" not in {part.lower() for part in p.parts}
    ]
    images.sort(key=lambda p: str(p.relative_to(image_dir)).lower())
    if limit > 0:
        images = images[:limit]
    return images


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(rows: list[dict], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("需要 pyarrow 才能生成 MiniMind-O parquet: python -m pip install pyarrow") from exc

    image_bytes = []
    conversations = []
    source_files = []
    packages = []
    intents = []
    for row in rows:
        image_path = Path(row["image_path"])
        image_bytes.append(image_path.read_bytes())
        conversations.append(json.dumps(row["conversations"], ensure_ascii=False))
        source_files.append(row["source_file"])
        packages.append(row["source_app_package"])
        intents.append(row["intent"])
    table = pa.Table.from_pydict(
        {
            "image_bytes": image_bytes,
            "conversations": conversations,
            "source_file": source_files,
            "source_app_package": packages,
            "intent": intents,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def copy_to_minimind_o(parquet_path: Path, minimind_o_root: Path) -> Path:
    dataset_dir = minimind_o_root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target = dataset_dir / "zemo_screen_i2t.parquet"
    shutil.copy2(parquet_path, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ZeMo phone screenshots to MiniMind-O I2T SFT parquet.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--out-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--minimind-o-root", type=Path, default=DEFAULT_MINIMIND_O_ROOT)
    parser.add_argument("--copy-to-minimind-o", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    args.image_dir = args.image_dir.resolve()
    args.out_jsonl = args.out_jsonl.resolve()
    args.out_parquet = args.out_parquet.resolve()
    args.minimind_o_root = args.minimind_o_root.resolve()

    if not args.image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {args.image_dir}")

    rows = []
    for image_path in iter_images(args.image_dir, args.limit):
        answer = build_answer(image_path)
        conversations = [
            {"role": "user", "content": build_prompt()},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
        rows.append(
            {
                "source": "zemo_phone_weak_label",
                "source_file": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                "image_path": str(image_path),
                "source_app_package": answer["source_app_package"],
                "intent": answer["intent"],
                "activity_type": answer["activity_type"],
                "action": answer["action"],
                "conversations": conversations,
                "answer": answer,
            }
        )

    write_jsonl(rows, args.out_jsonl)
    write_parquet(rows, args.out_parquet)
    copied = ""
    if args.copy_to_minimind_o:
        copied = str(copy_to_minimind_o(args.out_parquet, args.minimind_o_root))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["intent"]] = counts.get(row["intent"], 0) + 1

    print(
        json.dumps(
            {
                "rows": len(rows),
                "jsonl": str(args.out_jsonl),
                "parquet": str(args.out_parquet),
                "copied_to": copied,
                "intent_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
