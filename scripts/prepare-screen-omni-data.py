import csv
import json
import re
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "model" / "data"
RAW_ROOT = DATA_ROOT / "raw"
IMAGE_ROOT = DATA_ROOT / "images"
PROCESSED_ROOT = DATA_ROOT / "processed"


SOURCES = {
    "screen2words": {
        "url": "https://raw.githubusercontent.com/google-research-datasets/screen2words/main/screen_summaries.csv",
        "path": RAW_ROOT / "screen2words" / "screen_summaries.csv",
        "license": "Dataset metadata from google-research-datasets/screen2words; uses Rico image ids.",
    },
    "screen_annotation_train": {
        "url": "https://raw.githubusercontent.com/google-research-datasets/screen_annotation/main/train.csv",
        "path": RAW_ROOT / "screen_annotation" / "train.csv",
        "license": "CC BY 4.0; labels reference Rico screenshot ids.",
    },
    "screen_annotation_valid": {
        "url": "https://raw.githubusercontent.com/google-research-datasets/screen_annotation/main/valid.csv",
        "path": RAW_ROOT / "screen_annotation" / "valid.csv",
        "license": "CC BY 4.0; labels reference Rico screenshot ids.",
    },
    "screen_annotation_test": {
        "url": "https://raw.githubusercontent.com/google-research-datasets/screen_annotation/main/test.csv",
        "path": RAW_ROOT / "screen_annotation" / "test.csv",
        "license": "CC BY 4.0; labels reference Rico screenshot ids.",
    },
}


def ensure_dirs() -> None:
    for path in [RAW_ROOT, IMAGE_ROOT, PROCESSED_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def download_url(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"skip existing {path}")
        return
    print(f"download {url} -> {path}")
    urlretrieve(url, path)


def download_public_metadata() -> None:
    for item in SOURCES.values():
        download_url(item["url"], item["path"])


def download_mobile_actions() -> Path:
    target_dir = RAW_ROOT / "mobile_actions"
    target_dir.mkdir(parents=True, exist_ok=True)
    print("download google/mobile-actions metadata")
    return Path(hf_hub_download(
        repo_id="google/mobile-actions",
        repo_type="dataset",
        filename="dataset.jsonl",
        local_dir=str(target_dir),
    ))


def download_guiact(limit_images: int) -> Path:
    target_dir = RAW_ROOT / "guiact_smartphone_test"
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["metadata.json", "samples.json", "fiftyone.yml"]:
        hf_hub_download(
            repo_id="Voxel51/guiact_smartphone_test",
            repo_type="dataset",
            filename=filename,
            local_dir=str(target_dir),
        )

    samples_path = target_dir / "samples.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))["samples"]
    picked = sorted(samples, key=lambda item: (item.get("episode", ""), int(item.get("step", 0))))[:limit_images]
    image_dir = IMAGE_ROOT / "guiact_smartphone_test"
    image_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(picked, 1):
        rel = sample["filepath"]
        out = image_dir / Path(rel).name
        if not out.exists():
            print(f"download GUIAct image {idx}/{len(picked)} {rel}")
            hf_hub_download(
                repo_id="Voxel51/guiact_smartphone_test",
                repo_type="dataset",
                filename=rel,
                local_dir=str(target_dir),
            )
            source = target_dir / rel
            if source.exists():
                out.write_bytes(source.read_bytes())
        sample["local_image"] = str(out.relative_to(DATA_ROOT)).replace("\\", "/")
    (PROCESSED_ROOT / "guiact_sample_index.json").write_text(
        json.dumps(picked, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return samples_path


def short_ui_text(sample: dict, limit: int = 24) -> str:
    detections = sample.get("ui_elements", {}).get("detections", [])
    parts = []
    for det in detections:
        label = det.get("label", "")
        text = (det.get("text", "") or "").strip()
        bbox = det.get("bounding_box", [])
        if text:
            parts.append(f"{label}:{text}")
        elif label and len(parts) < 10:
            parts.append(label)
        if len(parts) >= limit:
            break
    return " | ".join(parts)


def build_guiact_jsonl() -> int:
    index_path = PROCESSED_ROOT / "guiact_sample_index.json"
    if not index_path.exists():
        return 0
    samples = json.loads(index_path.read_text(encoding="utf-8"))
    out_path = PROCESSED_ROOT / "screen_action_sft.jsonl"
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            image = sample.get("local_image", "")
            if not image:
                continue
            question = sample.get("question", "")
            history = sample.get("structured_history", [])
            ui_text = short_ui_text(sample)
            action = sample.get("current_action", "")
            prompt = (
                "你是手机屏幕助手。根据截图、当前任务、历史操作和可见UI元素，"
                "判断下一步应该做什么。只输出JSON。\n"
                f"当前任务: {question}\n"
                f"历史操作: {json.dumps(history, ensure_ascii=False)}\n"
                f"可见UI: {ui_text}"
            )
            answer = {
                "task": question,
                "next_action": action,
                "ui_summary": ui_text,
            }
            row = {
                "source": "Voxel51/guiact_smartphone_test",
                "image": image,
                "instruction": prompt,
                "output": json.dumps(answer, ensure_ascii=False),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_screen_summary_jsonl(max_rows: int) -> int:
    path = SOURCES["screen2words"]["path"]
    if not path.exists():
        return 0
    out_path = PROCESSED_ROOT / "screen_summary_text_only.jsonl"
    count = 0
    with path.open("r", encoding="utf-8", newline="") as src, out_path.open("w", encoding="utf-8") as dst:
        reader = csv.DictReader(src)
        for row in reader:
            image_id = row.get("screenId") or row.get("screen_id") or row.get("image_id") or row.get("id") or ""
            summary = row.get("summary") or row.get("screen_summary") or row.get("description") or ""
            if not summary:
                values = [v for v in row.values() if v]
                summary = values[-1] if values else ""
            if not image_id and row:
                image_id = next(iter(row.values()))
            if not summary:
                continue
            prompt = f"请总结手机截图的用途。Rico image_id={image_id}。"
            dst.write(json.dumps({
                "source": "google-research-datasets/screen2words",
                "image_id": image_id,
                "instruction": prompt,
                "output": summary,
            }, ensure_ascii=False) + "\n")
            count += 1
            if count >= max_rows:
                break
    return count


def build_mobile_actions_jsonl(max_rows: int) -> int:
    path = RAW_ROOT / "mobile_actions" / "dataset.jsonl"
    if not path.exists():
        return 0
    out_path = PROCESSED_ROOT / "mobile_actions_tool_sft.jsonl"
    count = 0
    with path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if count >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            messages = item.get("messages", [])
            user_text = ""
            assistant_call = None
            for msg in messages:
                if msg.get("role") == "user" and msg.get("content"):
                    user_text = msg["content"]
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    assistant_call = msg["tool_calls"][0]
            if not user_text or assistant_call is None:
                continue
            fn = assistant_call.get("function", {})
            output = {
                "tool": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}"),
            }
            dst.write(json.dumps({
                "source": "google/mobile-actions",
                "instruction": f"把用户手机助手请求转换成工具调用JSON: {user_text}",
                "output": json.dumps(output, ensure_ascii=False),
                "split": item.get("metadata", "train"),
            }, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_manifest(stats: dict) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": "Only public/open datasets are downloaded. Copyrighted anime/video and private chat scraping are intentionally excluded.",
        "sources": [
            {
                "name": "Voxel51/guiact_smartphone_test",
                "use": "phone screenshot step/action training",
                "url": "https://huggingface.co/datasets/Voxel51/guiact_smartphone_test",
            },
            {
                "name": "google/mobile-actions",
                "use": "chat/natural language to Android function call",
                "url": "https://huggingface.co/datasets/google/mobile-actions",
            },
            {
                "name": "google-research-datasets/screen2words",
                "use": "mobile UI screen summarization metadata",
                "url": "https://github.com/google-research-datasets/screen2words",
            },
            {
                "name": "google-research-datasets/screen_annotation",
                "use": "UI element text/location annotation metadata",
                "url": "https://github.com/google-research-datasets/screen_annotation",
                "license": "CC BY 4.0",
            },
        ],
        "stats": stats,
    }
    (DATA_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    limit_images = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    max_text_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    download_public_metadata()
    download_mobile_actions()
    download_guiact(limit_images)
    stats = {
        "guiact_screen_action_rows": build_guiact_jsonl(),
        "screen_summary_text_rows": build_screen_summary_jsonl(max_text_rows),
        "mobile_actions_rows": build_mobile_actions_jsonl(max_text_rows),
    }
    write_manifest(stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
