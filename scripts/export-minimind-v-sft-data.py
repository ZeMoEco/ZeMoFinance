import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "model" / "data"
PROCESSED_ROOT = DATA_ROOT / "processed"


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


def main():
    source = PROCESSED_ROOT / "screen_action_sft.jsonl"
    rows = load_jsonl(source)
    out_jsonl = PROCESSED_ROOT / "minimind_v_screen_sft.jsonl"
    out_parquet = PROCESSED_ROOT / "minimind_v_screen_sft.parquet"

    converted = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            image_rel = row.get("image", "")
            image_path = DATA_ROOT / image_rel
            if not image_rel or not image_path.exists():
                continue
            conversations = [
                {
                    "role": "user",
                    "content": "<|image_pad|>\n" + row.get("instruction", ""),
                },
                {
                    "role": "assistant",
                    "content": row.get("output", ""),
                },
            ]
            item = {
                "image": image_rel,
                "image_path": str(image_path),
                "conversations": conversations,
                "source": row.get("source", ""),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            converted.append(item)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        image_bytes = []
        conversations = []
        for item in converted:
            image_bytes.append(Path(item["image_path"]).read_bytes())
            conversations.append(json.dumps(item["conversations"], ensure_ascii=False))
        table = pa.Table.from_pydict({
            "image_bytes": image_bytes,
            "conversations": conversations,
        })
        pq.write_table(table, out_parquet)
        parquet_status = str(out_parquet)
    except Exception as e:
        parquet_status = f"skip parquet: {e}"

    print(json.dumps({
        "rows": len(converted),
        "jsonl": str(out_jsonl),
        "parquet": parquet_status,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
