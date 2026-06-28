import argparse
import json
import shutil
from pathlib import Path


# Package a Layout-Aware field model for app/runtime import.
#
# Training outputs use:
#   finance_layout_field_encoder_from_base_lite.ms
#   char_vocab.json
#   label_map.json
#   model_config.json
#
# Runtime package uses layout names by default:
#   finance_layout_field_cls.ms
#   vocab.txt
#   label_map.json
#   model_config.json
#
# Add --legacy-names only when you intentionally need:
#   finance_field_cls.ms
#   vocab.txt
#   label_map.json
#
# vocab.txt format is the same as the previous model: one token per line,
# and line number is token id.

DEFAULT_MODEL_DIR = r"E:\CamXAll\ZEMO\Data\model\layout_base_run_20260621_cpu\field_from_base"
DEFAULT_OUTPUT_DIR = r"E:\CamXAll\ZEMO\Data\model\layout_base_run_20260621_cpu\runtime_legacy"


def parse_args():
    parser = argparse.ArgumentParser(description="Package layout field model into runtime-friendly files.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--legacy-names", action="store_true", help="Use finance_field_cls.ms/vocab.txt/label_map.json.")
    parser.add_argument("--no-model-config", action="store_true", help="Do not copy model_config.json.")
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_vocab_txt(char_vocab, output_path):
    by_id = {}
    for token, index in char_vocab.items():
        index = int(index)
        if index in by_id:
            raise ValueError(f"Duplicate vocab id {index}: {by_id[index]} and {token}")
        by_id[index] = token
    if not by_id:
        raise ValueError("Empty char_vocab.json")
    max_id = max(by_id)
    missing = [index for index in range(max_id + 1) if index not in by_id]
    if missing:
        raise ValueError(f"Vocab id is not continuous, missing first id: {missing[0]}")
    lines = [by_id[index] for index in range(max_id + 1)]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ms_candidates = [
        model_dir / "finance_layout_field_encoder_from_base_lite.ms",
        model_dir / "finance_layout_field_encoder_scratch_lite.ms",
        model_dir / "finance_layout_field_cls.ms",
    ]
    ms_path = next((path for path in ms_candidates if path.exists()), None)
    if ms_path is None:
        raise FileNotFoundError(f"No lite .ms found under {model_dir}")

    char_vocab_path = model_dir / "char_vocab.json"
    label_map_path = model_dir / "label_map.json"
    config_path = model_dir / "model_config.json"
    if not char_vocab_path.exists():
        raise FileNotFoundError(char_vocab_path)
    if not label_map_path.exists():
        raise FileNotFoundError(label_map_path)

    model_name = "finance_field_cls.ms" if args.legacy_names else "finance_layout_field_cls.ms"
    shutil.copyfile(ms_path, output_dir / model_name)
    shutil.copyfile(label_map_path, output_dir / "label_map.json")
    write_vocab_txt(read_json(char_vocab_path), output_dir / "vocab.txt")
    if not args.no_model_config and config_path.exists():
        shutil.copyfile(config_path, output_dir / "model_config.json")

    manifest = {
        "model": model_name,
        "vocab": "vocab.txt",
        "labels": "label_map.json",
        "model_config": "" if args.no_model_config else ("model_config.json" if config_path.exists() else ""),
        "source_model": str(ms_path),
        "layout_inputs": ["input_ids", "text_mask", "bbox", "row_ids", "col_ids", "item_mask"],
        "fixed_shape": {
            "max_items": 96,
            "max_item_len": 32,
            "bbox_range": "0..1000",
        },
        "note": "This layout model is not compatible with the old text-only classifyFinanceFields runner.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Packaged runtime model: {output_dir}")
    print(f"model={model_name}")
    print("vocab=vocab.txt")
    print("labels=label_map.json")
    print(f"model_config={'omitted' if args.no_model_config else 'model_config.json'}")


if __name__ == "__main__":
    main()
