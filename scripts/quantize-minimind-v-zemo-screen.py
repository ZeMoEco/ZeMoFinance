import json
import zipfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MINIMIND_ROOT = ROOT / "model" / "minimind-v"
SRC = MINIMIND_ROOT / "out" / "zemo_screen_sft_768.pth"
OUT_DIR = MINIMIND_ROOT / "out"
Q_PATH = OUT_DIR / "zemo_screen_sft_768_int8.pth"
ZIP_PATH = OUT_DIR / "zemo_screen_sft_768_int8.zip"
REPORT_PATH = OUT_DIR / "zemo_screen_sft_768_int8_report.json"


def quantize_tensor(t: torch.Tensor):
    if not torch.is_floating_point(t):
        return {"kind": "raw", "dtype": str(t.dtype), "value": t.cpu()}
    x = t.float().cpu()
    max_abs = float(x.abs().max().item()) if x.numel() > 0 else 0.0
    if max_abs == 0.0:
        scale = 1.0
        q = torch.zeros_like(x, dtype=torch.int8)
    else:
        scale = max_abs / 127.0
        q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return {"kind": "int8", "shape": list(t.shape), "scale": scale, "value": q}


def main():
    if not SRC.exists():
        raise SystemExit(f"missing source weight: {SRC}")
    state = torch.load(SRC, map_location="cpu")
    q_state = {}
    for key, value in state.items():
        q_state[key] = quantize_tensor(value) if isinstance(value, torch.Tensor) else {"kind": "object", "value": value}
    torch.save({"format": "zemo_minimind_v_int8_state_v1", "state": q_state}, Q_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(Q_PATH, Q_PATH.name)
    report = {
        "source": str(SRC),
        "source_bytes": SRC.stat().st_size,
        "quantized": str(Q_PATH),
        "quantized_bytes": Q_PATH.stat().st_size,
        "zip": str(ZIP_PATH),
        "zip_bytes": ZIP_PATH.stat().st_size,
        "format": "zemo_minimind_v_int8_state_v1",
        "note": "Per-tensor symmetric int8 quantization. Use scripts/load-quantized-minimind-v.py or eval_vlm.py quantized loader patch to dequantize for inference.",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
