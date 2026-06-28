from pathlib import Path

import torch


def load_quantized_state_dict(path: str, dtype=torch.float16):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != "zemo_minimind_v_int8_state_v1":
        return payload
    out = {}
    state = payload["state"]
    for key, item in state.items():
        kind = item.get("kind")
        if kind == "int8":
            out[key] = (item["value"].float() * float(item["scale"])).to(dtype)
        elif kind == "raw":
            out[key] = item["value"]
        elif kind == "object":
            out[key] = item["value"]
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    state = load_quantized_state_dict(args.path)
    print(f"loaded tensors={len(state)}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, out)
        print(out)
