import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINIMIND_ROOT = ROOT / "model" / "minimind-o"
DEFAULT_MODEL_DIR = DEFAULT_MINIMIND_ROOT / "zemo-screen-omni-final"
DEFAULT_VISION_DIR = DEFAULT_MINIMIND_ROOT / "model" / "siglip2-base-p32-256-ve"
DEFAULT_OUTPUT = DEFAULT_MINIMIND_ROOT / "mindspore_lite" / "zemo_screen_minimind_o_prefill.onnx"


class MiniMindOPrefill(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, image_start: int, image_len: int):
        super().__init__()
        self.model = model
        self.image_start = int(image_start)
        self.image_len = int(image_len)

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[1]
        thinker = self.model.thinker
        hidden_states = thinker.dropout(thinker.embed_tokens(input_ids))

        image_embeds = self.model.vision_encoder(pixel_values=pixel_values).last_hidden_state
        image_embeds = self.model.vision_proj(image_embeds).to(hidden_states.dtype)
        image_embeds = image_embeds[:, :self.image_len, :]
        hidden_states = torch.cat(
            (
                hidden_states[:, :self.image_start, :],
                image_embeds,
                hidden_states[:, self.image_start + self.image_len:, :],
            ),
            dim=1,
        )

        position_embeddings = (
            thinker.freqs_cos[:seq_length],
            thinker.freqs_sin[:seq_length],
        )
        for layer in thinker.layers:
            hidden_states, _present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=None,
                use_cache=False,
                attention_mask=None,
            )
        hidden_states = thinker.norm(hidden_states)
        return thinker.lm_head(hidden_states)


def build_input_ids(tokenizer, seq_len: int, prompt: str, image_token: str, image_token_len: int, pad_token_id: int) -> tuple[torch.Tensor, list[int]]:
    content = prompt.rstrip() + "\n\n" + image_token * image_token_len
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        open_thinking=False,
    )
    ids = tokenizer(text).data["input_ids"]
    if len(ids) > seq_len:
        raise ValueError(f"prompt token length {len(ids)} exceeds seq-len {seq_len}")
    prompt_ids = list(ids)
    ids.extend([pad_token_id] * (seq_len - len(ids)))
    return torch.tensor([ids], dtype=torch.long), prompt_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ZeMo MiniMind-O image prefill graph to ONNX.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--vision-dir", type=Path, default=DEFAULT_VISION_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt", type=str, default="看图回JSON action=create_transaction/create_todos/import_investment/record_activity/none, reason说明, transaction含amount/type/note, todos含title。")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    vision_dir = args.vision_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(DEFAULT_MINIMIND_ROOT.resolve()))
    from model.model_omni import MiniMindOmni

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True)
    model.audio_encoder = None
    model.audio_processor = None
    model.vision_encoder, model.vision_processor = MiniMindOmni.load_vision(str(vision_dir))
    model.eval().to(device)

    image_token_id = int(model.config.image_ids[0])
    image_token = str(model.config.image_special_token)
    image_token_len = int(model.config.image_token_len)
    pad_token_id = int(getattr(model.config, "pad_token_id", 0) or 0)
    input_ids, prompt_ids = build_input_ids(tokenizer, args.seq_len, args.prompt, image_token, image_token_len, pad_token_id)
    try:
        image_start = prompt_ids.index(image_token_id)
    except ValueError as exc:
        raise ValueError("prompt_ids does not contain image token") from exc
    if prompt_ids[image_start:image_start + image_token_len] != [image_token_id] * image_token_len:
        raise ValueError("image tokens must be contiguous for fixed Lite export")
    input_ids = input_ids.to(device)
    # Use a non-zero dummy image. MiniMind-O treats all-zero pixels as "no image",
    # which lets ONNX export prune pixel_values from the graph.
    pixel_values = torch.full((1, 3, args.image_size, args.image_size), 0.5, dtype=torch.float32, device=device)
    wrapper = MiniMindOPrefill(model, image_start=image_start, image_len=image_token_len).eval().to(device)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, pixel_values),
            str(output),
            input_names=["input_ids", "pixel_values"],
            output_names=["logits"],
            opset_version=args.opset,
            do_constant_folding=False,
            dynamic_axes=None,
            dynamo=False,
        )
    manifest = {
        "format": "zemo_minimind_o_ms_prefill_v1",
        "model": "MiniMind-O ZeMo screen action",
        "seq_len": args.seq_len,
        "image_size": args.image_size,
        "image_token_id": image_token_id,
        "image_token_len": image_token_len,
        "pad_token_id": pad_token_id,
        "eos_token_id": int(model.config.eos_token_id),
        "input_names": ["input_ids", "pixel_values"],
        "output_names": ["logits"],
        "onnx_input_shapes": {
            "input_ids": [1, args.seq_len],
            "pixel_values": [1, 3, args.image_size, args.image_size],
        },
        "mindspore_lite_input_shapes": {
            "input_ids": [1, args.seq_len],
            "pixel_values": [1, args.image_size, args.image_size, 3],
        },
        "runtime_input_dtypes": {
            "input_ids": "int32",
            "pixel_values": "float32",
        },
        "prompt": args.prompt,
        "prompt_token_len": len(prompt_ids),
        "prompt_ids": prompt_ids,
        "image_token_start": image_start,
        "pixel_mean": [0.5, 0.5, 0.5],
        "pixel_std": [0.5, 0.5, 0.5],
        "notes": [
            "This graph is a fixed-shape prefill/generation-step graph.",
            "Runtime must keep image token positions identical to prompt_ids.",
            "The Lite graph uses a fixed single-image path to avoid dynamic NonZero/GatherND image-token injection."
        ],
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ONNX exported: {output}")


if __name__ == "__main__":
    main()
