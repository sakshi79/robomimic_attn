"""
SAM3 mask server. Run by the `sam3` conda env's Python (Python 3.10+, transformers
from main, huggingface_hub 1.x). Spoken to over stdin/stdout from `Sam3Masker` in
robomimic.models.sam3 (which runs under the older robodiff env, Python 3.9).

Wire protocol: length-prefixed pickle.
  - Each message starts with 8 little-endian bytes = uint64 payload length N.
  - Then N bytes of pickled dict.

Client → Server messages:
  {"cmd": "mask",
   "image": np.ndarray uint8 (H, W, 3),
   "prompts": list[str],
   "threshold": float, "mask_threshold": float}
  {"cmd": "shutdown"}

Server → Client messages:
  {"status": "ready"}       # once after model load
  {"mask": np.ndarray bool (H, W)}
  {"error": str}            # on exception

stdout is used ONLY for protocol bytes. Any library prints go to stderr.
"""
import argparse
import os
import pickle
import struct
import sys
import traceback


# ── Move protocol fd off of stdout BEFORE any library imports can print ─────
# Duplicate fd 1 (stdout, which the parent's subprocess pipe is attached to)
# onto a fresh fd, then redirect fd 1 → fd 2 (stderr). Library prints to stdout
# now go to stderr instead of corrupting the protocol stream.
_PROTO_FD = os.dup(1)
os.dup2(2, 1)
sys.stdout = os.fdopen(1, "w", buffering=1)
_PROTO_OUT = os.fdopen(_PROTO_FD, "wb", buffering=0)
_PROTO_IN = sys.stdin.buffer


import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
from transformers import Sam3Config, Sam3Model, Sam3Processor  # noqa: E402


def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def read_msg():
    hdr = _read_exact(_PROTO_IN, 8)
    if hdr is None:
        return None
    (n,) = struct.unpack("<Q", hdr)
    payload = _read_exact(_PROTO_IN, n)
    if payload is None:
        return None
    return pickle.loads(payload)


def write_msg(obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _PROTO_OUT.write(struct.pack("<Q", len(data)))
    _PROTO_OUT.write(data)
    _PROTO_OUT.flush()


def build_model(model_id: str, image_size: int, device: str):
    cfg = Sam3Config.from_pretrained(model_id)
    cfg.image_size = image_size
    model = Sam3Model.from_pretrained(model_id, config=cfg).to(device).eval()
    processor = Sam3Processor.from_pretrained(
        model_id, size={"height": image_size, "width": image_size}
    )
    return model, processor


@torch.no_grad()
def mask_one(model, processor, device, rgb_np, prompts, threshold, mask_threshold):
    H, W = rgb_np.shape[:2]
    pil = Image.fromarray(rgb_np)
    img_inputs = processor(images=pil, return_tensors="pt").to(device)
    vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

    union = np.zeros((H, W), dtype=bool)
    for prompt in prompts:
        text_inputs = processor(text=prompt, return_tensors="pt").to(device)
        outputs = model(vision_embeds=vision_embeds, **text_inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=img_inputs.get("original_sizes").tolist(),
        )[0]
        masks = results.get("masks", None)
        if masks is None or len(masks) == 0:
            # GPU-memory hygiene: drop per-iter GPU tensors before continuing,
            # otherwise the caching allocator's reserved pool grows monotonically
            # over a long eval (each iter allocates fresh `outputs`/`text_inputs`).
            del text_inputs, outputs, results, masks
            continue
        arr = masks.float().cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
        while arr.ndim > 3:
            arr = arr.squeeze(1)
        binary = arr > 0.5
        union |= binary.any(axis=0)
        # GPU-memory hygiene: free per-prompt intermediates inside the loop so
        # multi-prompt calls don't stack `outputs`/`results` on cuda:1.
        del text_inputs, outputs, results, masks, arr, binary
    # GPU-memory hygiene: free per-image intermediates so the caching allocator
    # can reuse the same blocks on the next call. Without these `del`s the locals
    # stay reachable until function exit -- fine in normal Python, but on cuda:1
    # `vision_embeds` (the largest tensor) ends up pinning a fresh block per call
    # because the next call re-allocates before the previous one drops.
    del img_inputs, vision_embeds, pil
    # Per-call torch.cuda.empty_cache() was removed: it forces a CUDA sync and
    # adds measurable per-image latency. The `del`s above are sufficient to keep
    # steady-state usage bounded -- `nvidia-smi` may still show high *reserved*
    # memory, but PyTorch will reuse it rather than allocate fresh.
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    return union


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id",   default="facebook/sam3")
    p.add_argument("--image-size", type=int, default=560)
    p.add_argument("--device",     default="cuda")
    args = p.parse_args()

    try:
        model, processor = build_model(args.model_id, args.image_size, args.device)
    except Exception as e:
        write_msg({"error": f"model load failed: {e}\n{traceback.format_exc()}"})
        return 1

    write_msg({"status": "ready"})

    while True:
        msg = read_msg()
        if msg is None:
            break
        cmd = msg.get("cmd")
        if cmd == "shutdown":
            break
        if cmd != "mask":
            write_msg({"error": f"unknown cmd: {cmd!r}"})
            continue
        try:
            mask = mask_one(
                model, processor, args.device,
                rgb_np=msg["image"],
                prompts=msg["prompts"],
                threshold=msg.get("threshold", 0.5),
                mask_threshold=msg.get("mask_threshold", 0.5),
            )
            write_msg({"mask": mask})
        except Exception as e:
            write_msg({"error": f"mask failed: {e}\n{traceback.format_exc()}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
