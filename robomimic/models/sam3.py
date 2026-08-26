"""
Drop-in SAM3 masking preprocessor for ResNet18Conv.

SAM3 only exists on `transformers` main (>= 2025-11-19), which requires Python
>=3.10 and huggingface_hub>=1.5. Our robodiff env is locked to Python 3.9 +
huggingface_hub 0.25 (held there by `cached_download` in diffusers 0.11.1 and
the cheng-chi robosuite fork's mujoco-py dependency).

So Sam3Masker runs SAM3 out-of-process: it spawns the existing `sam3` conda env
as a subprocess, exchanging images and masks over stdin/stdout using a length-
prefixed pickle protocol.
"""
import argparse
import atexit
import os
import pickle
import struct
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


def _default_sam3_python(env_name: str = "sam3") -> str:
    """
    Locate the python executable of the `sam3` conda env (created by
    robomimic/scripts/setup_sam3_env.sh) relative to the active conda install.
    """
    conda_exe = os.environ.get("CONDA_EXE") or os.environ.get("CONDA_PYTHON_EXE")
    if conda_exe:
        conda_base = Path(conda_exe).resolve().parent.parent
        candidate = conda_base / "envs" / env_name / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return str(Path.home() / "miniforge3" / "envs" / env_name / "bin" / "python")


class Sam3Masker(nn.Module):
    """
    Drop-in SAM3 masking preprocessor for ResNet18Conv. 
    
    Usage inside ResNet18Conv.__init__:
        self.masker = Sam3Masker(text_prompt="cylinder", device=device)

    Then at the top of forward(x):
        x = self.masker(x)
    """

    _DEFAULT_MODEL_ID      = "facebook/sam3"
    _DEFAULT_SAM3_PYTHON   = _default_sam3_python()
    _DEFAULT_SERVER_SCRIPT = str(Path(__file__).resolve().parent / "sam3_server.py")

    def __init__(
        self,
        text_prompt: Union[str, List[str]] = "cylinder",
        model_id: str = None,
        image_size: int = 224,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        device: str = "cuda:1",
        sam3_python: str = None,
        server_script: str = None,
    ):
        super().__init__()

        if isinstance(text_prompt, str):
            self.text_prompts = [text_prompt]
        else:
            self.text_prompts = list(text_prompt)

        self.threshold = threshold
        self.mask_threshold = mask_threshold

        # ImageNet stats — used to un-normalise for SAM3 and re-normalise after masking
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        py  = sam3_python   or self._DEFAULT_SAM3_PYTHON
        srv = server_script or self._DEFAULT_SERVER_SCRIPT
        if not Path(py).is_file():
            raise FileNotFoundError(
                f"Sam3Masker: sam3 env python not found at '{py}'. "
                f"Run `bash robomimic/scripts/setup_sam3_env.sh` to create it, "
                f"or pass sam3_python=... to point at an existing env."
            )
        if not Path(srv).is_file():
            raise FileNotFoundError(f"Sam3Masker: server script not found at '{srv}'")

        # ── Spawn the SAM3 server subprocess ────────────────────────────────
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            [
                py, srv,
                "--model-id",   model_id or self._DEFAULT_MODEL_ID,
                "--image-size", str(image_size),
                "--device",     device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,   # let SAM3 load / progress messages flow through
            bufsize=0,
        )
        atexit.register(self._shutdown)

        ready = self._read_msg()
        if not isinstance(ready, dict) or ready.get("status") != "ready":
            raise RuntimeError(f"Sam3Masker: server failed to start: {ready}")

    # ── Wire protocol ───────────────────────────────────────────────────

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                rc = self._proc.poll()
                raise RuntimeError(
                    f"Sam3Masker: server closed stdout mid-message (returncode={rc})"
                )
            buf.extend(chunk)
        return bytes(buf)

    def _read_msg(self):
        hdr = self._read_exact(8)
        (n,) = struct.unpack("<Q", hdr)
        return pickle.loads(self._read_exact(n))

    def _write_msg(self, obj):
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        self._proc.stdin.write(struct.pack("<Q", len(data)))
        self._proc.stdin.write(data)
        self._proc.stdin.flush()

    def _request(self, msg):
        with self._lock:
            self._write_msg(msg)
            resp = self._read_msg()
        if "error" in resp:
            raise RuntimeError(f"Sam3 server: {resp['error']}")
        return resp

    # ── Internal helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def _mask_for_image(self, rgb_np: np.ndarray) -> np.ndarray:
        resp = self._request({
            "cmd":            "mask",
            "image":          rgb_np,
            "prompts":        self.text_prompts,
            "threshold":      self.threshold,
            "mask_threshold": self.mask_threshold,
        })
        return resp["mask"]

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised tensor on any device.

        Returns:
            Same shape; background pixels zeroed, still ImageNet-normalised.
        """
        mean = self.mean.to(x)
        std  = self.std.to(x)

        x_01 = (x * std + mean).clamp(0.0, 1.0)

        masked_01 = x_01.clone()
        for i in range(x.shape[0]):
            rgb_np = (x_01[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            mask   = self._mask_for_image(rgb_np)                  # (H, W) bool
            mask_t = torch.from_numpy(mask).to(x.device)
            masked_01[i] = masked_01[i] * mask_t.unsqueeze(0)      # zero background

        return (masked_01 - mean) / std

    # ── Lifecycle ───────────────────────────────────────────────────────

    def _shutdown(self):
        proc = getattr(self, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            self._write_msg({"cmd": "shutdown"})
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def __del__(self):
        self._shutdown()


# ── CLI ─────────────────────────────────────────────────────────────────────


def _run_cli(args):
    image    = Image.open(args.input_image).convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0

    x    = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_norm = (x - mean) / std

    prompts = [p.strip() for p in args.text_prompt.split(",") if p.strip()]

    masker = Sam3Masker(
        text_prompt=prompts if len(prompts) > 1 else prompts[0],
        model_id=args.model_id,
        image_size=args.image_size,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        device=args.device,
        sam3_python=args.sam3_python,
    )

    with torch.no_grad():
        y_norm = masker(x_norm)

    y    = (y_norm * std + mean).clamp(0.0, 1.0)
    y_np = (y[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

    output_path = Path(args.output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(y_np).save(output_path)

    nonzero = int((y_np.sum(axis=2) > 0).sum())
    total   = y_np.shape[0] * y_np.shape[1]
    print(f"Saved masked image → {output_path}")
    print(f"Foreground pixels : {nonzero}/{total} ({100.0 * nonzero / total:.2f} %)")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run SAM3 masking on one image (via subprocess bridge).")
    p.add_argument("--input-image",  required=True, help="Input RGB image path")
    p.add_argument("--output-image", required=True, help="Output masked image path")
    p.add_argument("--model-id",       default="facebook/sam3", help="HF model id")
    p.add_argument("--image-size",     type=int,   default=560,  help="SAM3 internal image size")
    p.add_argument("--text-prompt",    default="cylinder",
                   help="Text prompt(s). Comma-separated for multiple, union taken.")
    p.add_argument("--threshold",      type=float, default=0.5, help="Detection threshold")
    p.add_argument("--mask-threshold", type=float, default=0.5, help="Mask probability threshold")
    p.add_argument("--device",         default="cuda", help="Device (cpu | cuda | cuda:N)")
    p.add_argument("--sam3-python",    default=None,
                   help="Path to the sam3 env python. Defaults to "
                        "/home/saks/miniforge3/envs/sam3/bin/python")
    return p


if __name__ == "__main__":
    _run_cli(_build_arg_parser().parse_args())
