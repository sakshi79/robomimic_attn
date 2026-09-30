import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Ensure vendored SAM package resolves when running this file directly.
_SAM_ROOT = Path(__file__).resolve().parent / "segment_anything"
if (_SAM_ROOT / "segment_anything").is_dir():
    _sam_root_s = str(_SAM_ROOT)
    if _sam_root_s not in sys.path:
        sys.path.insert(0, _sam_root_s)


class GroundedSAMMasker(nn.Module):
    """
    Drop-in SAM masking preprocessor for ResNet18Conv.

    Takes a batch of image tensors (B, 3, H, W) already on the model's device,
    normalised with ImageNet stats, and returns the same tensors with background
    pixels zeroed out using Grounded-SAM to produce the masks.

    Usage inside ResNet18Conv.__init__:
        self.masker = GroundedSAMMasker(
            dino_config, dino_checkpoint,
            sam_checkpoint, text_prompt, device=device
        )

    Then at the top of forward(x):
        x = self.masker(x)
    """

    # Default checkpoint / config paths (relative to this file's directory)
    _DEFAULT_DINO_CONFIG     = "GroundingDINO_SwinT_OGC.py"
    _DEFAULT_DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
    _DEFAULT_SAM_CHECKPOINT  = "sam_vit_b_01ec64.pth"

    def __init__(
        self,
        dino_config: str = None,
        dino_checkpoint: str = None,
        sam_checkpoint: str = None,
        text_prompt: str = "cylinder",
        sam_version: str = "vit_b",
        use_sam_hq: bool = True,
        sam_hq_checkpoint: str = "sam_hq_vit_b.pth",
        bert_base_uncased_path: str = None,
        box_threshold: float = 0.55,
        text_threshold: float = 0.25,
        device: str = "cuda",
    ):
        super().__init__()

        # ── Resolve default paths relative to this file ─────────────────
        _here = Path(__file__).resolve().parent
        if dino_config is None:
            dino_config = str(_here / self._DEFAULT_DINO_CONFIG)
        if dino_checkpoint is None:
            dino_checkpoint = str(_here / self._DEFAULT_DINO_CHECKPOINT)
        if sam_checkpoint is None:
            sam_checkpoint = str(_here / self._DEFAULT_SAM_CHECKPOINT)

        # ── Validate paths upfront (fail fast with clear messages) ───────
        for label, path in [
            ("dino_config", dino_config),
            ("dino_checkpoint", dino_checkpoint),
        ]:
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"GroundedSAMMasker: {label} not found at '{path}'"
                )

        if not use_sam_hq and not Path(sam_checkpoint).is_file():
            raise FileNotFoundError(
                f"GroundedSAMMasker: sam_checkpoint not found at '{sam_checkpoint}'"
            )

        # ── Import guards ────────────────────────────────────────────────
        try:
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.models import build_model
            from groundingdino.util.utils import clean_state_dict
        except Exception as e:
            raise ImportError(
                "GroundedSAMMasker requires GroundingDINO to be importable. "
                "Install it via: pip install groundingdino-py"
            ) from e

        try:
            from segment_anything import sam_model_registry, SamPredictor
        except Exception as e:
            raise ImportError(
                "GroundedSAMMasker requires the 'segment_anything' package. "
                "Install it via: pip install segment-anything"
            ) from e

        # if use_sam_hq:
        #     try:
        #         from segment_anything_hq import (
        #             sam_model_registry as sam_hq_model_registry,
        #         )
        #     except Exception as e:
        #         raise ImportError(
        #             "use_sam_hq=True requires 'segment_anything_hq'. "
        #             "Install via: pip install segment-anything-hq"
        #         ) from e

        self.text_prompt    = text_prompt
        self.box_threshold  = box_threshold
        self.text_threshold = text_threshold
        

        # ImageNet stats — used to un-normalise for SAM and re-normalise after masking
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # ── GroundingDINO ────────────────────────────────────────────────
        dino_args = SLConfig.fromfile(dino_config)
        dino_args.device = device
        if bert_base_uncased_path is not None:
            dino_args.bert_base_uncased_path = bert_base_uncased_path
        self.dino = build_model(dino_args)
        ckpt = torch.load(dino_checkpoint, map_location="cpu")
        self.dino.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        self.dino.eval().to(device)

        # ── SAM ──────────────────────────────────────────────────────────
        if use_sam_hq:
            sam_ckpt = sam_hq_checkpoint or sam_checkpoint  # FIX: was silently ignoring sam_hq_checkpoint
            #sam = sam_hq_model_registry[sam_version](checkpoint=sam_ckpt)
        # else:
        sam = sam_model_registry[sam_version](checkpoint=sam_checkpoint)
        sam = sam.to(device)
        self.sam = SamPredictor(sam)

    # ── Internal helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def _get_boxes(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run GroundingDINO on a single (already ImageNet-normalised) CHW tensor.

        Returns normalised cxcywh boxes (N, 4) on CPU.
        """
        caption = self.text_prompt.lower().strip()
        if not caption.endswith("."):
            caption += "."

        # FIX: Dynamically get the device DINO is currently on
        current_device = next(self.dino.parameters()).device

        # DINO expects a batched input
        outputs = self.dino(image_tensor.unsqueeze(0).to(current_device), captions=[caption])
        logits  = outputs["pred_logits"].cpu().sigmoid()[0]   # (nq, 256)
        boxes   = outputs["pred_boxes"].cpu()[0]              # (nq, 4) cxcywh norm

        keep   = logits.max(dim=1).values > self.box_threshold
        return boxes[keep]  # (K, 4)

    @torch.no_grad()
    def _masks_for_image(
        self, rgb_np: np.ndarray, boxes_norm: torch.Tensor
    ) -> np.ndarray:
        """
        Given a uint8 HxWx3 numpy image and normalised cxcywh boxes,
        return a binary (H, W) bool foreground mask.
        """
        H, W = rgb_np.shape[:2]
        self.sam.set_image(rgb_np)

        if boxes_norm.shape[0] == 0:
            return np.zeros((H, W), dtype=bool)

        # cxcywh normalised  →  xyxy pixel
        b = boxes_norm.clone().float()
        b = b * torch.tensor([W, H, W, H], dtype=torch.float32)
        # FIX: original in-place ops on b[:, :2] and b[:, 2:] were order-dependent
        cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1)

        # FIX: Dynamically get the device SAM is currently on
        current_device = next(self.sam.model.parameters()).device

        transformed = self.sam.transform.apply_boxes_torch(
            xyxy, rgb_np.shape[:2]
        ).to(current_device)

        masks, _, _ = self.sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=False,
        )
        # masks: (N, 1, H, W) bool  →  union over all detections
        return masks[:, 0].any(dim=0).cpu().numpy()   # (H, W)

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) ImageNet-normalised tensor on any device.

        Returns:
            Same shape; background pixels zeroed, still ImageNet-normalised.
        """
        # Move normalisation buffers to match input device/dtype
        mean = self.mean.to(x)
        std  = self.std.to(x)

        # Un-normalise → [0, 1] float
        x_01 = (x * std + mean).clamp(0.0, 1.0)

        masked_01 = x_01.clone()
        for i in range(x.shape[0]):
            # uint8 HxWx3 for SAM
            rgb_np = (x_01[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

            # DINO uses the already-normalised tensor
            boxes_norm = self._get_boxes(x[i].cpu())

            # (H, W) bool foreground mask
            mask   = self._masks_for_image(rgb_np, boxes_norm)
            mask_t = torch.from_numpy(mask).to(x.device)           # (H, W)

            masked_01[i] = masked_01[i] * mask_t.unsqueeze(0)      # zero background

        # Re-normalise for downstream ResNet
        return (masked_01 - mean) / std


# ── CLI ─────────────────────────────────────────────────────────────────────


def _run_cli(args):
    image    = Image.open(args.input_image).convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0

    x    = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_norm = (x - mean) / std

    masker = GroundedSAMMasker(
        dino_config=args.dino_config,
        dino_checkpoint=args.dino_checkpoint,
        sam_checkpoint=args.sam_checkpoint,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
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
    _here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run Grounded-SAM masking on one image.")
    p.add_argument("--input-image",  required=True,  help="Input RGB image path")
    p.add_argument("--output-image", required=True,  help="Output masked image path")
    p.add_argument(
        "--dino-config",
        default=str(_here / "GroundingDINO_SwinT_OGC.py"),
        help="GroundingDINO config path",
    )
    p.add_argument(
        "--dino-checkpoint",
        default=str(_here / "groundingdino_swint_ogc.pth"),
        help="GroundingDINO checkpoint path",
    )
    p.add_argument(
        "--sam-checkpoint",
        default=str(_here / "sam_vit_b_01ec64.pth"),
        help="SAM checkpoint path",
    )
    p.add_argument("--text-prompt",    default="cylinder", help="Grounding text prompt")
    p.add_argument("--box-threshold",  type=float, default=0.65,  help="Box filtering threshold")
    p.add_argument("--text-threshold", type=float, default=0.55, help="Text filtering threshold")
    p.add_argument("--device",         default="cpu",   help="Device (cpu | cuda | cuda:N)")
    return p


if __name__ == "__main__":
    _run_cli(_build_arg_parser().parse_args())
