from typing import Dict, Any, Optional, Tuple
import torch
import torchvision.transforms.v2.functional as Tv2F


def _round_to_multiple(v: int, m: int) -> int:
    return int(m * round(v / m))


class BatchTransform:
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class BatchMultiCrop(BatchTransform):
    """
    For each image in the batch, sample K patch-aligned crops.
    Returns hr_image (B*K,C,S,S), guidance_image (B*K,C,H,W),
    lr_image (B,C,h_lr,w_lr), guidance_crop (B*K,2) in patch units.
    """

    def __init__(self, crop_size: int, patch_size: int, num_crops: int = 4,
                 global_view_random_resize: Optional[Tuple[float, float]] = None):
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.num_crops = num_crops
        self.global_view_random_resize = global_view_random_resize

    @torch.no_grad()
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        img = batch.pop("image")  # (B,C,H,W)
        if img.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(img.shape)}")
        B, C, H, W = img.shape
        S = self.crop_size

        max_top = max(0, H - S)
        max_left = max(0, W - S)

        crops = []
        coords = []  # (row, col) in patch units
        for b in range(B):
            for _ in range(self.num_crops):
                # patch-aligned random top/left
                top = self.patch_size * torch.randint(0, (max_top // self.patch_size) + 1, (1,)).item()
                left = self.patch_size * torch.randint(0, (max_left // self.patch_size) + 1, (1,)).item()
                crops.append(Tv2F.crop(img[b], top=top, left=left, height=S, width=S))
                coords.append((top // self.patch_size, left // self.patch_size))

        hr = torch.stack(crops, dim=0)  # (B*K,C,S,S)
        lr = Tv2F.resize(img, size=[S, S], antialias=True)  # (B,C,S,S)
        augmented_img = batch.pop("aug_image")
        augmented_guidance = Tv2F.resize(augmented_img, size=[S, S], antialias=True)

        # This extremely slows down training
        if self.global_view_random_resize is not None:
            scale_min, scale_max = self.global_view_random_resize
            scale = torch.empty(1).uniform_(scale_min, scale_max).item()
            new_H = max(self.patch_size, _round_to_multiple(int(H * scale), self.patch_size))
            new_W = max(self.patch_size, _round_to_multiple(int(W * scale), self.patch_size))
            lr = Tv2F.resize(lr, size=[new_H, new_W], antialias=True)  # (B,C,new_H,new_W)
            augmented_guidance = Tv2F.resize(augmented_guidance, size=[new_H, new_W], antialias=True)

        batch["hr_image"] = hr
        batch["guidance_image"] = lr
        batch["augmented_guidance_image"] = augmented_guidance
        batch["lr_image"] = lr
        batch["guidance_crop"] = torch.tensor(coords, dtype=torch.long)  # (B*K,2)
        batch["upsampling_size"] = H // self.patch_size
        return batch
