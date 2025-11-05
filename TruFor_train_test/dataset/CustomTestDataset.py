# backend/models/TruFor/TruFor_train_test/dataset/CustomTestDataset.py

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Type accepted for each item in list_img:
# - dict with {'name','bytes'}                         (eager in-memory bytes)
# - dict with {'name','s3_key'} + resolver callable    (lazy; bytes fetched on demand)
# - dict with {'path'} or {'filepath'}                 (local file path)
# - Path or str (treated as a local file path)
ImageInput = Union[Dict[str, Any], str, Path]


def _pil_to_rgb_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to float32 RGB tensor in CHW with values in [0, 1].
    DataLoader with batch_size=1 will wrap this to [1, 3, H, W].
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    t = TF.to_tensor(img)  # -> FloatTensor [3, H, W], range [0,1]
    return t


class CustomTestDataset(Dataset):
    """
    Minimal dataset used by TruFor test-time inference.

    Yields:
        (rgb_tensor: FloatTensor[C, H, W], name: str)

    Supports items shaped as:
      - {'name': str, 'bytes': bytes}
      - {'name': str, 's3_key': str}  (requires resolver to be provided)
      - {'path': str | Path} or {'filepath': str | Path}
      - Path or str (treated as file path)
    """

    def __init__(
        self,
        list_img: Optional[Iterable[ImageInput]] = None,
        *,
        resolver: Optional[Callable[[Dict[str, Any]], bytes]] = None,
    ):
        """
        Args:
            list_img: Iterable of image descriptors (see supported shapes above).
            resolver: Optional callable that converts a dict item containing a
                      lazy reference (e.g., {'name','s3_key'}) into raw bytes.
                      Signature: resolver(item_dict) -> bytes
        """
        self.img_list: List[ImageInput] = list(list_img or [])
        self.resolver = resolver

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        item = self.img_list[idx]
        pil_img, name = self._to_pil_and_name(item)
        rgb = _pil_to_rgb_tensor(pil_img)
        return rgb, name

    # ---- Internal helpers -------------------------------------------------

    def _to_pil_and_name(self, item: ImageInput) -> Tuple[Image.Image, str]:
        # dict cases first
        if isinstance(item, dict):
            # 1) {'name','bytes'} — allow None bytes to fall back to resolver
            if "bytes" in item:
                name = str(item.get("name") or "image")
                data = item.get("bytes", None)
                if data is None and self.resolver is not None:
                    data = self.resolver(item)
                if data is None:
                    raise ValueError(
                        "Item had 'bytes' key but the value was None and no resolver "
                        "was provided to recover the bytes."
                    )
                return Image.open(io.BytesIO(data)), name

            # 2) {'name','s3_key'} — requires resolver
            if "s3_key" in item:
                if self.resolver is None:
                    raise ValueError(
                        "Encountered item with 's3_key' but no resolver was provided. "
                        "Pass resolver=... when constructing CustomTestDataset."
                    )
                name = str(item.get("name") or item["s3_key"] or "image")
                data = self.resolver(item)
                return Image.open(io.BytesIO(data)), name

            # 3) {'path'} / {'filepath'}
            if "path" in item:
                path = Path(item["path"])
                return Image.open(path), path.name
            if "filepath" in item:
                path = Path(item["filepath"])
                return Image.open(path), path.name

            # Unsupported dict shape
            raise ValueError(
                "Unsupported dict format. Expected one of: "
                "{'name','bytes' (not None)}, {'path'}, or {'name','s3_key'} with a resolver."
            )

        # path-like cases
        if isinstance(item, (str, Path)):
            p = Path(item)
            return Image.open(p), p.name

        # Fallback
        raise TypeError(
            f"Unsupported item type {type(item)}. "
            "Expected dict with {'name','bytes'} / {'name','s3_key'} / {'path'}, or a path-like."
        )
