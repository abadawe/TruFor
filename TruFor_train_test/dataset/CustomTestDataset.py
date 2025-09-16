from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union, Optional
import io
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

ImageInput = Union[
    Dict[str, Any],      # {"name": str, "bytes": bytes}  OR {"path": str}
    Tuple[str, bytes],   # (name, bytes)
    bytes,               # raw image bytes
    str,                 # file path
    Path,                # file path
    Image.Image,         # PIL image (uses .filename if present)
]


class CustomTestDataset(Dataset):
    """
    Accepts a heterogeneous list of image items:
      - {"name": str, "bytes": bytes}
      - {"path": str|Path}
      - (name, bytes)
      - bytes
      - str|Path  (file path)
      - PIL.Image.Image

    Returns:
      (C,H,W) float32 tensor in [0,1], filename (str)
    """

    def __init__(self, list_img: Optional[Iterable[ImageInput]] = None):
        self.img_list: List[ImageInput] = list(list_img or [])

    def shuffle(self) -> None:
        if self.img_list:
            random.shuffle(self.img_list)

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int):
        assert self.img_list, "Empty dataset"
        assert 0 <= index < len(self.img_list), f"Index {index} is not available!"

        item = self.img_list[index]
        pil_img, name = self._to_pil_and_name(item)
        pil_img = pil_img.convert("RGB")

        # (H, W, C) -> (C, H, W), float32 in [0, 1]
        arr = np.array(pil_img, dtype=np.uint8)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32) / 255.0

        return tensor, name

    # --------------------------
    # Helpers
    # --------------------------
    def _to_pil_and_name(self, item: ImageInput) -> Tuple[Image.Image, str]:
        # dict cases
        if isinstance(item, dict):
            if "bytes" in item:
                name = str(item.get("name") or "image")
                return Image.open(io.BytesIO(item["bytes"])), name
            if "path" in item:
                path = Path(item["path"])
                return Image.open(path), path.name
            if "filepath" in item:
                path = Path(item["filepath"])
                return Image.open(path), path.name
            raise ValueError("Unsupported dict format. Expected keys: {'name','bytes'} or {'path'}.")

        # tuple(name, bytes)
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (bytes, bytearray)):
            return Image.open(io.BytesIO(item[1])), item[0]

        # raw bytes
        if isinstance(item, (bytes, bytearray)):
            return Image.open(io.BytesIO(item)), "image"

        # path-like
        if isinstance(item, (str, Path)):
            p = Path(item)
            return Image.open(p), p.name

        # PIL.Image
        if isinstance(item, Image.Image):
            name = getattr(item, "filename", None)
            name = Path(name).name if name else "image"
            return item, name

        # Fallback: try file-like objects
        if hasattr(item, "read"):
            # file-like object with .read()
            data = item.read()
            return Image.open(io.BytesIO(data)), "image"

        raise TypeError(f"Unsupported image item type: {type(item)}")

    def get_filename(self, index: int) -> str:
        _, name = self.__getitem__(index)
        return name
