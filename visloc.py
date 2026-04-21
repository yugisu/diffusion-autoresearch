from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.merge import merge as rasterio_merge
from torch.utils.data import Dataset


def _read_sat_bounds(root: Path, flight_id: str) -> tuple[float, float, float, float]:
    """Returns (lat_min, lon_min, lat_max, lon_max) from the VisLoc coordinates CSV."""
    df = pd.read_csv(root / "satellite_coordinates_range.csv")
    row = df[df["mapname"].isin([f"satellite{flight_id}.tif", f"{flight_id}.tif"])].iloc[0]
    return (
        float(row["RB_lat_map"]),  # lat_min  (RB = right-bottom corner)
        float(row["LT_lon_map"]),  # lon_min  (LT = left-top corner)
        float(row["LT_lat_map"]),  # lat_max
        float(row["RB_lon_map"]),  # lon_max
    )


class UAVDataset(Dataset):
    """VisLoc UAV drone images. Used as the query set — never for training."""

    def __init__(self, root: Path, flight_id: str, transform=None):
        self.drone_dir = root / flight_id / "drone"
        self.transform = transform
        df = pd.read_csv(root / flight_id / f"{flight_id}.csv")
        self.records = df[["filename", "lat", "lon"]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
        row = self.records.iloc[idx]
        img = Image.open(self.drone_dir / row["filename"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, float(row["lat"]), float(row["lon"])


class SatChunkDataset(Dataset):
    """Fixed, evenly-tiled satellite chunks. The retrieval gallery."""

    def __init__(
        self,
        root: Path,
        flight_id: str,
        chunk_pixels: int,
        stride_pixels: int,
        scale_factor: float,
        transform=None,
    ):
        self.chunk_pixels = chunk_pixels
        self.scale_factor = scale_factor
        self.transform = transform

        sat_path = root / flight_id / f"satellite{flight_id}.tif"
        tile_paths = sorted((root / flight_id).glob(f"satellite{flight_id}_*.tif"))

        if sat_path.exists():
            with rasterio.open(sat_path) as src:
                h = int(src.height * scale_factor)
                w = int(src.width * scale_factor)
                data = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.bilinear)
        elif tile_paths:
            srcs = [rasterio.open(p) for p in tile_paths]
            native_res = srcs[0].res
            target_res = (native_res[0] / scale_factor, native_res[1] / scale_factor)
            merged, _ = rasterio_merge(srcs, res=target_res, indexes=[1, 2, 3], resampling=Resampling.bilinear)
            for s in srcs:
                s.close()
            data = merged
            h, w = data.shape[1], data.shape[2]
        else:
            raise FileNotFoundError(f"No satellite TIF found for flight {flight_id}")

        self._img = np.transpose(data, (1, 2, 0))  # (H, W, 3) uint8
        self._bounds = _read_sat_bounds(root, flight_id)  # bounds from CSV (authoritative)
        lat_min, lon_min, lat_max, lon_max = self._bounds

        self._chunks: list[tuple[int, int, float, float]] = []
        self._bboxes: list[tuple[float, float, float, float]] = []
        for y in range(0, h - chunk_pixels, stride_pixels):
            for x in range(0, w - chunk_pixels, stride_pixels):
                cx = x + chunk_pixels // 2
                cy = y + chunk_pixels // 2
                lat = lat_max - (cy / h) * (lat_max - lat_min)
                lon = lon_min + (cx / w) * (lon_max - lon_min)
                self._chunks.append((x, y, lat, lon))
                self._bboxes.append((
                    lat_max - ((y + chunk_pixels) / h) * (lat_max - lat_min),  # lat_min of chunk
                    lon_min + (x / w) * (lon_max - lon_min),  # lon_min of chunk
                    lat_max - (y / h) * (lat_max - lat_min),  # lat_max of chunk
                    lon_min + ((x + chunk_pixels) / w) * (lon_max - lon_min),  # lon_max of chunk
                ))

    @property
    def chunk_coords(self) -> list[tuple[float, float]]:
        return [(lat, lon) for _, _, lat, lon in self._chunks]

    @property
    def chunk_bboxes(self) -> list[tuple[float, float, float, float]]:
        return self._bboxes

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int):  # ty:ignore[invalid-method-override]
        x, y, lat, lon = self._chunks[idx]
        crop = self._img[y : y + self.chunk_pixels, x : x + self.chunk_pixels]
        img = Image.fromarray(crop)
        if self.transform is not None:
            img = self.transform(img)
        return img, lat, lon
