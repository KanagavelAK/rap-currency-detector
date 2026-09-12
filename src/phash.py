"""Perceptual hash (pHash) for finding duplicate and near-duplicate photos.

How it works (be ready to explain this in the verbal round):
1. Shrink the image to 32x32 grayscale, which throws away detail and colour.
2. Apply a 2-D DCT, which describes the image as a mix of frequency patterns.
3. Keep the 8x8 lowest frequencies (the overall structure of the picture).
4. Each of the 64 values becomes 1 if above the median, else 0.
Two photos of the same scene give fingerprints that differ in only a few bits,
so the Hamming distance (number of differing bits) measures similarity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HASH_SIZE = 8
SAMPLE_SIZE = HASH_SIZE * 4  # 32


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    mat = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    mat[0, :] /= np.sqrt(2.0)
    return mat


_DCT = _dct_matrix(SAMPLE_SIZE)


def phash_image(img: Image.Image) -> np.ndarray:
    """Return a 64-element boolean fingerprint."""
    gray = ImageOps.exif_transpose(img).convert("L").resize(
        (SAMPLE_SIZE, SAMPLE_SIZE), Image.Resampling.LANCZOS
    )
    pixels = np.asarray(gray, dtype=np.float64)
    dct = _DCT @ pixels @ _DCT.T
    low = dct[:HASH_SIZE, :HASH_SIZE]
    return (low > np.median(low)).flatten()


def phash_file(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return phash_image(img)


def near_duplicate_pairs(hashes: np.ndarray, max_distance: int, chunk: int = 256) -> list[tuple[int, int, int]]:
    """All pairs (i, j, distance) with i < j and Hamming distance <= max_distance."""
    n = len(hashes)
    pairs: list[tuple[int, int, int]] = []
    for start in range(0, n, chunk):
        block = hashes[start:start + chunk]
        dist = (block[:, None, :] != hashes[None, :, :]).sum(axis=2)
        rows, cols = np.nonzero(dist <= max_distance)
        for r, c in zip(rows, cols):
            i = start + int(r)
            j = int(c)
            if i < j:
                pairs.append((i, j, int(dist[r, c])))
    return pairs


class UnionFind:
    """Groups items that are connected through near-duplicate pairs."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out
