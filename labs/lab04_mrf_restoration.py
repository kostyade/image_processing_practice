from __future__ import annotations

"""Lab 04 (skeleton): Markov Random Field (MRF) image restoration."""

import argparse
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

PenaltyType = Literal["quadratic", "huber"]


def mrf_energy(
    x: np.ndarray,
    y: np.ndarray,
    lambda_smooth: float,
    penalty: PenaltyType = "quadratic",
    huber_delta: float = 1.0,
) -> float:
    """
    Compute pairwise MRF energy for grayscale image restoration.

    Energy:
        E(x) = sum_p (x_p - y_p)^2 + lambda * sum_(p,q in N) rho(x_p - x_q)

    Args:
        x: Restored image candidate `(H,W)`.
        y: Observed noisy image `(H,W)`.
        lambda_smooth: Smoothness weight.
        penalty: `"quadratic"` or `"huber"`.
        huber_delta: Delta parameter for Huber penalty.

    Returns:
        Scalar energy.
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale images, got x.ndim={x.ndim}, y.ndim={y.ndim}")
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: x {x.shape} vs y {y.shape}")
    if lambda_smooth < 0:
        raise ValueError(f"lambda_smooth must be non-negative, got {lambda_smooth}")

    x = x.astype(np.float64)
    y = y.astype(np.float64)

    # Data term: sum_p (x_p - y_p)^2
    data_term = np.sum((x - y) ** 2)

    # Smoothness term: sum over 4-connected neighbor pairs
    def _penalty(diff: np.ndarray) -> np.ndarray:
        if penalty == "quadratic":
            return diff ** 2
        else:  # huber
            abs_d = np.abs(diff)
            return np.where(
                abs_d <= huber_delta,
                0.5 * diff ** 2,
                huber_delta * (abs_d - 0.5 * huber_delta),
            )

    smooth_term = (
        np.sum(_penalty(x[:, 1:] - x[:, :-1]))   # horizontal
        + np.sum(_penalty(x[1:, :] - x[:-1, :]))  # vertical
    )

    return data_term + lambda_smooth * smooth_term


def mrf_denoise(
    y: np.ndarray,
    lambda_smooth: float,
    num_iters: int,
    step: float = 0.1,
    penalty: PenaltyType = "quadratic",
    huber_delta: float = 1.0,
) -> np.ndarray:
    """
    Restore grayscale image by minimizing MRF energy.

    Args:
        y: Observed noisy image `(H,W)`.
        lambda_smooth: Smoothness weight.
        num_iters: Number of optimization iterations.
        step: Optimization step size.
        penalty: `"quadratic"` or `"huber"`.
        huber_delta: Delta parameter for Huber penalty.

    Returns:
        Restored image with the same shape as `y`.
    """
    if y.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale image, got y.ndim={y.ndim}")
    if num_iters < 0:
        raise ValueError(f"num_iters must be non-negative, got {num_iters}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")

    x = y.astype(np.float64).copy()
    y_f = y.astype(np.float64)

    for _ in range(num_iters):
        # Gradient of data term: 2*(x - y)
        grad = 2.0 * (x - y_f)

        # Gradient of smoothness term (4-connected neighbors)
        def _penalty_grad(diff: np.ndarray) -> np.ndarray:
            if penalty == "quadratic":
                return 2.0 * diff
            else:  # huber
                return np.where(
                    np.abs(diff) <= huber_delta,
                    diff,
                    huber_delta * np.sign(diff),
                )

        smooth_grad = np.zeros_like(x)
        # Horizontal: x[i,j] - x[i,j+1]
        dh = _penalty_grad(x[:, 1:] - x[:, :-1])
        smooth_grad[:, 1:] += dh
        smooth_grad[:, :-1] -= dh
        # Vertical: x[i,j] - x[i+1,j]
        dv = _penalty_grad(x[1:, :] - x[:-1, :])
        smooth_grad[1:, :] += dv
        smooth_grad[:-1, :] -= dv

        grad += lambda_smooth * smooth_grad

        x -= step * grad
        x = np.clip(x, 0.0, 255.0)

    return x.astype(np.float32)


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0,255] uint8 for visualization."""
    if x.size == 0:
        raise ValueError("Input array is empty")
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    return ((x - x_min) / (x_max - x_min) * 255.0).astype(np.uint8)


def main() -> int:
    """
    Lab 04 demo (skeleton).

    Expected behavior after implementation:
    - load grayscale image from `./imgs/`
    - add Gaussian noise (deterministic seed)
    - denoise with MRF (quadratic and/or huber)
    - save side-by-side result to `./out/lab04/mrf_denoise.png`
    """
    parser = argparse.ArgumentParser(description="Lab 04 skeleton (implement functions first).")
    parser.add_argument("--img", type=str, default="lenna.png", help="Input image from ./imgs/")
    parser.add_argument("--out", type=str, default="out/lab04", help="Output directory (relative to repo root)")
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_figure(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    repo_root = Path(__file__).resolve().parents[1]
    imgs_dir = repo_root / "imgs"
    out_dir = (repo_root / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(imgs_dir / args.img), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(imgs_dir / args.img))

    missing: list[str] = []

    try:
        clean = img.astype(np.float32)
        rng = np.random.default_rng(0)
        noisy = clean + rng.normal(0.0, 18.0, size=clean.shape).astype(np.float32)
        noisy = np.clip(noisy, 0.0, 255.0)

        den_quad = mrf_denoise(noisy, lambda_smooth=0.25, num_iters=80, step=0.1, penalty="quadratic")
        den_hub = mrf_denoise(noisy, lambda_smooth=0.25, num_iters=80, step=0.1, penalty="huber", huber_delta=8.0)

        e_noisy_q = mrf_energy(noisy, noisy, lambda_smooth=0.25, penalty="quadratic")
        e_quad = mrf_energy(den_quad, noisy, lambda_smooth=0.25, penalty="quadratic")
        e_noisy_h = mrf_energy(noisy, noisy, lambda_smooth=0.25, penalty="huber", huber_delta=8.0)
        e_hub = mrf_energy(den_hub, noisy, lambda_smooth=0.25, penalty="huber", huber_delta=8.0)

        plt.figure(figsize=(12, 4))
        panels = [
            ("Original", clean),
            ("Noisy (seed=0)", noisy),
            (f"MRF quadratic\nE: {e_noisy_q:.1f} -> {e_quad:.1f}", den_quad),
            (f"MRF huber\nE: {e_noisy_h:.1f} -> {e_hub:.1f}", den_hub),
        ]
        for i, (title, im) in enumerate(panels, start=1):
            plt.subplot(1, 4, i)
            plt.title(title)
            plt.imshow(normalize_to_uint8(im), cmap="gray")
            plt.axis("off")
        save_figure(out_dir / "mrf_denoise.png")
    except NotImplementedError as exc:
        missing.append(str(exc))

    if missing:
        (out_dir / "STATUS.txt").write_text(
            "Lab 04 demo is incomplete. Implement the TODO functions in labs/lab04_mrf_restoration.py.\n\n"
            + "\n".join(f"- {m}" for m in missing)
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {out_dir / 'STATUS.txt'}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
