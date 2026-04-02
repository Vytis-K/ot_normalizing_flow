import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def make_toy_moons(n_samples: int = 10000, noise: float = 0.06, scale: float = 2.5, seed: int = 42):
    """
    Pure PyTorch two-moons generator.
    Returns a tensor of shape [n_samples, 2].
    """
    g = torch.Generator().manual_seed(seed)

    n_top = n_samples // 2
    n_bottom = n_samples - n_top

    theta_top = torch.rand(n_top, generator=g) * math.pi
    top = torch.stack([torch.cos(theta_top), torch.sin(theta_top)], dim=1)

    theta_bottom = torch.rand(n_bottom, generator=g) * math.pi
    bottom = torch.stack([1.0 - torch.cos(theta_bottom), -torch.sin(theta_bottom) - 0.5], dim=1)

    data = torch.cat([top, bottom], dim=0)
    data = data + noise * torch.randn(data.shape, generator=g)
    data = data * scale
    return data.float()


def save_dataset(data: torch.Tensor, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = out_dir / f"{stem}.pt"
    csv_path = out_dir / f"{stem}.csv"
    plot_path = out_dir / f"{stem}.png"

    torch.save(data, tensor_path)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("x,y\n")
        for row in data.tolist():
            f.write(f"{row[0]},{row[1]}\n")

    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=4, alpha=0.75)
    plt.axis("equal")
    plt.title("Toy Moons Dataset")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    print(f"Saved tensor dataset to: {tensor_path}")
    print(f"Saved CSV dataset to:    {csv_path}")
    print(f"Saved preview plot to:   {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a toy moons dataset into a data folder.")
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--noise", type=float, default=0.06)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--stem", type=str, default="toy_moons")
    args = parser.parse_args()

    data = make_toy_moons(
        n_samples=args.n_samples,
        noise=args.noise,
        scale=args.scale,
        seed=args.seed,
    )
    save_dataset(data, Path(args.out_dir), args.stem)
