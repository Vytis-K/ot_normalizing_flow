import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import VelocityMLP


class TensorDataset2D(Dataset):
    def __init__(self, tensor_path: str):
        x = torch.load(tensor_path, map_location="cpu")
        if not isinstance(x, torch.Tensor):
            raise TypeError("Loaded dataset is not a torch.Tensor.")
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"Expected tensor shape [N, 2], got {tuple(x.shape)}")
        self.x = x.float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.x[idx]


@torch.no_grad()
def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).T
    return x2 + y2 - 2.0 * (x @ y.T)


@torch.no_grad()
def sinkhorn_log_domain(cost: torch.Tensor, epsilon: float = 0.05, n_iters: int = 100) -> torch.Tensor:
    """
    Entropic OT between two uniform empirical minibatches.
    Returns plan P with shape [B, B].
    """
    batch_size = cost.shape[0]
    device = cost.device
    log_a = torch.full((batch_size,), -math.log(batch_size), device=device)
    log_b = torch.full((batch_size,), -math.log(batch_size), device=device)

    log_k = -cost / epsilon
    u = torch.zeros(batch_size, device=device)
    v = torch.zeros(batch_size, device=device)

    for _ in range(n_iters):
        u = log_a - torch.logsumexp(log_k + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_k + u[:, None], dim=0)

    log_p = log_k + u[:, None] + v[None, :]
    p = torch.exp(log_p)
    p = p / p.sum().clamp_min(1e-12)
    return p


@torch.no_grad()
def independent_coupling(z: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    perm = torch.randperm(x.shape[0], device=x.device)
    return z, x[perm], None


@torch.no_grad()
def minibatch_ot_coupling(
    z: torch.Tensor,
    x: torch.Tensor,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cost = pairwise_sq_dists(z, x)
    plan = sinkhorn_log_domain(cost, epsilon=epsilon, n_iters=sinkhorn_iters)
    cond = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
    idx = torch.multinomial(cond, num_samples=1).squeeze(1)
    x_paired = x[idx]
    return z, x_paired, plan



def conditional_flow_matching_loss(model: VelocityMLP, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    batch_size = z.shape[0]
    t = torch.rand(batch_size, device=z.device)
    x_t = (1.0 - t[:, None]) * z + t[:, None] * x
    target_velocity = x - z
    pred_velocity = model(x_t, t)
    return F.mse_loss(pred_velocity, target_velocity)


@torch.no_grad()
def sample_model(model: VelocityMLP, n_samples: int, device: str, n_steps: int = 200) -> torch.Tensor:
    x = torch.randn(n_samples, 2, device=device)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((n_samples,), step / n_steps, device=device)
        v = model(x, t)
        x = x + dt * v
    return x


@torch.no_grad()
def trajectory_energy(model: VelocityMLP, z0: torch.Tensor, n_steps: int = 100) -> float:
    x = z0.clone()
    dt = 1.0 / n_steps
    total = torch.zeros(z0.shape[0], device=z0.device)
    for step in range(n_steps):
        t = torch.full((z0.shape[0],), step / n_steps, device=z0.device)
        v = model(x, t)
        total += (v ** 2).sum(dim=1) * dt
        x = x + dt * v
    return total.mean().item()


@torch.no_grad()
def save_scatter_plot(real_x: torch.Tensor, fake_x: torch.Tensor, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(real_x[:, 0].cpu(), real_x[:, 1].cpu(), s=4, alpha=0.35, label="real")
    plt.scatter(fake_x[:, 0].cpu(), fake_x[:, 1].cpu(), s=4, alpha=0.55, label="generated")
    plt.legend()
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


@torch.no_grad()
def save_trajectory_plot(model: VelocityMLP, out_path: Path, device: str, n_traj: int = 32, n_steps: int = 100) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z = torch.randn(n_traj, 2, device=device)
    x = z.clone()
    traj = [x.cpu()]
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((n_traj,), step / n_steps, device=device)
        v = model(x, t)
        x = x + dt * v
        traj.append(x.cpu())
    traj = torch.stack(traj, dim=1)

    plt.figure(figsize=(6, 6))
    for i in range(n_traj):
        plt.plot(traj[i, :, 0], traj[i, :, 1], linewidth=1.0)
    plt.axis("equal")
    plt.title("Learned trajectories")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()



def main() -> None:
    parser = argparse.ArgumentParser(description="Train 2D conditional flow matching with optional minibatch OT coupling.")
    parser.add_argument("--data_path", type=str, default="data/toy_moons.pt")
    parser.add_argument("--save_dir", type=str, default="runs/toy_moons")
    parser.add_argument("--coupling", type=str, choices=["independent", "ot"], default="ot")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn_iters", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = TensorDataset2D(args.data_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = VelocityMLP(data_dim=2, hidden_dim=args.hidden_dim, time_dim=args.time_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for x in loader:
            x = x.to(device)
            z = torch.randn_like(x)

            if args.coupling == "independent":
                z_pair, x_pair, _ = independent_coupling(z, x)
            else:
                z_pair, x_pair, _ = minibatch_ot_coupling(
                    z, x, epsilon=args.epsilon, sinkhorn_iters=args.sinkhorn_iters
                )

            loss = conditional_flow_matching_loss(model, z_pair, x_pair)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)
        loss_history.append(epoch_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            energy = trajectory_energy(model, torch.randn(512, 2, device=device), n_steps=100)
            print(
                f"Epoch {epoch + 1:04d} | loss={epoch_loss:.6f} | trajectory_energy={energy:.6f}"
            )

    torch.save(model.state_dict(), save_dir / f"model_{args.coupling}.pt")
    torch.save(torch.tensor(loss_history), save_dir / f"loss_history_{args.coupling}.pt")

    model.eval()
    generated = sample_model(model, n_samples=5000, device=device, n_steps=args.sample_steps)
    save_scatter_plot(
        dataset.x,
        generated.cpu(),
        save_dir / f"samples_{args.coupling}.png",
        title=f"Toy Moons: {args.coupling} coupling",
    )
    save_trajectory_plot(model, save_dir / f"trajectories_{args.coupling}.png", device=device)

    print(f"Saved weights to: {(save_dir / f'model_{args.coupling}.pt').resolve()}")
    print(f"Saved plots in:  {save_dir.resolve()}")


if __name__ == "__main__":
    main()
