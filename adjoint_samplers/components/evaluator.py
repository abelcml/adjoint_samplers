# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from typing import Dict
from pathlib import Path
import torch
import ot as pot
import numpy as np

from adjoint_samplers.energies import DoubleWellEnergy, LennardJonesEnergy
from adjoint_samplers.utils.graph_utils import remove_mean
from adjoint_samplers.utils.eval_utils import (
    dist_point_clouds,
    interatomic_dist,
    get_fig_axes,
    fig2img,
)


class DemoEvaluator:
    def __init__(self, energy) -> None:
        from adjoint_samplers.energies.dist_energy import DistEnergy
        assert isinstance(energy, DistEnergy)
        self.dist = energy.dist

        # Plot target samples
        self.fig, axes = get_fig_axes(ncol=6, nrow=10, ax_length_in=3)
        self.axes = axes.reshape(-1)
        self.subplot_idx = 0

    @property
    def ax(self):
        return self.axes[self.subplot_idx]

    def plot_hist(self, x, title=None) -> None:
        B, D = x.shape
        assert D == 1

        if title is None:
            title = f"Eval #{self.subplot_idx}"

        x = (x.reshape(-1)).detach().cpu()
        self.ax.hist(x, bins=50, density=True)
        self.ax.set_xlim(-3.5, 3.5)
        self.ax.set_ylim(0, 0.7)
        self.ax.grid(True)
        self.ax.set_title(title)

    def __call__(self, samples: torch.Tensor) -> Dict:
        # Plot target samples for reference
        if self.subplot_idx == 0:
            target_samples = self.dist.sample([10000,]).cpu()
            self.plot_hist(target_samples, title="Target")
            self.subplot_idx += 1

        # Plot model samples
        self.plot_hist(samples.cpu())
        self.subplot_idx += 1

        # Return figure
        self.fig.canvas.draw()
        PIL_img = fig2img(self.fig)
        return {"hist_img": PIL_img}


class GMM2DEvaluator:
    def __init__(self, energy, num_reference_samples: int, ) -> None:
        from adjoint_samplers.energies.dist_energy import DistEnergy

        assert isinstance(energy, DistEnergy)

        self.dist = energy.dist
        self.reference_samples = self.dist.sample((num_reference_samples,)).detach().cpu()
        ## 保留最開始當下的sample 不該重複抽
        ## shape: (num_reference_samples, 2)

        self.fig, axes = get_fig_axes(
            ncol=6,
            nrow=10,
            ax_length_in=3,
        )

        self.axes = axes.reshape(-1)
        self.subplot_idx = 0

        ## 每次 eval 累積 (eval 序號, SWD)，寫成 CSV 才不會只留在 stdout
        self.swd_rows = []
        self.swd_csv_path = Path("swd.csv")

        ## 直接 savefig，不走 train.py 的 hist_img 路徑：
        ## upstream 的 fig2img 依賴 matplotlib 的 tostring_rgb，3.10 起已移除
        self.fig_path = Path("eval_figs") / "gen.png"
        self.fig_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def ax(self):
        return self.axes[self.subplot_idx]

    def plot_scatter(self, x, title=None):
        B, D = x.shape
        assert D == 2

        x = x.detach().cpu()

        self.ax.scatter(
            x[:, 0],
            x[:, 1],
            s=5,
            alpha=0.5,
        )

        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        self.ax.set_aspect("equal")
        self.ax.grid(True)

        if title is not None:
            self.ax.set_title(title)

    def __call__(self, samples: torch.Tensor) -> Dict:

        samples = samples.detach().cpu()

        # Plot true target once
        if self.subplot_idx == 0:
           
            # target_plot_samples = self.dist.sample((10000,)).cpu()  # this is wrong

            self.plot_scatter(self.reference_samples, title="Target")

            self.subplot_idx += 1

        # # Draw the same number of target samples for SWD comparison
        # target_samples = self.dist.sample(
        #     (samples.shape[0],)
        # ).cpu()   

        # 以上錯誤 改成
        # Use the fixed reference samples for SWD comparison
        target_samples = self.reference_samples

        # Compute SWD
        swd = sliced_wasserstein_distance(
            samples,
            target_samples,
            n_projections=256,
        )

        print(f"GMM2D SWD = {swd:.6f}")

        # Plot generated ASBS samples
        self.plot_scatter(
            samples,
            title=f"SWD = {swd:.3f}",
        )

        self.subplot_idx += 1

        # 記錄這次的 SWD 並覆寫 CSV
        self.swd_rows.append((len(self.swd_rows), swd))
        with open(self.swd_csv_path, "w", encoding="utf-8") as f:
            f.write("eval_idx,swd\n")
            for eval_idx, value in self.swd_rows:
                f.write(f"{eval_idx},{value:.6f}\n")

        # 每次 eval 覆寫同一張圖，跑完就是填滿的 6x10 格
        self.fig.savefig(self.fig_path, bbox_inches="tight")

        return {
            "swd": swd,
        }



class SyntheticEenergyEvaluator:
    def __init__(self, ref_samples_path, energy) -> None:

        assert isinstance(energy, (DoubleWellEnergy, LennardJonesEnergy))
        self.energy = energy
        self.n_particles = energy.n_particles
        self.n_spatial_dim = energy.n_spatial_dim

        # Extract reference samples
        root = Path(os.path.abspath(__file__)).parent.parent.parent
        ref_samples_np = np.load(root / Path(ref_samples_path), allow_pickle=True)
        self.ref_samples = remove_mean(
            torch.tensor(ref_samples_np),
            energy.n_particles,
            energy.n_spatial_dim,
        )

    def __call__(self, samples: torch.Tensor) -> Dict:

        B, D = samples.shape
        assert D == self.energy.dim

        # Sample reference samples
        idxs = torch.randperm(len(self.ref_samples))[:B]
        ref_samples = self.ref_samples[idxs].to(samples.device)


        print("Computing energy W2...")
        gen_energy = self.energy.eval(samples)
        ref_energy = self.energy.eval(ref_samples)
        energy_w2 = pot.emd2_1d(ref_energy.cpu().numpy(), gen_energy.cpu().numpy())**0.5


        print("Computing interatomic W2...")
        gen_dist = interatomic_dist(samples, self.n_particles, self.n_spatial_dim)
        ref_dist = interatomic_dist(ref_samples, self.n_particles, self.n_spatial_dim)
        dist_w2 = pot.emd2_1d(
            gen_dist.cpu().numpy().reshape(-1),
            ref_dist.cpu().numpy().reshape(-1),
        )


        print("Computing particles W2...")
        M = dist_point_clouds(
            samples.reshape(-1, self.n_particles, self.n_spatial_dim).cpu(),
            ref_samples.reshape(-1, self.n_particles, self.n_spatial_dim).cpu(),
        )
        a = torch.ones(M.shape[0]) / M.shape[0]
        b = torch.ones(M.shape[0]) / M.shape[0]
        eq_w2 = pot.emd2(M=M**2, a=a, b=b)**0.5
        eq_w2 = eq_w2.item()


        return {
            "energy_w2": energy_w2,
            "eq_w2": eq_w2,
            "dist_w2": dist_w2,
        }

def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 256,
) -> float:

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {x.shape} and {y.shape}"
        )

    x = x.detach().cpu()
    y = y.detach().cpu()

    directions = torch.randn(
        x.shape[1],
        n_projections,
    )

    directions = directions / directions.norm(
        dim=0,
        keepdim=True,
    )

    x_projected = x @ directions
    y_projected = y @ directions

    x_sorted = torch.sort(
        x_projected,
        dim=0,
    ).values

    y_sorted = torch.sort(
        y_projected,
        dim=0,
    ).values

    return torch.mean(
        torch.abs(x_sorted - y_sorted)
    ).item()