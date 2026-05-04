from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


A = math.sqrt(3.0)
N_REFS = [30, 100]
DELTA_REFS = [-0.5, -0.25, 0.0, 0.25, 0.5]
COMPARISON_N_REFS = [30, 100]
COMPARISON_DELTA = 0.25
SAMPLE_SIZE = 400
RANDOM_SEED = 20260504
PLOTS_DIR = Path(__file__).resolve().parent


def bin_probabilities(xi: float, delta: float) -> np.ndarray:
    h = A * float(xi)
    x0 = -A + float(delta) * h

    index_min = math.floor((-A - x0) / h) - 2
    index_max = math.ceil((A - x0) / h) + 2
    indices = np.arange(index_min, index_max + 1, dtype=float)

    left = x0 + indices * h
    right = left + h
    overlap = np.maximum(0.0, np.minimum(right, A) - np.maximum(left, -A))
    probabilities = overlap / (2.0 * A)
    return probabilities[probabilities > 1e-15]


def rmise_histogram(xi: np.ndarray | float, n: int, delta: float) -> np.ndarray:
    xi_array = np.asarray(xi, dtype=float)
    values = np.empty_like(xi_array)

    for index, xi_value in np.ndenumerate(xi_array):
        probabilities = bin_probabilities(float(xi_value), delta)
        h = A * float(xi_value)
        sum_p2 = float(np.sum(probabilities**2))
        values[index] = (
            1.0
            + 2.0 * A / (n * h)
            - 2.0 * A * (n + 1.0) * sum_p2 / (n * h)
        )

    return values


def rmise_kernel(xi: np.ndarray | float, n: int) -> np.ndarray:
    xi_array = np.asarray(xi, dtype=float)
    values = np.empty_like(xi_array)

    mask_1 = xi_array <= 1.0
    mask_2 = (xi_array > 1.0) & (xi_array <= 2.0)
    mask_3 = xi_array > 2.0

    xi_1 = xi_array[mask_1]
    xi_2 = xi_array[mask_2]
    xi_3 = xi_array[mask_3]

    values[mask_1] = (
        xi_1 / 6.0
        + (1.0 / n) * (1.0 / xi_1 - (3.0 - xi_1) / 3.0)
    )
    values[mask_2] = (
        (3.0 * xi_2**3 - 6.0 * xi_2**2 + 6.0 * xi_2 - 2.0)
        / (6.0 * xi_2**2)
        + 1.0 / (3.0 * n * xi_2**2)
    )
    values[mask_3] = (
        (3.0 * xi_3**2 - 3.0 * xi_3 - 1.0) / (3.0 * xi_3**2)
        + 1.0 / (3.0 * n * xi_3**2)
    )

    return values


def build_xi_grid() -> np.ndarray:
    left = np.geomspace(0.02, 0.25, 550, endpoint=True)
    middle = np.linspace(0.25, 1.5, 1000, endpoint=True)[1:]
    right = np.linspace(1.5, 4.0, 800, endpoint=True)[1:]
    return np.concatenate([left, middle, right])


def find_optimum(n: int, delta: float, xi_grid: np.ndarray) -> tuple[float, float]:
    candidate_grid = np.unique(np.concatenate([xi_grid, np.array([0.5, 1.0, 2.0])]))
    values = rmise_histogram(candidate_grid, n, delta)
    index = int(np.argmin(values))
    return float(candidate_grid[index]), float(values[index])


def find_kernel_optimum(n: int, xi_grid: np.ndarray) -> tuple[float, float]:
    candidate_grid = np.unique(
        np.concatenate([xi_grid, np.array([math.sqrt(6.0 / (n + 2.0)), 1.0, 2.0])])
    )
    values = rmise_kernel(candidate_grid, n)
    index = int(np.argmin(values))
    return float(candidate_grid[index]), float(values[index])


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.titlesize": 14,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
            "axes.facecolor": "#fbfbfd",
            "figure.facecolor": "white",
        }
    )


def density_uniform(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) <= A, 1.0 / (2.0 * A), 0.0)


def build_histogram_edges(m: int, delta: float) -> np.ndarray:
    h = 2.0 * A / float(m)
    left = -A + delta * h
    right = A + delta * h
    return np.linspace(left, right, m + 1)


def save_sample_histograms_m_grid(sample: np.ndarray) -> None:
    m_values = [5, 10, 15, 20, 25]
    x_plot = np.linspace(-2.3, 2.3, 900)
    y_plot = density_uniform(x_plot)

    fig, axes = plt.subplots(3, 2, figsize=(10.4, 12.0), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for ax, m in zip(axes_flat, m_values):
        edges = build_histogram_edges(m, 0.0)
        ax.hist(
            sample,
            bins=edges,
            density=True,
            color="#9ecae1",
            edgecolor="#4f8ca8",
            linewidth=1.0,
            alpha=0.78,
            label="гистограмма",
        )
        ax.plot(x_plot, y_plot, color="#d62728", linewidth=2.0, label=r"$f(x)$")
        ax.set_title(rf"$m={m}$")
        ax.grid(True, alpha=0.28)
        ax.legend(frameon=True, fontsize=9)

    axes_flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("плотность")
    for ax in axes[-1, :]:
        ax.set_xlabel("$x$")
    axes[1, 1].set_xlabel("$x$")

    fig.suptitle(
        rf"Гистограммные оценки для $m=5(5)25$, $n={SAMPLE_SIZE}$",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sample_histograms_m_grid.png", dpi=180)
    plt.close(fig)


def save_sample_histograms_delta_grid(sample: np.ndarray) -> None:
    delta_values = [-0.4, 0.0, 0.4]
    m = 10
    x_plot = np.linspace(-2.3, 2.3, 900)
    y_plot = density_uniform(x_plot)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharex=True, sharey=True)
    for ax, delta in zip(axes, delta_values):
        edges = build_histogram_edges(m, delta)
        ax.hist(
            sample,
            bins=edges,
            density=True,
            color="#a1d99b",
            edgecolor="#4b8f49",
            linewidth=1.0,
            alpha=0.78,
            label="гистограмма",
        )
        ax.plot(x_plot, y_plot, color="#d62728", linewidth=2.0, label=r"$f(x)$")
        ax.set_title(rf"$\Delta={delta:g}$")
        ax.set_xlabel("$x$")
        ax.grid(True, alpha=0.28)
        ax.legend(frameon=True, fontsize=9)

    axes[0].set_ylabel("плотность")
    fig.suptitle(rf"Влияние сдвига сетки, $m={m}$, $n={SAMPLE_SIZE}$", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sample_histograms_delta_grid.png", dpi=180)
    plt.close(fig)


def save_rmise_vs_xi_plot(xi_grid: np.ndarray, n_refs: list[int]) -> None:
    for n_ref in n_refs:
        fig, ax = plt.subplots(figsize=(8.4, 5.2))

        for delta in DELTA_REFS:
            values = rmise_histogram(xi_grid, n_ref, delta)
            xi_opt, rmise_min = find_optimum(n_ref, delta, xi_grid)
            ax.plot(xi_grid, values, linewidth=1.9, label=rf"$\Delta={delta:g}$")
            ax.scatter([xi_opt], [rmise_min], s=26, zorder=3)

        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"$\xi=h/a$")
        ax.set_ylabel(r"$\operatorname{RMISE}_H(\xi,\Delta)$")
        ax.set_title(rf"Ошибка гистограммной оценки, $n={n_ref}$")
        ax.legend(title=r"сдвиг", frameon=True, ncol=2)

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"hist_rmise_vs_xi_n{n_ref}.png", dpi=180)
        plt.close(fig)


def save_rmise_vs_delta_plot(xi_grid: np.ndarray, n_ref: int) -> None:
    deltas = np.linspace(-0.5, 0.5, 401)
    xi_candidates = [0.25, 0.5, 1.0]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for xi in xi_candidates:
        values = np.array([rmise_histogram(np.array([xi]), n_ref, delta)[0] for delta in deltas])
        ax.plot(deltas, values, linewidth=2.0, label=rf"$\xi={xi:g}$")

    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"$\operatorname{RMISE}_H(\xi,\Delta)$")
    ax.set_title(rf"Зависимость от сдвига сетки, $n={n_ref}$")
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"hist_rmise_vs_delta_n{n_ref}.png", dpi=180)
    plt.close(fig)


def save_optimum_vs_n_plot(xi_grid: np.ndarray) -> None:
    n_values = np.arange(5, 501)
    delta_values = [-0.25, 0.0, 0.25]

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for delta in delta_values:
        optima = np.array([find_optimum(int(n), delta, xi_grid) for n in n_values])
        ax.plot(n_values, optima[:, 0], linewidth=2.0, label=rf"$\Delta={delta:g}$")

    ax.set_xlim(int(n_values[0]), int(n_values[-1]))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\xi_{\mathrm{opt}}$")
    ax.set_title(r"Оптимальная ширина гистограммного бина")
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hist_xi_opt_vs_n.png", dpi=180)
    plt.close(fig)


def save_hist_kernel_comparison_plot(xi_grid: np.ndarray) -> None:
    labels = [str(n) for n in COMPARISON_N_REFS]
    kernel_values = []
    hist_shifted_values = []
    hist_aligned_values = []

    for n_ref in COMPARISON_N_REFS:
        _, kernel_min = find_kernel_optimum(n_ref, xi_grid)
        _, hist_shifted_min = find_optimum(n_ref, COMPARISON_DELTA, xi_grid)
        _, hist_aligned_min = find_optimum(n_ref, 0.0, xi_grid)
        kernel_values.append(kernel_min)
        hist_shifted_values.append(hist_shifted_min)
        hist_aligned_values.append(hist_aligned_min)

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.bar(x - width, kernel_values, width, label="ядерная оценка")
    ax.bar(x, hist_shifted_values, width, label=rf"гистограмма, $\Delta={COMPARISON_DELTA:g}$")
    ax.bar(x + width, hist_aligned_values, width, label=r"гистограмма, $\Delta=0$")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("минимальная нормированная ОИСКО")
    ax.set_title("Сравнение гистограммной и ядерной оценок")
    ax.legend(frameon=True)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hist_kernel_comparison.png", dpi=180)
    plt.close(fig)


def verify_formula(xi_grid: np.ndarray) -> list[str]:
    checks: list[str] = []

    for xi in [0.05, 0.25, 1.0, 2.5]:
        for delta in [-0.5, 0.0, 0.5]:
            probabilities = bin_probabilities(xi, delta)
            checks.append(
                f"sum p_i, xi={xi:g}, Delta={delta:g}: "
                f"{float(np.sum(probabilities)):.12f}"
            )

    for n_ref in N_REFS:
        xi_opt, rmise_min = find_optimum(n_ref, 0.0, xi_grid)
        checks.append(
            f"Delta=0 optimum, n={n_ref}: xi_opt={xi_opt:.6f}, "
            f"RMISE_min={rmise_min:.6f}"
        )

    small_values = rmise_histogram(np.array([0.2, 0.05, 0.02]), 30, 0.0)
    checks.append(
        "growth near 0+, n=30, Delta=0: "
        f"{small_values[0]:.6f}, {small_values[1]:.6f}, {small_values[2]:.6f}"
    )

    return checks


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    rng = np.random.default_rng(RANDOM_SEED)
    sample = rng.uniform(-A, A, SAMPLE_SIZE)

    xi_grid = build_xi_grid()
    save_sample_histograms_m_grid(sample)
    save_sample_histograms_delta_grid(sample)
    save_rmise_vs_xi_plot(xi_grid, N_REFS)
    save_rmise_vs_delta_plot(xi_grid, 100)
    save_optimum_vs_n_plot(xi_grid)
    save_hist_kernel_comparison_plot(xi_grid)

    print("Generated plots:")
    for filename in (
        "sample_histograms_m_grid.png",
        "sample_histograms_delta_grid.png",
        "hist_rmise_vs_xi_n30.png",
        "hist_rmise_vs_xi_n100.png",
        "hist_rmise_vs_delta_n100.png",
        "hist_xi_opt_vs_n.png",
        "hist_kernel_comparison.png",
    ):
        print(f"  - {PLOTS_DIR / filename}")

    print()
    print("Verification checks:")
    for line in verify_formula(xi_grid):
        print(f"  - {line}")


if __name__ == "__main__":
    main()
