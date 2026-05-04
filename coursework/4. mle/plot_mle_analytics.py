from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


A = math.sqrt(3.0)
PLOTS_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 20260504
SAMPLE_N_REFS = [5, 20, 100]
COMPARISON_N_REFS = [30, 100]
COMPARISON_DELTA = 0.25


def rmise_mle(n: np.ndarray | float) -> np.ndarray:
    n_array = np.asarray(n, dtype=float)
    return 2.0 / (n_array - 2.0)


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


def histogram_bin_probabilities(xi: float, delta: float) -> np.ndarray:
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
        probabilities = histogram_bin_probabilities(float(xi_value), delta)
        sum_p2 = float(np.sum(probabilities**2))
        values[index] = 1.0 + 2.0 / (n * xi_value) - 2.0 * (n + 1.0) * sum_p2 / (n * xi_value)

    return values


def build_xi_grid() -> np.ndarray:
    left = np.geomspace(0.02, 0.25, 550, endpoint=True)
    middle = np.linspace(0.25, 1.5, 1000, endpoint=True)[1:]
    right = np.linspace(1.5, 4.0, 800, endpoint=True)[1:]
    return np.concatenate([left, middle, right])


def find_kernel_optimum(n: int, xi_grid: np.ndarray) -> tuple[float, float]:
    candidate_grid = np.unique(
        np.concatenate([xi_grid, np.array([math.sqrt(6.0 / (n + 2.0)), 1.0, 2.0])])
    )
    values = rmise_kernel(candidate_grid, n)
    index = int(np.argmin(values))
    return float(candidate_grid[index]), float(values[index])


def find_histogram_optimum(n: int, delta: float, xi_grid: np.ndarray) -> tuple[float, float]:
    candidate_grid = np.unique(np.concatenate([xi_grid, np.array([0.5, 1.0, 2.0])]))
    values = rmise_histogram(candidate_grid, n, delta)
    index = int(np.argmin(values))
    return float(candidate_grid[index]), float(values[index])


def density_uniform(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) <= A, 1.0 / (2.0 * A), 0.0)


def mle_density(x: np.ndarray, sample: np.ndarray) -> np.ndarray:
    left = float(np.min(sample))
    right = float(np.max(sample))
    width = right - left
    return np.where((x >= left) & (x <= right), 1.0 / width, 0.0)


def integrated_squared_error(sample: np.ndarray) -> float:
    sample_min = float(np.min(sample))
    sample_max = float(np.max(sample))
    sample_range = sample_max - sample_min
    return 1.0 / sample_range - 1.0 / (2.0 * A)


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


def save_rmise_vs_n_plot() -> None:
    n_values = np.arange(3, 401)
    values = rmise_mle(n_values)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(n_values, values, color="#1f77b4", linewidth=2.2)
    ax.scatter([5, 20, 100, 400], rmise_mle(np.array([5, 20, 100, 400])), s=34)

    ax.set_xlim(3, 400)
    ax.set_ylim(0.0, min(1.2, float(values[2])))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\operatorname{RMISE}_{ML}(n)$")
    ax.set_title(r"Нормированная ОИСКО ММП-оценки")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mle_rmise_vs_n.png", dpi=180)
    plt.close(fig)


def save_mle_density_examples(rng: np.random.Generator) -> None:
    x_plot = np.linspace(-2.2, 2.2, 900)
    true_density = density_uniform(x_plot)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), sharex=True, sharey=True)
    for ax, n_ref in zip(axes, SAMPLE_N_REFS):
        sample = rng.uniform(-A, A, n_ref)
        estimated_density = mle_density(x_plot, sample)
        left = float(np.min(sample))
        right = float(np.max(sample))

        ax.plot(x_plot, true_density, color="#d62728", linewidth=2.0, label=r"$f(x)$")
        ax.plot(
            x_plot,
            estimated_density,
            color="#1f77b4",
            linewidth=2.2,
            label=r"$\widehat f^{ML}(x)$",
        )
        ax.scatter(sample, np.zeros_like(sample), s=18, color="#333333", alpha=0.55)
        ax.axvline(left, color="#1f77b4", linestyle="--", linewidth=1.0)
        ax.axvline(right, color="#1f77b4", linestyle="--", linewidth=1.0)
        ax.set_title(rf"$n={n_ref}$")
        ax.set_xlabel("$x$")
        ax.legend(frameon=True, fontsize=9)

    axes[0].set_ylabel("плотность")
    fig.suptitle("Примеры ММП-оценки плотности", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mle_density_examples.png", dpi=180)
    plt.close(fig)


def save_mc_check_plot(rng: np.random.Generator) -> None:
    n_values = np.array([5, 10, 20, 50, 100, 200])
    repetitions = 30_000
    mc_values: list[float] = []

    for n_ref in n_values:
        sample = rng.uniform(-A, A, size=(repetitions, int(n_ref)))
        ranges = np.max(sample, axis=1) - np.min(sample, axis=1)
        ise = 1.0 / ranges - 1.0 / (2.0 * A)
        normalized_ise = 2.0 * A * ise
        mc_values.append(float(np.mean(normalized_ise)))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    n_grid = np.arange(3, 221)
    ax.plot(n_grid, rmise_mle(n_grid), color="#1f77b4", linewidth=2.1, label="аналитика")
    ax.scatter(n_values, mc_values, color="#d62728", s=42, zorder=3, label="Монте-Карло проверка")

    ax.set_xlim(3, 220)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\operatorname{RMISE}_{ML}(n)$")
    ax.set_title("Проверка аналитической формулы")
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mle_mc_check.png", dpi=180)
    plt.close(fig)


def save_all_methods_comparison_plot() -> None:
    xi_grid = build_xi_grid()
    labels = [str(n) for n in COMPARISON_N_REFS]

    kernel_values = []
    hist_shifted_values = []
    hist_aligned_values = []
    mle_values = []

    for n_ref in COMPARISON_N_REFS:
        _, kernel_min = find_kernel_optimum(n_ref, xi_grid)
        _, hist_shifted_min = find_histogram_optimum(n_ref, COMPARISON_DELTA, xi_grid)
        _, hist_aligned_min = find_histogram_optimum(n_ref, 0.0, xi_grid)
        kernel_values.append(kernel_min)
        hist_shifted_values.append(hist_shifted_min)
        hist_aligned_values.append(hist_aligned_min)
        mle_values.append(float(rmise_mle(n_ref)))

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.bar(x - 1.5 * width, kernel_values, width, label="ядерная")
    ax.bar(x - 0.5 * width, hist_shifted_values, width, label=rf"гист., $\Delta={COMPARISON_DELTA:g}$")
    ax.bar(x + 0.5 * width, mle_values, width, label="ММП")
    ax.bar(x + 1.5 * width, hist_aligned_values, width, label=r"гист., $\Delta=0$")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel("минимальная нормированная ОИСКО")
    ax.set_title("Сравнение методов оценки плотности")
    ax.legend(frameon=True, ncol=2)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mle_methods_comparison.png", dpi=180)
    plt.close(fig)


def verify_formula(rng: np.random.Generator) -> list[str]:
    checks: list[str] = []

    for n_ref in [3, 5, 10, 20, 100, 400]:
        checks.append(f"RMISE_ML({n_ref}) = {float(rmise_mle(n_ref)):.6f}")

    sample = rng.uniform(-A, A, 20)
    left = float(np.min(sample))
    right = float(np.max(sample))
    ise = integrated_squared_error(sample)
    checks.append(f"sample MLE support: [{left:.6f}, {right:.6f}]")
    checks.append(f"sample ISE = {ise:.6f}, normalized = {2.0 * A * ise:.6f}")

    n_ref = 20
    repetitions = 20_000
    mc_sample = rng.uniform(-A, A, size=(repetitions, n_ref))
    ranges = np.max(mc_sample, axis=1) - np.min(mc_sample, axis=1)
    normalized_ise = 2.0 * A * (1.0 / ranges - 1.0 / (2.0 * A))
    mc_mean = float(np.mean(normalized_ise))
    analytic = float(rmise_mle(n_ref))
    checks.append(
        f"MC sanity n={n_ref}: mean={mc_mean:.6f}, analytic={analytic:.6f}, "
        f"diff={abs(mc_mean - analytic):.6f}"
    )

    return checks


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    rng = np.random.default_rng(RANDOM_SEED)
    save_rmise_vs_n_plot()
    save_mle_density_examples(rng)
    save_mc_check_plot(rng)
    save_all_methods_comparison_plot()

    print("Generated plots:")
    for filename in (
        "mle_rmise_vs_n.png",
        "mle_density_examples.png",
        "mle_mc_check.png",
        "mle_methods_comparison.png",
    ):
        print(f"  - {PLOTS_DIR / filename}")

    print()
    print("Verification checks:")
    for line in verify_formula(rng):
        print(f"  - {line}")


if __name__ == "__main__":
    main()
