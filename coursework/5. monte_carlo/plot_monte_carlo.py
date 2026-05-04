from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


A = math.sqrt(3.0)
PLOTS_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 20260504
N_REFS = [30, 100]
XI_POINTS = np.array([0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0])
N_MC = 3000
BATCH_SIZE = 150
X_GRID_SIZE = 2401
XI_MAX = 2.0


def true_density(x: np.ndarray) -> np.ndarray:
    return np.where(np.abs(x) <= A, 1.0 / (2.0 * A), 0.0)


def rmise_analytic(xi: np.ndarray | float, n: int) -> np.ndarray:
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


def build_x_grid() -> np.ndarray:
    margin = 0.08
    bound = A * (1.0 + XI_MAX) + margin
    return np.linspace(-bound, bound, X_GRID_SIZE)


def kde_rectangular_batch(samples: np.ndarray, x_grid: np.ndarray, xi: float) -> np.ndarray:
    half_width = A * xi
    inside = np.abs(x_grid[None, :, None] - samples[:, None, :]) <= half_width
    counts = np.sum(inside, axis=2)
    return counts / (samples.shape[1] * 2.0 * A * xi)


def mc_rmise_for_xi(
    n: int,
    xi: float,
    x_grid: np.ndarray,
    f_grid: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    remaining = N_MC

    while remaining > 0:
        batch = min(BATCH_SIZE, remaining)
        samples = rng.uniform(-A, A, size=(batch, n))
        estimates = kde_rectangular_batch(samples, x_grid, xi)
        ise = np.trapezoid((estimates - f_grid[None, :]) ** 2, x_grid, axis=1)
        values.append(2.0 * A * ise)
        remaining -= batch

    all_values = np.concatenate(values)
    mean = float(np.mean(all_values))
    standard_error = float(np.std(all_values, ddof=1) / math.sqrt(len(all_values)))
    return mean, standard_error


def run_monte_carlo() -> dict[int, dict[str, np.ndarray]]:
    rng = np.random.default_rng(RANDOM_SEED)
    x_grid = build_x_grid()
    f_grid = true_density(x_grid)
    results: dict[int, dict[str, np.ndarray]] = {}

    for n_ref in N_REFS:
        means = []
        standard_errors = []
        for xi in XI_POINTS:
            mean, standard_error = mc_rmise_for_xi(n_ref, float(xi), x_grid, f_grid, rng)
            means.append(mean)
            standard_errors.append(standard_error)

        means_array = np.array(means)
        se_array = np.array(standard_errors)
        analytic_array = rmise_analytic(XI_POINTS, n_ref)
        results[n_ref] = {
            "xi": XI_POINTS.copy(),
            "mc": means_array,
            "se": se_array,
            "analytic": analytic_array,
            "relative_error": (means_array - analytic_array) / analytic_array,
        }

    return results


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


def save_mc_vs_analytic_plots(results: dict[int, dict[str, np.ndarray]]) -> None:
    xi_curve = np.linspace(0.04, 2.2, 900)

    for n_ref, result in results.items():
        fig, ax = plt.subplots(figsize=(8.4, 5.2))
        ax.plot(
            xi_curve,
            rmise_analytic(xi_curve, n_ref),
            color="#1f77b4",
            linewidth=2.2,
            label="аналитическая RMISE",
        )
        ax.errorbar(
            result["xi"],
            result["mc"],
            yerr=1.96 * result["se"],
            fmt="o",
            color="#d62728",
            ecolor="#d62728",
            elinewidth=1.0,
            capsize=3,
            label="Монте-Карло",
        )
        ax.set_xlim(0.0, 2.2)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\operatorname{RMISE}(\xi)$")
        ax.set_title(rf"Проверка RMISE методом Монте-Карло, $n={n_ref}$")
        ax.legend(frameon=True)

        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"mc_kernel_rmise_n{n_ref}.png", dpi=180)
        plt.close(fig)


def save_relative_error_plot(results: dict[int, dict[str, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    for n_ref, result in results.items():
        ax.plot(
            result["xi"],
            100.0 * result["relative_error"],
            marker="o",
            linewidth=1.9,
            label=rf"$n={n_ref}$",
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("относительная ошибка MC, %")
    ax.set_title("Отклонение Монте-Карло от аналитической формулы")
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mc_relative_error.png", dpi=180)
    plt.close(fig)


def print_summary(results: dict[int, dict[str, np.ndarray]]) -> None:
    print("Monte Carlo parameters:")
    print(f"  - N_MC = {N_MC}")
    print(f"  - X_GRID_SIZE = {X_GRID_SIZE}")
    print(f"  - seed = {RANDOM_SEED}")
    print()
    print("Comparison table:")
    print("  n      xi      analytic      mc_mean      rel_error")
    for n_ref, result in results.items():
        for xi, analytic, mc_mean, relative_error in zip(
            result["xi"],
            result["analytic"],
            result["mc"],
            result["relative_error"],
        ):
            print(
                f"  {n_ref:<5d}  {xi:>5.2f}  {analytic:>10.6f}  "
                f"{mc_mean:>10.6f}  {relative_error:>9.3%}"
            )


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    results = run_monte_carlo()
    save_mc_vs_analytic_plots(results)
    save_relative_error_plot(results)

    print("Generated plots:")
    for filename in (
        "mc_kernel_rmise_n30.png",
        "mc_kernel_rmise_n100.png",
        "mc_relative_error.png",
    ):
        print(f"  - {PLOTS_DIR / filename}")
    print()
    print_summary(results)


if __name__ == "__main__":
    main()
