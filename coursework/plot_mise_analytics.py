from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


A = math.sqrt(3.0)
F_NORM = 1.0 / (2.0 * A)
N_REF = 400
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def delta_bar(xi: np.ndarray | float, n: int) -> np.ndarray:
    xi_array = np.asarray(xi, dtype=float)
    values = np.empty_like(xi_array)

    mask_1 = xi_array <= 1.0
    mask_2 = (xi_array > 1.0) & (xi_array <= 2.0)
    mask_3 = xi_array > 2.0

    xi_1 = xi_array[mask_1]
    xi_2 = xi_array[mask_2]
    xi_3 = xi_array[mask_3]

    values[mask_1] = (
        xi_1 / (12.0 * A)
        + (1.0 / n) * (1.0 / (2.0 * A * xi_1) - (3.0 - xi_1) / (6.0 * A))
    )
    values[mask_2] = (
        (3.0 * xi_2**3 - 6.0 * xi_2**2 + 6.0 * xi_2 - 2.0)
        / (12.0 * A * xi_2**2)
        + (1.0 / n) * (1.0 / (6.0 * A * xi_2**2))
    )
    values[mask_3] = (
        (3.0 * xi_3**2 - 3.0 * xi_3 - 1.0) / (6.0 * A * xi_3**2)
        + (1.0 / n) * (1.0 / (6.0 * A * xi_3**2))
    )

    return values


def build_xi_grid() -> np.ndarray:
    left = np.geomspace(0.02, 1.0, 1200, endpoint=True)
    middle = np.linspace(1.0, 2.0, 900, endpoint=True)[1:]
    right = np.linspace(2.0, 8.0, 1600, endpoint=True)[1:]
    return np.concatenate([left, middle, right])


def find_optimum(n: int, xi_grid: np.ndarray) -> tuple[float, float]:
    candidates = [math.sqrt(6.0 / (n + 2.0)), 1.0, 2.0]
    if not 0.0 < candidates[0] <= 1.0:
        values = delta_bar(xi_grid, n)
        index = int(np.argmin(values))
        candidates.append(float(xi_grid[index]))

    candidate_array = np.array(candidates, dtype=float)
    values = delta_bar(candidate_array, n)
    index = int(np.argmin(values))
    return float(candidate_array[index]), float(values[index])


def verify_formula() -> list[str]:
    checks: list[str] = []

    left_at_1 = float(delta_bar(np.array([1.0 - 1e-9]), N_REF)[0])
    right_at_1 = float(delta_bar(np.array([1.0 + 1e-9]), N_REF)[0])
    left_at_2 = float(delta_bar(np.array([2.0 - 1e-9]), N_REF)[0])
    right_at_2 = float(delta_bar(np.array([2.0 + 1e-9]), N_REF)[0])

    checks.append(
        f"continuity xi=1: |left-right| = {abs(left_at_1 - right_at_1):.3e}"
    )
    checks.append(
        f"continuity xi=2: |left-right| = {abs(left_at_2 - right_at_2):.3e}"
    )

    small_values = delta_bar(np.array([0.2, 0.05, 0.02]), N_REF)
    growth_ok = bool(small_values[2] > small_values[1] > small_values[0])
    checks.append(
        "growth near 0+: "
        + (
            f"delta_bar(0.2)={small_values[0]:.6f}, "
            f"delta_bar(0.05)={small_values[1]:.6f}, "
            f"delta_bar(0.02)={small_values[2]:.6f}, monotone={growth_ok}"
        )
    )

    large_value = float(delta_bar(np.array([50.0]), N_REF)[0])
    checks.append(
        f"large xi limit: delta_bar(50) = {large_value:.6f}, target = {F_NORM:.6f}, "
        f"diff = {abs(large_value - F_NORM):.3e}"
    )

    xi_opt, delta_min = find_optimum(N_REF, build_xi_grid())
    off_opt_values = delta_bar(np.array([xi_opt / 2.0, xi_opt * 2.0]), N_REF)
    efficiency_at_opt = float(delta_min / delta_bar(np.array([xi_opt]), N_REF)[0])
    off_opt_ok = bool(np.all(delta_min / off_opt_values < 1.0))
    checks.append(
        f"efficiency: e_400(xi_opt)={efficiency_at_opt:.6f}, "
        f"off optimum < 1 is {off_opt_ok}"
    )

    return checks


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


def save_delta_plot(xi_grid: np.ndarray, xi_opt: float, delta_min: float) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    values = delta_bar(xi_grid, N_REF)

    ax.plot(xi_grid, values, color="#1f5aa6", linewidth=2.2)
    ax.scatter([xi_opt], [delta_min], color="#c0392b", s=45, zorder=3)
    ax.annotate(
        rf"минимум: $\xi_{{\mathrm{{opt}}}} \approx {xi_opt:.3f}$",
        xy=(xi_opt, delta_min),
        xytext=(xi_opt + 0.35, delta_min + 0.02),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=10,
    )
    ax.set_xlim(0.0, 8.0)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\bar{\delta}_{400}(\xi)$")
    ax.set_title(r"ОИСКО $\bar{\delta}_{400}(\xi)$")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "delta_bar_n400.png", dpi=180)
    plt.close(fig)


def save_xi_opt_plot(n_values: np.ndarray, xi_opt_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    ax.plot(n_values, xi_opt_values, color="#0b7a75", linewidth=2.2)
    ax.set_xlim(int(n_values[0]), int(n_values[-1]))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\xi_{\mathrm{opt}}(n)$")
    ax.set_title(r"Оптимальный параметр $\xi_{\mathrm{opt}}(n)$")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "xi_opt_vs_n.png", dpi=180)
    plt.close(fig)


def save_delta_min_plot(n_values: np.ndarray, delta_min_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    ax.plot(n_values, delta_min_values, color="#7a3e00", linewidth=2.2)
    ax.set_xlim(int(n_values[0]), int(n_values[-1]))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\bar{\delta}_{n,\min}$")
    ax.set_title(r"Минимальная ОИСКО $\bar{\delta}_{n,\min}$")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "delta_min_vs_n.png", dpi=180)
    plt.close(fig)


def save_efficiency_plot(xi_grid: np.ndarray, xi_opt: float, delta_min: float) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    delta_values = delta_bar(xi_grid, N_REF)
    efficiency = delta_min / delta_values

    ax.plot(xi_grid, efficiency, color="#8e3b8a", linewidth=2.2)
    ax.axhline(1.0, color="#666666", linewidth=1.1, linestyle="--", alpha=0.75)
    ax.scatter([xi_opt], [1.0], color="#c0392b", s=45, zorder=3)
    ax.annotate(
        r"$e_{400}(\xi)=1$ при $\xi_{\mathrm{opt}}$",
        xy=(xi_opt, 1.0),
        xytext=(xi_opt + 0.55, 0.87),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=10,
    )
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$e_{400}(\xi)$")
    ax.set_title(r"Эффективность $e_{400}(\xi)$")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "efficiency_en_n400.png", dpi=180)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    xi_grid = build_xi_grid()
    n_values = np.arange(10, 1001)

    optima = np.array([find_optimum(int(n), xi_grid) for n in n_values], dtype=float)
    xi_opt_values = optima[:, 0]
    delta_min_values = optima[:, 1]

    xi_opt_ref, delta_min_ref = find_optimum(N_REF, xi_grid)

    save_delta_plot(xi_grid, xi_opt_ref, delta_min_ref)
    save_xi_opt_plot(n_values, xi_opt_values)
    save_delta_min_plot(n_values, delta_min_values)
    save_efficiency_plot(xi_grid, xi_opt_ref, delta_min_ref)

    print("Generated plots:")
    for filename in (
        "delta_bar_n400.png",
        "efficiency_en_n400.png",
        "xi_opt_vs_n.png",
        "delta_min_vs_n.png",
    ):
        print(f"  - {PLOTS_DIR / filename}")

    print()
    print("Verification checks:")
    for line in verify_formula():
        print(f"  - {line}")

    print()
    print("Reference optimum for n=400:")
    print(f"  - xi_opt = {xi_opt_ref:.6f}")
    print(f"  - delta_bar_min = {delta_min_ref:.6f}")
    print(f"  - e_400(xi_opt) = {delta_min_ref / delta_bar(np.array([xi_opt_ref]), N_REF)[0]:.6f}")


if __name__ == "__main__":
    main()
