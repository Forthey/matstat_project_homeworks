from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


A = math.sqrt(3.0)
F_NORM = 1.0 / (2.0 * A)
N_REF = [30, 100]
PLOTS_DIR = Path(__file__).resolve().parent


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


def normalize_n_ref(n_ref: int | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(n_ref, int):
        return [n_ref]
    return [int(n) for n in n_ref]


def format_n_refs(n_refs: list[int]) -> str:
    return "_".join(str(n) for n in n_refs)


def delta_plot_filename(n_refs: list[int]) -> str:
    if len(n_refs) == 1:
        return f"delta_bar_n{n_refs[0]}.png"
    return f"delta_bar_n{format_n_refs(n_refs)}.png"


def efficiency_plot_filename(n_refs: list[int]) -> str:
    if len(n_refs) == 1:
        return f"efficiency_en_n{n_refs[0]}.png"
    return f"efficiency_en_n{format_n_refs(n_refs)}.png"


def verify_formula(n_refs: list[int]) -> list[str]:
    checks: list[str] = []
    n_ref = n_refs[0]

    left_at_1 = float(delta_bar(np.array([1.0 - 1e-9]), n_ref)[0])
    right_at_1 = float(delta_bar(np.array([1.0 + 1e-9]), n_ref)[0])
    left_at_2 = float(delta_bar(np.array([2.0 - 1e-9]), n_ref)[0])
    right_at_2 = float(delta_bar(np.array([2.0 + 1e-9]), n_ref)[0])

    checks.append(
        f"continuity xi=1: |left-right| = {abs(left_at_1 - right_at_1):.3e}"
    )
    checks.append(
        f"continuity xi=2: |left-right| = {abs(left_at_2 - right_at_2):.3e}"
    )

    small_values = delta_bar(np.array([0.2, 0.05, 0.02]), n_ref)
    growth_ok = bool(small_values[2] > small_values[1] > small_values[0])
    checks.append(
        "growth near 0+: "
        + (
            f"delta_bar(0.2)={small_values[0]:.6f}, "
            f"delta_bar(0.05)={small_values[1]:.6f}, "
            f"delta_bar(0.02)={small_values[2]:.6f}, monotone={growth_ok}"
        )
    )

    large_value = float(delta_bar(np.array([50.0]), n_ref)[0])
    checks.append(
        f"large xi limit: delta_bar(50) = {large_value:.6f}, target = {F_NORM:.6f}, "
        f"diff = {abs(large_value - F_NORM):.3e}"
    )

    xi_opt, delta_min = find_optimum(n_ref, build_xi_grid())
    off_opt_values = delta_bar(np.array([xi_opt / 2.0, xi_opt * 2.0]), n_ref)
    efficiency_at_opt = float(delta_min / delta_bar(np.array([xi_opt]), n_ref)[0])
    off_opt_ok = bool(np.all(delta_min / off_opt_values < 1.0))
    checks.append(
        f"efficiency: e_{n_ref}(xi_opt)={efficiency_at_opt:.6f}, "
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


def save_delta_plot(
    xi_grid: np.ndarray,
    ref_optima: dict[int, tuple[float, float]],
    n_refs: list[int],
) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    for n_ref in n_refs:
        xi_opt, delta_min = ref_optima[n_ref]
        values = delta_bar(xi_grid, n_ref)
        ax.plot(xi_grid, values, linewidth=2.2, label=rf"$n={n_ref}$")
        ax.scatter([xi_opt], [delta_min], s=38, zorder=3)

    if len(n_refs) == 1:
        n_ref = n_refs[0]
        xi_opt, delta_min = ref_optima[n_ref]
        ax.annotate(
            rf"минимум: $\xi_{{\mathrm{{opt}}}} \approx {xi_opt:.3f}$",
            xy=(xi_opt, delta_min),
            xytext=(xi_opt + 0.35, delta_min + 0.02),
            arrowprops={"arrowstyle": "->", "color": "#444444"},
            fontsize=10,
        )
    else:
        ax.legend(title=r"$n$", frameon=True)

    ax.set_xlim(0.0, 8.0)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$\bar{\delta}_{n}(\xi)$")
    ax.set_title(r"ОИСКО $\bar{\delta}_{n}(\xi)$")

    fig.tight_layout()
    filename = delta_plot_filename(n_refs)
    fig.savefig(PLOTS_DIR / filename, dpi=180)
    plt.close(fig)
    return filename


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


def save_efficiency_plot(
    xi_grid: np.ndarray,
    ref_optima: dict[int, tuple[float, float]],
    n_refs: list[int],
) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    for n_ref in n_refs:
        xi_opt, delta_min = ref_optima[n_ref]
        delta_values = delta_bar(xi_grid, n_ref)
        efficiency = delta_min / delta_values
        ax.plot(xi_grid, efficiency, linewidth=2.2, label=rf"$n={n_ref}$")
        ax.scatter([xi_opt], [1.0], s=38, zorder=3)

    ax.axhline(1.0, color="#666666", linewidth=1.1, linestyle="--", alpha=0.75)
    if len(n_refs) == 1:
        n_ref = n_refs[0]
        xi_opt, _ = ref_optima[n_ref]
        ax.annotate(
            rf"$e_{{{n_ref}}}(\xi)=1$ при $\xi_{{\mathrm{{opt}}}}$",
            xy=(xi_opt, 1.0),
            xytext=(xi_opt + 0.55, 0.87),
            arrowprops={"arrowstyle": "->", "color": "#444444"},
            fontsize=10,
        )
    else:
        ax.legend(title=r"$n$", frameon=True)

    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$e_n(\xi)$")
    ax.set_title(r"Эффективность $e_n(\xi)$")

    fig.tight_layout()
    filename = efficiency_plot_filename(n_refs)
    fig.savefig(PLOTS_DIR / filename, dpi=180)
    plt.close(fig)
    return filename


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    n_refs = normalize_n_ref(N_REF)
    xi_grid = build_xi_grid()
    n_values = np.arange(10, 1001)

    optima = np.array([find_optimum(int(n), xi_grid) for n in n_values], dtype=float)
    xi_opt_values = optima[:, 0]
    delta_min_values = optima[:, 1]

    ref_optima = {n_ref: find_optimum(n_ref, xi_grid) for n_ref in n_refs}

    delta_plot = save_delta_plot(xi_grid, ref_optima, n_refs)
    save_xi_opt_plot(n_values, xi_opt_values)
    save_delta_min_plot(n_values, delta_min_values)
    efficiency_plot = save_efficiency_plot(xi_grid, ref_optima, n_refs)

    print("Generated plots:")
    for filename in (
        delta_plot,
        efficiency_plot,
        "xi_opt_vs_n.png",
        "delta_min_vs_n.png",
    ):
        print(f"  - {PLOTS_DIR / filename}")

    print()
    print("Verification checks:")
    for line in verify_formula(n_refs):
        print(f"  - {line}")

    print()
    print("Reference optima:")
    for n_ref in n_refs:
        xi_opt_ref, delta_min_ref = ref_optima[n_ref]
        efficiency_at_opt = delta_min_ref / delta_bar(np.array([xi_opt_ref]), n_ref)[0]
        print(f"  - n={n_ref}:")
        print(f"      xi_opt = {xi_opt_ref:.6f}")
        print(f"      delta_bar_min = {delta_min_ref:.6f}")
        print(f"      e_{n_ref}(xi_opt) = {efficiency_at_opt:.6f}")


if __name__ == "__main__":
    main()
