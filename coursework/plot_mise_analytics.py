from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


A = math.sqrt(3.0)
F_NORM = 1.0 / (2.0 * A)
N_REF = 400
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def mise(h: np.ndarray | float, n: int) -> np.ndarray:
    h_array = np.asarray(h, dtype=float)
    values = np.empty_like(h_array)

    mask_1 = h_array <= 1.0
    mask_2 = (h_array > 1.0) & (h_array <= 2.0)
    mask_3 = h_array > 2.0

    h_1 = h_array[mask_1]
    h_2 = h_array[mask_2]
    h_3 = h_array[mask_3]

    values[mask_1] = (
        h_1 / (12.0 * A)
        + (1.0 / n) * (1.0 / (2.0 * A * h_1) - (3.0 - h_1) / (6.0 * A))
    )
    values[mask_2] = (
        (3.0 * h_2**3 - 6.0 * h_2**2 + 6.0 * h_2 - 2.0) / (12.0 * A * h_2**2)
        + (1.0 / n) * (1.0 / (6.0 * A * h_2**2))
    )
    values[mask_3] = (
        (3.0 * h_3**2 - 3.0 * h_3 - 1.0) / (6.0 * A * h_3**2)
        + (1.0 / n) * (1.0 / (6.0 * A * h_3**2))
    )

    return values


def delta(h: np.ndarray | float, n: int) -> np.ndarray:
    return mise(h, n) / F_NORM


def build_h_grid() -> np.ndarray:
    left = np.geomspace(0.02, 1.0, 1200, endpoint=True)
    middle = np.linspace(1.0, 2.0, 900, endpoint=True)[1:]
    right = np.linspace(2.0, 8.0, 1600, endpoint=True)[1:]
    return np.concatenate([left, middle, right])


def find_optimum(n: int, h_grid: np.ndarray) -> tuple[float, float]:
    values = mise(h_grid, n)
    index = int(np.argmin(values))
    return float(h_grid[index]), float(values[index])


def verify_formula() -> list[str]:
    checks: list[str] = []

    left_at_1 = float(mise(np.array([1.0 - 1e-9]), N_REF)[0])
    right_at_1 = float(mise(np.array([1.0 + 1e-9]), N_REF)[0])
    left_at_2 = float(mise(np.array([2.0 - 1e-9]), N_REF)[0])
    right_at_2 = float(mise(np.array([2.0 + 1e-9]), N_REF)[0])

    checks.append(
        f"continuity h=1: |left-right| = {abs(left_at_1 - right_at_1):.3e}"
    )
    checks.append(
        f"continuity h=2: |left-right| = {abs(left_at_2 - right_at_2):.3e}"
    )

    small_values = mise(np.array([0.2, 0.05, 0.02]), N_REF)
    growth_ok = bool(small_values[2] > small_values[1] > small_values[0])
    checks.append(
        "growth near 0+: "
        + (
            f"MISE(0.2)={small_values[0]:.6f}, "
            f"MISE(0.05)={small_values[1]:.6f}, "
            f"MISE(0.02)={small_values[2]:.6f}, monotone={growth_ok}"
        )
    )

    large_value = float(mise(np.array([50.0]), N_REF)[0])
    checks.append(
        f"large h limit: MISE(50) = {large_value:.6f}, target = {F_NORM:.6f}, "
        f"diff = {abs(large_value - F_NORM):.3e}"
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


def save_delta_plot(h_grid: np.ndarray, h_opt: float, mise_opt: float) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    values = mise(h_grid, N_REF)

    ax.plot(h_grid, values, color="#1f5aa6", linewidth=2.2)
    ax.scatter([h_opt], [mise_opt], color="#c0392b", s=45, zorder=3)
    ax.annotate(
        rf"минимум: $\xi^* \approx {h_opt:.3f}$",
        xy=(h_opt, mise_opt),
        xytext=(h_opt + 0.35, mise_opt + 0.02),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=10,
    )
    ax.set_xlim(0.0, 8.0)
    ax.set_xlabel(r"$\xi = h$")
    ax.set_ylabel(r"$\delta(\xi)=\operatorname{MISE}(h,400)$")
    ax.set_title(r"Сырая ошибка $\delta(\xi)$ при фиксированном $n=400$")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "mise_delta_n400.png", dpi=180)
    plt.close(fig)


def save_h_opt_plot(n_values: np.ndarray, h_opt_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    ax.plot(n_values, h_opt_values, color="#0b7a75", linewidth=2.2)
    ax.set_xlim(int(n_values[0]), int(n_values[-1]))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$h_{\mathrm{opt}}(n)$")
    ax.set_title(r"Оптимальная ширина окна $h_{\mathrm{opt}}(n)$")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "optimal_h_vs_n.png", dpi=180)
    plt.close(fig)


def save_min_mise_plot(n_values: np.ndarray, min_mise_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))

    ax.plot(n_values, min_mise_values, color="#7a3e00", linewidth=2.2)
    ax.set_xlim(int(n_values[0]), int(n_values[-1]))
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\min_h \operatorname{MISE}(h,n)$")
    ax.set_title(r"Нижняя граница ошибки по $n$")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "min_mise_vs_n.png", dpi=180)
    plt.close(fig)


def save_efficiency_plot(h_grid: np.ndarray, h_opt: float, delta_min: float) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    efficiency = delta_min / delta(h_grid, N_REF)

    ax.plot(h_grid, efficiency, color="#8e3b8a", linewidth=2.2)
    ax.scatter([h_opt], [1.0], color="#c0392b", s=45, zorder=3)
    ax.annotate(
        r"$E(\xi)$ максимально вблизи $\xi^*$",
        xy=(h_opt, 1.0),
        xytext=(h_opt + 0.55, 0.87),
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=10,
    )
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$\xi = h$")
    ax.set_ylabel(r"$E(\xi)=\Delta_{\min}/\Delta(\xi)$")
    ax.set_title(r"Эффективность по нормированной ошибке при $n=400$")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "efficiency_n400.png", dpi=180)
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    h_grid = build_h_grid()
    n_values = np.arange(10, 1001)

    optima = np.array([find_optimum(int(n), h_grid) for n in n_values], dtype=float)
    h_opt_values = optima[:, 0]
    min_mise_values = optima[:, 1]

    h_opt_ref, mise_opt_ref = find_optimum(N_REF, h_grid)
    delta_min_ref = mise_opt_ref / F_NORM

    save_delta_plot(h_grid, h_opt_ref, mise_opt_ref)
    save_h_opt_plot(n_values, h_opt_values)
    save_min_mise_plot(n_values, min_mise_values)
    save_efficiency_plot(h_grid, h_opt_ref, delta_min_ref)

    print("Generated plots:")
    for filename in (
        "mise_delta_n400.png",
        "optimal_h_vs_n.png",
        "min_mise_vs_n.png",
        "efficiency_n400.png",
    ):
        print(f"  - {PLOTS_DIR / filename}")

    print()
    print("Verification checks:")
    for line in verify_formula():
        print(f"  - {line}")

    print()
    print("Reference optimum for n=400:")
    print(f"  - h_opt = {h_opt_ref:.6f}")
    print(f"  - min MISE = {mise_opt_ref:.6f}")
    print(f"  - min Delta = {delta_min_ref:.6f}")


if __name__ == "__main__":
    main()
