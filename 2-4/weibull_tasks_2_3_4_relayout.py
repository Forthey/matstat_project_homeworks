from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt


LAMBDA = 1.0
C = 3.0
N_SAMPLE = 400

M_VALUES_HIST = [5, 10, 15, 20, 25, 30, 40]
M_GROUPS_HIST = [
    [5, 10, 15, 20, 25],
    [10, 20, 30, 40],
]
N_VALUES_PROJ = list(range(10, 101, 10))
N_VALUES_PROJ_SHOW = [10, 20, 30, 50, 100]
H_VALUES_KDE = np.arange(0.05, 1.01, 0.05)
H_VALUES_KDE_SHOW = [0.05, 0.10, 0.20, 0.50]

R_HIST = 1000
R_PROJ = 1000
R_KDE = 300

L_HIST = 3.0
L_PROJ_PLOT = 5.0
L_PROJ_INT = 20.0
L_KDE = 5.0
DX_KDE = 0.005

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_FILE = BASE_DIR / "results_tasks_2_3_4_relayout.txt"


plt.rcParams.update(
    {
        "figure.facecolor": "#f4f4f4",
        "axes.facecolor": "#f2f2f2",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
    }
)


# ============================================================
# Базовые функции
# ============================================================
def weibull_pdf(x: np.ndarray | float, lam: float = LAMBDA, c: float = C) -> np.ndarray | float:
    return lam * c * np.asarray(x) ** (c - 1) * np.exp(-lam * np.asarray(x) ** c)



def sample_weibull_inverse_transform(
    size: int,
    lam: float = LAMBDA,
    c: float = C,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Метод Монте-Карло через метод обращения:
        U ~ Uniform(0, 1)
        X = (-ln(1-U)/lam)^(1/c)
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(size)
    return (-np.log(1.0 - u) / lam) ** (1.0 / c)


# ============================================================
# Задача 2. Гистограммная оценка
# ============================================================
def histogram_density_estimate(sample: np.ndarray, m: int, L: float = L_HIST):
    counts, edges = np.histogram(sample, bins=m, range=(0.0, L), density=False)
    width = edges[1] - edges[0]
    heights = counts / (len(sample) * width)
    return counts, edges, heights, width



def histogram_ise(sample: np.ndarray, m: int, L: float = L_HIST, grid_size: int = 30001) -> float:
    _, _, heights, width = histogram_density_estimate(sample, m, L=L)
    x_grid = np.linspace(0.0, L, grid_size)
    dx = x_grid[1] - x_grid[0]
    f_true = weibull_pdf(x_grid)
    idx = np.floor(x_grid[:-1] / width).astype(int)
    idx = np.clip(idx, 0, m - 1)
    f_hat = heights[idx]
    return float(np.sum((f_hat - f_true[:-1]) ** 2) * dx)



def compute_histogram_stats() -> tuple[dict[int, float], dict[int, float]]:
    rng = np.random.default_rng(12345)
    mean_vals: dict[int, float] = {}
    std_vals: dict[int, float] = {}

    for m in M_VALUES_HIST:
        vals = np.empty(R_HIST)
        for i in range(R_HIST):
            sample = sample_weibull_inverse_transform(N_SAMPLE, rng=rng)
            vals[i] = histogram_ise(sample, m)
        mean_vals[m] = float(vals.mean())
        std_vals[m] = float(vals.std(ddof=1))

    return mean_vals, std_vals



def plot_histogram_for_m(sample: np.ndarray, m: int, output_path: Path) -> None:
    x = np.linspace(0.0, L_HIST, 1200)
    y = weibull_pdf(x)

    plt.figure(figsize=(8.5, 4.8))
    plt.hist(
        sample,
        bins=m,
        range=(0.0, L_HIST),
        density=True,
        color="#a6cee3",
        edgecolor="#5b8fa8",
        alpha=0.9,
        label="Гистограмма",
    )
    plt.plot(x, y, color="red", linewidth=2, label="Теоретическая f(x)")
    plt.title(f"Гистограмма, m = {m}")
    plt.xlabel("x")
    plt.ylabel("Плотность")
    plt.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()



def draw_histogram_on_ax(ax, sample: np.ndarray, m: int) -> None:
    x = np.linspace(0.0, L_HIST, 1200)
    y = weibull_pdf(x)

    ax.hist(
        sample,
        bins=m,
        range=(0.0, L_HIST),
        density=True,
        color="#a6cee3",
        edgecolor="#5b8fa8",
        alpha=0.9,
        label="Гистограмма",
    )
    ax.plot(x, y, color="red", linewidth=2, label="Теоретическая f(x)")
    ax.set_title(f"m = {m}")
    ax.set_xlabel("x")
    ax.set_ylabel("Плотность")
    ax.legend(loc="upper right", framealpha=0.95)


def plot_histogram_grid(
    sample: np.ndarray,
    m_values: list[int],
    output_path: Path,
    title: str,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.ravel()

    for ax, m in zip(axes_flat, m_values):
        draw_histogram_on_ax(ax, sample, m)

    for ax in axes_flat[len(m_values) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_histogram_delta(hist_mean: dict[int, float], output_path: Path) -> None:
    best_m = min(hist_mean, key=hist_mean.get)
    ms = sorted(hist_mean.keys())
    ys = [hist_mean[m] for m in ms]

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(ms, ys, marker="o", linewidth=2, color="#1f77b4")
    plt.scatter([best_m], [hist_mean[best_m]], color="green", s=80, zorder=3, label=f"Минимум при m={best_m}")
    plt.title("Зависимость ОИСКО от числа разрядов m")
    plt.xlabel("Число разрядов m")
    plt.ylabel(r"$\overline{\delta_n}(m)$")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


# ============================================================
# Задача 3. Проекционная оценка по функциям Лагерра
# ============================================================
def laguerre_basis_matrix(x: np.ndarray, N: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    phi = np.empty((N + 1, x.size), dtype=float)

    expfac = np.exp(-x / 2.0)
    l0 = np.ones_like(x)
    phi[0] = expfac * l0

    if N == 0:
        return phi

    l1 = 1.0 - x
    phi[1] = expfac * l1

    prevprev = l0
    prev = l1
    for k in range(1, N):
        nxt = ((2 * k + 1 - x) * prev - k * prevprev) / (k + 1)
        phi[k + 1] = expfac * nxt
        prevprev, prev = prev, nxt

    return phi



def compute_projection_stats() -> tuple[dict[int, float], dict[int, float]]:
    x_int = np.linspace(0.0, L_PROJ_INT, 200001)
    dx_int = x_int[1] - x_int[0]
    phi_true_grid = laguerre_basis_matrix(x_int, 100)
    f_true_grid = weibull_pdf(x_int)
    c_true = phi_true_grid.dot(f_true_grid) * dx_int
    f_l2_norm_sq = float(np.sum(f_true_grid**2) * dx_int)
    cum_energy = np.cumsum(c_true**2)
    tail_energy = {N: float(f_l2_norm_sq - cum_energy[N]) for N in N_VALUES_PROJ}

    rng = np.random.default_rng(54321)
    mean_vals: dict[int, float] = {}
    std_vals: dict[int, float] = {}

    for N in N_VALUES_PROJ:
        vals = np.empty(R_PROJ)
        true_coeffs = c_true[: N + 1]
        tail = tail_energy[N]
        for i in range(R_PROJ):
            sample = sample_weibull_inverse_transform(N_SAMPLE, rng=rng)
            c_hat = laguerre_basis_matrix(sample, N).mean(axis=1)
            vals[i] = float(np.sum((c_hat - true_coeffs) ** 2) + tail)
        mean_vals[N] = float(vals.mean())
        std_vals[N] = float(vals.std(ddof=1))

    return mean_vals, std_vals



def projection_estimate(sample: np.ndarray, N: int, x: np.ndarray) -> np.ndarray:
    phi_sample = laguerre_basis_matrix(sample, N)
    c_hat = phi_sample.mean(axis=1)
    phi_x = laguerre_basis_matrix(x, N)
    return c_hat @ phi_x



def plot_projection_for_N(sample: np.ndarray, N: int, output_path: Path) -> None:
    x = np.linspace(0.0, L_PROJ_PLOT, 1600)
    f_true = weibull_pdf(x)
    f_hat = projection_estimate(sample, N, x)

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(x, f_true, color="red", linewidth=2, label="Теоретическая f(x)")
    plt.plot(x, f_hat, color="blue", linestyle="--", linewidth=2, label=rf"$\tilde{{f}}_{{{N}}}(x)$")
    plt.title(f"Проекционная оценка, N = {N}")
    plt.xlabel("x")
    plt.ylabel("Плотность")
    plt.ylim(min(-0.25, float(np.min(f_hat)) - 0.02), max(0.85, float(np.max(f_true)) + 0.05))
    plt.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()



def draw_projection_on_ax(ax, sample: np.ndarray, N: int) -> None:
    x = np.linspace(0.0, L_PROJ_PLOT, 1600)
    f_true = weibull_pdf(x)
    f_hat = projection_estimate(sample, N, x)

    ax.plot(x, f_true, color="red", linewidth=2, label="Теоретическая f(x)")
    ax.plot(x, f_hat, color="blue", linestyle="--", linewidth=2, label=rf"$\tilde{{f}}_{{{N}}}(x)$")
    ax.set_title(f"N = {N}")
    ax.set_xlabel("x")
    ax.set_ylabel("Плотность")
    ax.set_ylim(min(-0.25, float(np.min(f_hat)) - 0.02), max(0.85, float(np.max(f_true)) + 0.05))
    ax.legend(loc="upper right", framealpha=0.95)


def plot_projection_grid(
    sample: np.ndarray,
    N_values: list[int],
    output_path: Path,
    nrows: int,
    ncols: int,
    figsize: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = axes.ravel()

    for ax, N in zip(axes_flat, N_values):
        draw_projection_on_ax(ax, sample, N)

    for ax in axes_flat[len(N_values) :]:
        ax.axis("off")

    fig.suptitle("Проекционные оценки для различных N", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_projection_delta(proj_mean: dict[int, float], output_path: Path) -> None:
    best_N = min(proj_mean, key=proj_mean.get)
    xs = sorted(proj_mean.keys())
    ys = [proj_mean[N] for N in xs]

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(xs, ys, marker="o", linewidth=2, color="purple")
    plt.scatter([best_N], [proj_mean[best_N]], color="green", s=80, zorder=3, label=f"Минимум при N={best_N}")
    plt.title(r"Зависимость $\overline{\delta_n}(N)$ от числа членов разложения N")
    plt.xlabel("Число членов разложения N")
    plt.ylabel(r"$\overline{\delta_n}(N)$")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


# ============================================================
# Задача 4. Ядерная оценка с ядром Епанечникова
# ============================================================
SQRT5 = math.sqrt(5.0)
X_KDE_GRID = np.arange(0.0, L_KDE + DX_KDE, DX_KDE)
F_TRUE_KDE_GRID = weibull_pdf(X_KDE_GRID)



def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= SQRT5
    out[mask] = (3.0 / (4.0 * SQRT5)) * (1.0 - (u[mask] ** 2) / 5.0)
    return out



def convolve_same_length(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    full = np.convolve(a, b, mode="full")
    start = (len(full) - len(a)) // 2
    end = start + len(a)
    return full[start:end]



def kde_epanechnikov_grid(sample: np.ndarray, h: float, x_grid: np.ndarray = X_KDE_GRID, dx: float = DX_KDE) -> np.ndarray:
    edges = np.concatenate([x_grid - dx / 2.0, [x_grid[-1] + dx / 2.0]])
    counts, _ = np.histogram(sample, bins=edges)
    density_impulses = counts / (len(sample) * dx)

    half_width = int(np.ceil((SQRT5 * h) / dx))
    offsets = np.arange(-half_width, half_width + 1)
    z = offsets * dx
    kernel_vals = (1.0 / h) * epanechnikov_kernel(z / h)

    return convolve_same_length(density_impulses, kernel_vals) * dx



def kde_ise(
    sample: np.ndarray,
    h: float,
    x_grid: np.ndarray = X_KDE_GRID,
    f_true_grid: np.ndarray = F_TRUE_KDE_GRID,
    dx: float = DX_KDE,
) -> float:
    f_hat = kde_epanechnikov_grid(sample, h, x_grid=x_grid, dx=dx)
    return float(np.sum((f_hat - f_true_grid) ** 2) * dx)



def compute_kde_stats() -> tuple[dict[float, float], dict[float, float]]:
    rng = np.random.default_rng(24680)
    mean_vals: dict[float, float] = {}
    std_vals: dict[float, float] = {}

    for h in H_VALUES_KDE:
        vals = np.empty(R_KDE)
        for i in range(R_KDE):
            sample = sample_weibull_inverse_transform(N_SAMPLE, rng=rng)
            vals[i] = kde_ise(sample, float(h))
        hk = round(float(h), 3)
        mean_vals[hk] = float(vals.mean())
        std_vals[hk] = float(vals.std(ddof=1))

    return mean_vals, std_vals



def plot_kde_for_h(sample: np.ndarray, h: float, output_path: Path, title_suffix: str | None = None) -> None:
    f_hat = kde_epanechnikov_grid(sample, h)

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(X_KDE_GRID, F_TRUE_KDE_GRID, color="red", linewidth=2, label="Теоретическая f(x)")
    plt.plot(X_KDE_GRID, f_hat, color="blue", linestyle="--", linewidth=2, label=rf"Ядерная оценка, h={h:.2f}")
    plt.title(title_suffix or f"Ядерная оценка с ядром Епанечникова, h = {h:.2f}")
    plt.xlabel("x")
    plt.ylabel("Плотность")
    plt.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()



def draw_kde_on_ax(ax, sample: np.ndarray, h: float, title_suffix: str | None = None) -> None:
    f_hat = kde_epanechnikov_grid(sample, h)

    ax.plot(X_KDE_GRID, F_TRUE_KDE_GRID, color="red", linewidth=2, label="Теоретическая f(x)")
    ax.plot(X_KDE_GRID, f_hat, color="blue", linestyle="--", linewidth=2, label=rf"Ядерная оценка, h={h:.2f}")
    ax.set_title(title_suffix or f"h = {h:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("Плотность")
    ax.legend(loc="upper right", framealpha=0.95)


def plot_kde_grid(sample: np.ndarray, h_values: list[float], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    axes_flat = axes.ravel()

    for ax, h in zip(axes_flat, h_values):
        draw_kde_on_ax(ax, sample, h)

    for ax in axes_flat[len(h_values) :]:
        ax.axis("off")

    fig.suptitle("Ядерные оценки для различных h", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_kde_delta(kde_mean: dict[float, float], output_path: Path) -> None:
    best_h = min(kde_mean, key=kde_mean.get)
    xs = sorted(kde_mean.keys())
    ys = [kde_mean[h] for h in xs]

    plt.figure(figsize=(8.5, 4.8))
    plt.plot(xs, ys, marker="o", linewidth=2, color="purple")
    plt.scatter([best_h], [kde_mean[best_h]], color="green", s=80, zorder=3, label=f"Минимум при h={best_h:.2f}")
    plt.title(r"Зависимость $\overline{\delta_n}(h)$ от ширины окна h")
    plt.xlabel("Ширина окна h")
    plt.ylabel(r"$\overline{\delta_n}(h)$")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


# ============================================================
# Сохранение таблиц результатов
# ============================================================
def write_results_file(
    hist_mean: dict[int, float],
    hist_std: dict[int, float],
    proj_mean: dict[int, float],
    proj_std: dict[int, float],
    kde_mean: dict[float, float],
    kde_std: dict[float, float],
) -> None:
    lines: list[str] = []

    lines.append("Задача 2. Средние значения delta_bar_n(m)")
    for m in M_VALUES_HIST:
        lines.append(f"m={m:2d}: mean={hist_mean[m]:.6f}, std={hist_std[m]:.6f}")

    lines.append("")
    lines.append("Задача 3. Средние значения delta_bar_n(N)")
    for N in N_VALUES_PROJ:
        lines.append(f"N={N:3d}: mean={proj_mean[N]:.6f}, std={proj_std[N]:.6f}")

    lines.append("")
    lines.append("Задача 4. Средние значения delta_bar_n(h)")
    for h in sorted(kde_mean.keys()):
        lines.append(f"h={h:>4.2f}: mean={kde_mean[h]:.6f}, std={kde_std[h]:.6f}")

    RESULTS_FILE.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Полная генерация всех графиков
# ============================================================
def generate_all_figures() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    hist_mean, hist_std = compute_histogram_stats()
    proj_mean, proj_std = compute_projection_stats()
    kde_mean, kde_std = compute_kde_stats()

    # Одна и та же демонстрационная выборка внутри каждого задания
    rng_hist_plot = np.random.default_rng(20260413)
    sample_hist_plot = sample_weibull_inverse_transform(N_SAMPLE, rng=rng_hist_plot)
    plot_histogram_grid(
        sample_hist_plot,
        M_GROUPS_HIST[0],
        PLOTS_DIR / "task2_hist_grid_5_25_step5.png",
        "Гистограммные оценки для m = 5(5)25",
        nrows=3,
        ncols=2,
        figsize=(10.5, 12.0),
    )
    plot_histogram_grid(
        sample_hist_plot,
        M_GROUPS_HIST[1],
        PLOTS_DIR / "task2_hist_grid_10_40_step10.png",
        "Гистограммные оценки для m = 10(10)40",
        nrows=2,
        ncols=2,
        figsize=(10.5, 8.5),
    )
    plot_histogram_delta(hist_mean, PLOTS_DIR / "task2_delta_vs_m.png")

    rng_proj_plot = np.random.default_rng(20260414)
    sample_proj_plot = sample_weibull_inverse_transform(N_SAMPLE, rng=rng_proj_plot)
    plot_projection_grid(
        sample_proj_plot,
        N_VALUES_PROJ_SHOW,
        PLOTS_DIR / "task3_proj_grid.png",
        nrows=3,
        ncols=2,
        figsize=(10.5, 12.0),
    )
    plot_projection_delta(proj_mean, PLOTS_DIR / "task3_delta_vs_N.png")

    rng_kde_plot = np.random.default_rng(20260415)
    sample_kde_plot = sample_weibull_inverse_transform(N_SAMPLE, rng=rng_kde_plot)
    best_h = min(kde_mean, key=kde_mean.get)
    plot_kde_grid(sample_kde_plot, H_VALUES_KDE_SHOW, PLOTS_DIR / "task4_kde_grid.png")
    plot_kde_for_h(sample_kde_plot, best_h, PLOTS_DIR / "task4_kde_best_h.png", f"Ядерная оценка, h = {best_h:.2f}")
    plot_kde_delta(kde_mean, PLOTS_DIR / "task4_delta_vs_h.png")

    write_results_file(hist_mean, hist_std, proj_mean, proj_std, kde_mean, kde_std)


if __name__ == "__main__":
    generate_all_figures()
