import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from datetime import datetime
from pathlib import Path
from .stat_tests import get_rejected_hypothesis_by_fdr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def save_fdr_plot(
    p_values: np.ndarray,
    n_rejected: int | None = None,
    alpha: float = 0.05,
    save_path: Path = Path("fdr_plots"),
    filename: str | None = None,
    title: str | None = None,
    show_plot: bool = False,
) -> Path:
    """Сохраняет диагностический график процедуры Беньямини-Хохберга.

    Создаёт двухпанельную визуализацию результатов контроля FDR и сохраняет
    её в файл PNG с высоким разрешением. Левая панель — график Беньямини-Хохберга
    в логарифмических координатах (p(i) против i с линией порога FDR).
    Правая панель — гистограмма распределения p-значений с QQ-plot
    для проверки равномерности.

    Parameters
    ----------
    p_values : np.ndarray
        Одномерный массив p-значений проверяемых гипотез.
        Должен содержать значения в диапазоне [0, 1].
    n_rejected: int or None
        Предрасчитанное количество отклоненных гипотез.
    alpha : float, default=0.05
        Требуемый уровень контроля FDR. Должен лежать в интервале (0, 1).
    save_path : str, default="fdr_plots"
        Путь к директории для сохранения графика. Если директория
        не существует, она будет создана автоматически.
    filename : str or None, default=None
        Имя файла для сохранения (без расширения, добавляется .png).
        Если None, генерируется автоматически с временной меткой:
        ``fdr_plot_YYYYMMDD_HHMMSS.png``.
    title : str or None, default=None
        Заголовок графика. Если None, используется автоматически
        сгенерированный заголовок с параметрами N, d и α.
    show_plot : bool, default=False
        Если True, график отображается на экране после сохранения.

    Returns
    -------
    str
        Полный путь к сохранённому файлу PNG.

    Notes
    -----
    Левая панель («График FDR»):
        - Синие точки — log₁₀(p(i)) против log₁₀(ранга).
        - Красная линия — теоретический порог FDR: log₁₀(α · i / N).
        - Зелёная точка — последнее p-значение ниже линии БХ.
        - Фиолетовая точка и вертикальная линия — порог FDR и количество
          отвергнутых гипотез d.

    Правая панель («Гистограмма и QQ-plot»):
        - Гистограмма p-значений с наложенной линией равномерного
          распределения (y=1).
        - Встроенный QQ-plot равномерности для визуальной проверки
          отклонений от равномерного распределения.

    Requires
    --------
    - matplotlib
    - numpy
    - scipy.stats
    - mpl_toolkits.axes_grid1 (для встроенного QQ-plot)

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> # Преимущественно нулевые гипотезы (равномерные p-values)
    >>> p_uniform = rng.uniform(0, 1, 1000)
    >>> filepath = save_fdr_plot(p_uniform, alpha=0.05)
    >>> os.path.exists(filepath)
    True

    >>> # Смесь с сильными сигналами (много малых p-values)
    >>> p_mixed = np.concatenate([
    ...     rng.uniform(0, 0.01, 50),  # сигнал
    ...     rng.uniform(0, 1, 950),     # шум
    ... ])
    >>> filepath = save_fdr_plot(p_mixed, alpha=0.05,
    ...                          title="Анализ FDR: смесь сигнала и шума")
    """
    n = len(p_values)
    sorted_p = np.sort(p_values)
    ranks = np.arange(1, n + 1)

    # Линия FDR
    fdr_line = (alpha * ranks) / n

    # Применяем FDR
    if n_rejected is None:
        _, d = get_rejected_hypothesis_by_fdr(p_values, alpha)
    else:
        d = n_rejected

    # Создаем папку если нет
    parent_path = Path(save_path)
    parent_path.mkdir(exist_ok=True)

    # Генерируем имя файла если не задано
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(f"fdr_plot_{timestamp}.png")
    else:
        filename = Path(filename)

    # Создаем график
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ==================== ГРАФИК 1: p(i) против i (логарифмический) ====================
    log_p = np.log10(sorted_p + 1e-10)
    log_ranks = np.log10(ranks)
    log_fdr_line = np.log10(fdr_line)

    # Синие точки: все p-значения
    ax1.plot(log_ranks, log_p, 'b.', markersize=3, alpha=0.6, label='$p(i)$')

    # Красная линия: FDR
    ax1.plot(log_ranks, log_fdr_line, 'r-', linewidth=2,
             label=f'$q i / N$, $q={alpha}$')

    # Находим последнюю точку ниже линии
    below_line = sorted_p < fdr_line
    if np.any(below_line):
        last_below = np.max(np.where(below_line)[0])
        ax1.plot(log_ranks[last_below], log_p[last_below],
                'go', markersize=10, label='Последняя точка ниже линии')

    # Отмечаем порог FDR
    if d > 0:
        fdr_threshold_p = sorted_p[d-1]
        ax1.plot(log_ranks[d-1], log_p[d-1], 'mo', markersize=10,
                label=f'Порог FDR: i={d}, p={fdr_threshold_p:.4f}')

        # Вертикальная линия на количестве обнаружений
        ax1.axvline(x=np.log10(d), color='purple', linestyle=':',
                   label=f'Обнаружения (d) = {d}')

    ax1.set_xlabel('log₁₀(Ранг)')
    ax1.set_ylabel('log₁₀(p-значение)')
    if title:
        ax1.set_title(f'{title}\nВсего гипотез: {n}, Отвергнуто FDR: {d}')
    else:
        ax1.set_title(f'График FDR: N={n}, d={d}, α={alpha}')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ==================== ГРАФИК 2: Гистограмма и QQ-plot ====================
    # Гистограмма
    ax2_hist = ax2
    ax2_hist.hist(p_values, bins=30, alpha=0.7, color='skyblue',
                  edgecolor='black', density=True)
    ax2_hist.axhline(y=1, color='red', linestyle='--',
                     label=f'Равномерное распределение')
    ax2_hist.set_xlabel('p-value')
    ax2_hist.set_ylabel('Плотность')
    ax2_hist.set_title('Гистограмма p-значений')
    ax2_hist.legend()
    ax2_hist.grid(True, alpha=0.3)

    # QQ-plot на том же графике (инсет)
    ax_inset = inset_axes(ax2_hist, width="40%", height="40%", loc='upper right')
    stats.probplot(p_values, dist="uniform", plot=ax_inset)
    ax_inset.set_title('QQ-plot', fontsize=9)
    ax_inset.grid(True, alpha=0.3)

    fig.tight_layout()

    # Сохраняем график
    filepath = parent_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()

    return filepath