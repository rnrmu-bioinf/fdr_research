import numpy as np
from scipy.stats import ttest_1samp, chi2


def compute_pvalues_from_data(data_matrix: np.ndarray) -> np.ndarray:
    """Вычисляет p-значения для каждого столбца матрицы данных.

    Для каждого признака (столбца) выполняется одновыборочный t-тест,
    проверяющий нулевую гипотезу о равенстве среднего значения нулю.

    Parameters
    ----------
    data_matrix : np.ndarray
        Двумерный массив формы `(n_samples, n_features)`.

    Returns
    -------
    np.ndarray
        Одномерный массив длины `n_features`, содержащий p-значения
        для каждого столбца.

    Notes
    -----
    Используется реализация `scipy.stats.ttest_1samp`. Предполагается,
    что наблюдения независимы и распределены приблизительно нормально.
    """
    _, n_features = data_matrix.shape
    p_values = np.empty(n_features, dtype=np.float64)

    for j in range(n_features):
        _, p_values[j] = ttest_1samp(data_matrix[:, j], 0)

    return p_values

import numpy as np


def get_rejected_hypothesis_by_fdr(
    p_values: np.ndarray, 
    q: float = 0.05
) -> tuple[np.ndarray, int]:
    """Определяет отвергаемые нулевые гипотезы методом Беньямини-Хохберга.

    Процедура контроля ожидаемой доли ложных отклонений (False Discovery Rate, FDR).
    Для заданного уровня `q` находит максимальное количество гипотез,
    которые могут быть отвергнуты при контролируемом FDR.

    Алгоритм:
        1. p-значения сортируются по возрастанию.
        2. Для каждого ранга i вычисляется порог q * i / N.
        3. Находится последний индекс k, где p_{(k)} ≤ q * k / N.
        4. Отвергаются все гипотезы с рангами 1..k.

    Parameters
    ----------
    p_values : np.ndarray
        Одномерный массив p-значений проверяемых гипотез.
        Должен содержать значения в диапазоне [0, 1].
    q : float, default=0.05
        Требуемый уровень контроля FDR. Должен лежать в интервале (0, 1).

    Returns
    -------
    rejected : np.ndarray
        Булев массив той же длины, что и `p_values`.
        `True` соответствует отвергнутым нулевым гипотезам.
    n_rejected : int
        Количество отвергнутых гипотез.

    Notes
    -----
    Предполагается независимость (или положительная регрессионная зависимость)
    тест-статистик. Для произвольной структуры зависимости используйте
    процедуру Беньямини-Йекутиели.

    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate:
    A Practical and Powerful Approach to Multiple Testing.
    Journal of the Royal Statistical Society. Series B, 57(1), 289-300.

    Examples
    --------
    >>> p = np.array([0.0001, 0.0004, 0.0016, 0.0053, 0.0098, 0.023, 0.041, 0.052])
    >>> rejected, n = get_rejected_hypothesis_by_fdr(p, q=0.05)
    >>> n
    6
    >>> rejected
    array([ True,  True,  True,  True,  True,  True, False, False])
    """
    assert 0 < q < 1, (
        "Некорректное критическое значение. Должно быть в интервале (0; 1)"
    )

    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Критические значения: q * i / N для i = 1..N
    critical_values = (np.arange(1, n + 1) / n) * q

    # Индикатор: p(i) <= q * i / N
    below = sorted_p <= critical_values

    if np.any(below):
        last_rejected = int(np.max(np.where(below)[0]))
        rejected_indices = sorted_indices[: last_rejected + 1]
        n_rejected = last_rejected + 1
    else:
        rejected_indices = np.array([], dtype=int)
        n_rejected = 0

    rejected = np.zeros(n, dtype=bool)
    rejected[rejected_indices] = True

    return rejected, n_rejected

def chi_square_uniformity_test(p_values: np.ndarray, bins: int = 10) -> tuple[float, float]:
    """Проверяет равномерность распределения p-значений критерием хи-квадрат.

    Тест используется для валидации метода множественного тестирования:
    при справедливости всех нулевых гипотез p-значения должны быть
    распределены равномерно на [0, 1]. Отклонение от равномерности
    указывает на некорректность модели или нарушение допущений.

    Parameters
    ----------
    p_values : np.ndarray
        Одномерный массив p-значений в диапазоне [0, 1].
    bins : int, default=10
        Количество интервалов разбиения. Рекомендуется выбирать так,
        чтобы ожидаемая частота в каждом интервале была не менее 5.

    Returns
    -------
    chi2_stat : float
        Значение статистики хи-квадрат.
    chi2_p : float
        P-значение для проверки нулевой гипотезы о равномерности.
        Малое значение (< 0.05) свидетельствует об отклонении от равномерности.

    Raises
    ------
    AssertionError
        Если ожидаемая частота хотя бы в одном интервале меньше 5,
        что делает результаты теста статистически некорректными. Или слишком малое число бинов.

    Notes
    -----
    Статистика критерия вычисляется как:

        χ² = Σ (O_i - E_i)² / E_i,

    где O_i — наблюдаемая, а E_i — ожидаемая частота в i-м интервале.
    Распределение статистики аппроксимируется χ² с (bins - 1) степенями свободы.
    P-значение вычисляется через функцию распределения хи-квадрат:
    p = 1 - CDF(χ², df).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> p_uniform = rng.uniform(0, 1, 1000)
    >>> stat, p_val = chi_square_uniformity_test(p_uniform, bins=10)
    >>> p_val > 0.05  # Не отвергаем равномерность
    True

    >>> p_skewed = rng.beta(0.5, 2, 1000)  # Смещённые к 0
    >>> _, p_val = chi_square_uniformity_test(p_skewed, bins=10)
    >>> p_val < 0.05  # Отвергаем равномерность
    True
    """
    assert bins > 1, "Слишком мало бинов"
    n = len(p_values)

    bin_edges = np.linspace(0, 1, bins + 1)
    observed, _ = np.histogram(p_values, bins=bin_edges)

    expected = np.full(bins, n / bins)

    assert np.all(expected >= 5), "Предупреждение: ожидаемая частота < 5. bins={bins}, n={n}. Тест некорректен."

    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    df = bins - 1
    chi2_p = float(1 - chi2.cdf(chi2_stat, df))

    return chi2_stat, chi2_p