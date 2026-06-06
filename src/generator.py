import numpy as np

def generate_correlated_data(
    n_samples: int,
    n_features: int,
    r: float = 0,
    seed: int | None = None
) -> np.ndarray:
    """Генерирует матрицу данных с заданным уровнем попарной корреляции признаков.

    Создаёт двумерный массив формы `(n_samples, n_features)`, где каждый признак
    представляет собой случайную величину, имеющую коэффициент корреляции Пирсона
    приблизительно равный `r` с любым другим признаком. Данные моделируются как

        X_i = sqrt(r) * Z + sqrt(1 - r) * E_i,

    где Z ~ N(0,1) — общий фактор, а E_i ~ N(0,1) — индивидуальный шум.

    Parameters
    ----------
    n_samples : int
        Число наблюдений (строк).
    n_features : int
        Число признаков (столбцов).
    r : float, default=0
        Требуемый коэффициент корреляции. Должен лежать в диапазоне [-1, 1].
        При r=0 данные некоррелированы, при r=1 все признаки полностью совпадают,
        при r=-1 признаки противоположны друг другу.
    seed : int or None, default=None
        Зерно для генератора случайных чисел. Если None, используется системное время.

    Returns
    -------
    np.ndarray
        Массив формы `(n_samples, n_features)` сгенерированных данных.

    Raises
    ------
    AssertionError
        Если `r` вне [-1, 1], `n_samples <= 0` или `n_features <= 0`.
    """

    assert r >= -1 and r <= 1, "Недопустимое значение коэффициента корреляции. Должен быть в пределах [-1;1]"
    assert n_samples > 0, "Количество образцов должно быть > 0"
    assert n_features > 0, "Количество переменных должно быть > 0"

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    if r == 0.0:
        data = rng.normal(0, 1, (n_samples, n_features))
    else:
        common_factor = rng.normal(0, 1, n_samples)
        noise = rng.normal(0, 1, (n_samples, n_features))
        data = np.sqrt(r) * common_factor.reshape(-1, 1) + np.sqrt(1 - r) * noise

    return data

