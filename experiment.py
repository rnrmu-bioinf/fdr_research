
import os
import sys
import json
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
import traceback
from scipy import stats

from functools import partial
from src.generator import generate_correlated_data
from src.stat_tests import compute_pvalues_from_data, chi_square_uniformity_test, get_rejected_hypothesis_by_fdr
from src.visualisation import save_fdr_plot
from tqdm import tqdm
from pathlib import Path
from pprint import pprint, pformat
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


def check_config_types(config):
    assert type(config["n_iterations"]) is int, "Количество итераций должно быть натуральным числом."
    assert type(config["n_samples"]) is int, "Количество образцов должно быть натуральным числом."
    assert type(config["n_features"]) is int, "Количество переменных должно быть натуральным числом."
    assert type(config["correlation"]) is float or type(config["correlation"]) is int, "Обнаружены нечисловое значение коэф. корреляции."
    assert type(config["save_all_plots"]) is bool, "Некорректный тип параметра save_all_plots"
    assert type(config["seed"]) is int or config["seed"] is None, "Некорректное стартовое значение ГПСЧ."
    assert type(config["alpha"]) is float, "Ошибка 1-го рода имеет нечисловой тип."

def parse_config(path):
    parameters_names = ["n_iterations", "n_samples", "n_features", "correlation", "save_all_plots", "seed", "alpha"]
    with open(path, "r") as f:
        config = json.load(f)
    for name in parameters_names:
        assert name in config.keys(), f"Не найден обязательный параметр {name}"

    check_config_types(config)

    assert config["n_iterations"] > 0, "Количество итераций должно быть натуральным числом."
    assert config["n_samples"] > 0, "Количество образцов должно быть натуральным числом."
    assert config["n_features"] > 0, "Количество переменных должно быть натуральным числом."
    assert config["correlation"] >= -1 and config["correlation"] <= 1, "Коэффициент корреляции является некорректным значением."
    assert config["alpha"] > 0 and config["alpha"] < 1, "Ошибка 1-го рода является некорректным значением."

    if config["n_iterations"] > 10000:
        warnings.warn("Большое количество итераций.", UserWarning)

    if config["n_samples"] > 10000:
        warnings.warn("Большое количество образцов.", UserWarning)

    if config["n_features"] > 10000:
        warnings.warn("Большое количество переменных.", UserWarning)
    
    if config["alpha"] > 0.2:
        warnings.warn("Большое ошибка 1-го рода.", UserWarning)

    return config

def run_single_iteration(i, parameters, output_dirs):
    data = generate_correlated_data(
            parameters["n_samples"],
            parameters["n_features"],
            parameters["correlation"],
            parameters["seed"]
        )
        
    p_values = compute_pvalues_from_data(data)
    _, n_rejected = get_rejected_hypothesis_by_fdr(p_values, parameters["alpha"])
    ks_stat, ks_p = stats.kstest(p_values, 'uniform')
    chi2_stat, chi2_p = chi_square_uniformity_test(p_values, bins=10)

    if parameters["save_all_plots"]:
        plot_filename = f"{parameters["correlation"]}_{i+1:03d}.png"
        save_fdr_plot(
            p_values,
            n_rejected,
            alpha=parameters["alpha"],
            save_path=output_dirs["plots"],
            filename=plot_filename,
            title=f"ρ={parameters["correlation"]}, Итерация {i+1}",
            show_plot=False
        )

    iteration_stats = {
        'correlation': parameters["correlation"],
        'iteration': i + 1,
        'n': parameters["n_features"],
        'false_discoveries': n_rejected,
        'd_percentage': n_rejected / parameters["n_features"] * 100,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_p,
        'is_uniform_ks': ks_p > 0.05,
        'chi2_statistic': chi2_stat if not np.isnan(chi2_stat) else None,
        'chi2_pvalue': chi2_p if not np.isnan(chi2_p) else None,
        'is_uniform_chi2': chi2_p > 0.05 if not np.isnan(chi2_p) else None,
        'min_p': np.min(p_values),
        'max_p': np.max(p_values),
        'mean_p': np.mean(p_values),
        'median_p': np.median(p_values),
        'std_p': np.std(p_values),
        'skewness': stats.skew(p_values),
        'kurtosis': stats.kurtosis(p_values)
    }

    return iteration_stats

def experiment(parameters, output_dirs, n_jobs = 1):
    print("Параметры эксперимента:")
    pprint(parameters)
    logging.debug(pformat(parameters))
    print("Пути к файлам:")
    pprint(output_dirs)
    logging.debug(pformat(output_dirs))

    with open(output_dirs['basepath'] / Path("parameters.json"), 'w', encoding='utf-8') as f:
        json.dump(parameters, f, indent=4, ensure_ascii=False)

    statistics = []

    run_iter = partial(
        run_single_iteration,
        parameters=parameters,
        output_dirs=output_dirs,
    )
    
    statistics = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Отправляем все задачи
        futures = {
            executor.submit(run_iter, i): i 
            for i in range(parameters["n_iterations"])
        }
        
        iterator = tqdm(
            as_completed(futures),
            total=parameters["n_iterations"],
            desc=f"ρ={parameters['correlation']}"
        )
        
        for future in iterator:
            try:
                result = future.result()
                statistics.append(result)
            except Exception as e:
                iteration = futures[future]
                print(f"Ошибка в итерации {iteration}: {e}")
                logging.error(f"Ошибка в итерации {iteration}: {e}")
    
    statistics_table = pd.DataFrame(statistics).sort_values(by='iteration')
    statistics_table.to_csv(output_dirs['stats']/Path(f"statistics.csv"), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Запуск симуляции с конфигурацией из JSON-файла."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Путь к JSON-файлу с конфигурацией эксперимента."
    )
    parser.add_argument(
        "output_basepath",
        type=str,
        help="Путь к родительской папке, где следует сохранить результаты.",
        default=os.getcwd()
    )

    parser.add_argument(
        "-t", '--threads',
        type=int,
        help="Путь к родительской папке, где следует сохранить результаты.",
        default=1
    )

    args = parser.parse_args()

    if not 1 < args.threads < 2 * os.cpu_count() - 1:
        print(f"Слишком большое количество потоков.")
        sys.exit(1)

    try: 
        config = parse_config(args.config_path)
    except (FileNotFoundError, PermissionError) as fe:
        print(f"Указанный файл {args.config_path} не существует или не хватает прав для чтения.")
        sys.exit(13)
    except (json.decoder.JSONDecodeError, TypeError) as je:
        print(f"Указанный файл {args.config_path} не существует или не хватает прав для чтения.")
        sys.exit(2)
    except AssertionError as ase:
        print(f"В параметрах запуска эксперимента обнаружена ошибка: {ase}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"experiment_{timestamp}")
    
    
    output_dir = Path(args.output_basepath) / experiment_dir
    plots_dir = output_dir / Path("plots")
    stats_dir = output_dir / Path("statistics")
    summaries_dir = output_dir / Path("summaries")

    output_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)
    stats_dir.mkdir(exist_ok=True, parents=True)
    summaries_dir.mkdir(exist_ok=True, parents=True)

    output_dirs = {
        "basepath": output_dir, 
        "plots": plots_dir,
        "stats": stats_dir,
        "summaries": summaries_dir
    }

    logging.basicConfig(
        filename= output_dir / Path(f"{timestamp}.log"), 
        level=logging.DEBUG, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    try:
        experiment(config, output_dirs)
    except Exception as e:
        traceback.print_exc()
        logging.error(str(e))