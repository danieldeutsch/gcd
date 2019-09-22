import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple

plt.switch_backend('agg')


def get_train_percents(result_dir: str) -> List[str]:
    percents = []
    for name in os.listdir(result_dir):
        if os.path.isdir(os.path.join(result_dir, name)):
            percents.append(name)
    return sort_numerically(percents)


def get_metrics(output_dir: str, split: str) -> Tuple[Dict[str, float], ...]:
    unconstrained = json.load(open(os.path.join(output_dir, f'{split}.unconstrained.json')))
    constrained = json.load(open(os.path.join(output_dir, f'{split}.constrained.json')))
    post_hoc = json.load(open(os.path.join(output_dir, f'{split}.post-hoc.json')))
    return unconstrained, constrained, post_hoc


def sort_numerically(items: List[str]) -> List[str]:
    # Given a list of numbers as strings, sort them numerically
    # then return the numbers as strings again.
    return list(map(str, sorted(map(float, items))))


def plot_format_metrics(results: List[Tuple[float, Dict[str, float]]],
                        group: str,
                        plot_file_path: str,
                        table_file_path: str) -> None:
    data = [['', 'train-percent', 'metric', 'percent']]
    with open(table_file_path, 'w') as out:
        out.write('frac, balanced, non-empty-phrase, correct-num-preterminals, valid\n')
        for train_percent, metrics in results:
            valid_percent = metrics[group]['valid_percent']
            balanced_percent = 100 - metrics[group]['unbalanced_percent']
            nonempty_percent = 100 - metrics[group]['empty_phrase_percent']
            correct_num_preterm_percent = 100 - metrics[group]['incorrect_num_preterminal_percent']
            out.write(f'{train_percent * 100}, {balanced_percent}, {nonempty_percent}, {correct_num_preterm_percent}, {valid_percent}\n')

            data.append([
                f'Row{len(data)}',
                train_percent * 100,
                'valid',
                valid_percent
            ])
            data.append([
                f'Row{len(data)}',
                train_percent * 100,
                'balanced',
                balanced_percent
            ])
            data.append([
                f'Row{len(data)}',
                train_percent * 100,
                'non-empty-phrase',
                nonempty_percent
            ])
            data.append([
                f'Row{len(data)}',
                train_percent * 100,
                'correct-num-preterminals',
                correct_num_preterm_percent
            ])

    data = np.array(data)
    df = pd.DataFrame(data=data[1:, 1:],
                      index=data[1:, 0],
                      columns=data[0, 1:])
    df['train-percent'] = pd.to_numeric(df['train-percent'])
    df['percent'] = pd.to_numeric(df['percent'])

    plot = sns.lineplot(x='train-percent', y='percent',
                        hue='metric', data=df)
    plt.xlim(0, 100)
    plt.ylim(0, 110)
    dirname = os.path.dirname(plot_file_path)
    os.makedirs(dirname, exist_ok=True)
    plot.figure.savefig(plot_file_path)
    plt.clf()


def plot_f1_coverage(results: List[Tuple[float, Dict[str, float]]],
                     constrained_results: List[Tuple[float, Dict[str, float]]],
                     gfl_results: List[Tuple[float, Dict[str, float]]],
                     group: str,
                     plot_file_path: str,
                     table_file_path: str) -> None:
    data = [['', 'train-percent', 'method', 'metric', 'percent']]
    for method, res in zip(['unconstrained', 'constrained', 'gfl'], [results, constrained_results, gfl_results]):
        for train_percent, metrics in res:
            # If 'evalb' isn't in the output metrics, it's very likely
            # that the output was nonsense and it could not successfully run
            if group in metrics:
                if 'Bracketing FMeasure' not in metrics[group]:
                    f1 = 0.0
                else:
                    f1 = metrics[group]['Bracketing FMeasure']
                if math.isnan(f1):
                    f1 = 0.0

                if 'Number of Valid sentence' in metrics[group] and 'Number of sentence' in metrics[group]:
                    coverage = metrics[group]['Number of Valid sentence'] / metrics[group]['Number of sentence'] * 100
                else:
                    coverage = 0.0

                data.append([
                    f'Row{len(data)}',
                    train_percent * 100,
                    method,
                    'coverage',
                    coverage
                ])
                data.append([
                    f'Row{len(data)}',
                    train_percent * 100,
                    method,
                    'f1-evalb',
                    f1
                ])

    data = np.array(data)
    df = pd.DataFrame(data=data[1:, 1:],
                      index=data[1:, 0],
                      columns=data[0, 1:])
    df['train-percent'] = pd.to_numeric(df['train-percent'])
    df['percent'] = pd.to_numeric(df['percent'])

    plot = sns.lineplot(x='train-percent', y='percent',
                        hue='method', style='metric', data=df)
    plt.xlim(0, 100)
    plt.ylim(0, 110)
    dirname = os.path.dirname(plot_file_path)
    os.makedirs(dirname, exist_ok=True)
    plot.figure.savefig(plot_file_path)
    plt.clf()

    with open(table_file_path, 'w') as out:
        out.write('frac, un-f1, un-coverage, con-f1, con-coverage, gafl-f1, gafl-coverage\n')
        for train_percent in sorted(set(df['train-percent'])):
            un_f1 = df[(df['train-percent'] == train_percent) & (df['method'] == 'unconstrained') & (df['metric'] == 'f1-evalb')]['percent'].item()
            un_coverage = df[(df['train-percent'] == train_percent) & (df['method'] == 'unconstrained') & (df['metric'] == 'coverage')]['percent'].item()
            con_f1 = df[(df['train-percent'] == train_percent) & (df['method'] == 'constrained') & (df['metric'] == 'f1-evalb')]['percent'].item()
            con_coverage = df[(df['train-percent'] == train_percent) & (df['method'] == 'constrained') & (df['metric'] == 'coverage')]['percent'].item()
            gfl_f1 = df[(df['train-percent'] == train_percent) & (df['method'] == 'gfl') & (df['metric'] == 'f1-evalb')]['percent'].item()
            gfl_coverage = df[(df['train-percent'] == train_percent) & (df['method'] == 'gfl') & (df['metric'] == 'coverage')]['percent'].item()
            out.write(f'{train_percent}, {un_f1}, {un_coverage}, {con_f1}, {con_coverage}, {gfl_f1}, {gfl_coverage}\n')


def main():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    results_dir = os.path.join(root_dir, 'results')
    plots_dir = os.path.join(root_dir, 'plots')
    train_percents = get_train_percents(results_dir)

    valid_results, test_results = [], []
    valid_constrained_results, test_constrained_results = [], []
    valid_gfl_results, test_gfl_results = [], []
    for train_percent in train_percents:
        output_dir = os.path.join(results_dir, train_percent, 'metrics')
        # valid_unconstrained, valid_constrained, valid_post_hoc = get_metrics(output_dir, 'valid')
        test_unconstrained, test_constrained, test_post_hoc = get_metrics(output_dir, 'test')
        # valid_results.append((float(train_percent), valid_unconstrained))
        test_results.append((float(train_percent), test_unconstrained))
        # valid_constrained_results.append((float(train_percent), valid_constrained))
        test_constrained_results.append((float(train_percent), test_constrained))
        # valid_gfl_results.append((float(train_percent), valid_post_hoc))
        test_gfl_results.append((float(train_percent), test_post_hoc))

    # plot_format_metrics(valid_results, 'format', os.path.join(plots_dir, 'valid-format.png'), os.path.join(plots_dir, 'valid-format.csv'))
    # plot_format_metrics(valid_results, 'format30', os.path.join(plots_dir, 'valid-format-30.png'), os.path.join(plots_dir, 'valid-format-30.csv'))
    plot_format_metrics(test_results, 'format', os.path.join(plots_dir, 'test-format.png'), os.path.join(plots_dir, 'test-format.csv'))
    plot_format_metrics(test_results, 'format30', os.path.join(plots_dir, 'test-format-30.png'), os.path.join(plots_dir, 'test-format-30.csv'))

    # plot_f1_coverage(valid_results, valid_constrained_results, valid_gfl_results, 'evalb', os.path.join(plots_dir, 'valid-f1.png'), os.path.join(plots_dir, 'valid-f1.csv'))
    # plot_f1_coverage(valid_results, valid_constrained_results, valid_gfl_results, 'evalb30', os.path.join(plots_dir, 'valid-f1-30.png'), os.path.join(plots_dir, 'valid-f1-30.csv'))
    plot_f1_coverage(test_results, test_constrained_results, test_gfl_results, 'evalb', os.path.join(plots_dir, 'test-f1.png'), os.path.join(plots_dir, 'test-f1.csv'))
    plot_f1_coverage(test_results, test_constrained_results, test_gfl_results, 'evalb30', os.path.join(plots_dir, 'test-f1-30.png'), os.path.join(plots_dir, 'test-f1-30.csv'))


if __name__ == '__main__':
    main()
