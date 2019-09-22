# flake8: noqa
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from glob import glob
from typing import Any, Dict


def load_results(output_dir: str) -> Dict[float, Dict[str, float]]:
    results = {}
    for percent in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        unconstrained_file = f'{output_dir}/dev-set.{percent}.FULL_false_BEAM_1.NODUP_false_LEGAL_false_ARGCANDS_false.metrics.json'
        nodup_file = f'{output_dir}/dev-set.{percent}.FULL_false_BEAM_1.NODUP_true_LEGAL_false_ARGCANDS_false.metrics.json'
        legal_args_file = f'{output_dir}/dev-set.{percent}.FULL_false_BEAM_1.NODUP_true_LEGAL_true_ARGCANDS_false.metrics.json'
        arg_cands_file = f'{output_dir}/dev-set.{percent}.FULL_false_BEAM_1.NODUP_true_LEGAL_true_ARGCANDS_true.metrics.json'

        results[percent] = {
            'unconstrained': json.load(open(unconstrained_file, 'r')),
            '+nodup': json.load(open(nodup_file, 'r')),
            '+legalargs': json.load(open(legal_args_file, 'r')),
            '+argcands': json.load(open(arg_cands_file, 'r'))
        }
    return results


def plot_f1(results: Dict[float, Any],
            plot_file: str,
            csv_file: str) -> None:

    plt.figure()
    plt.title('Constraints Learning Curve')
    plt.xlabel('Training Percent')
    plt.ylabel('CoNLL F1')
    plt.grid()
    percents = list(sorted(results.keys()))
    for model in ['unconstrained', '+nodup', '+legalargs', '+argcands']:
        f1s = []
        for percent in percents:
            f1s.append(results[percent][model]['f1']['conll_f1'])
        plt.plot(percents, f1s, label=model)
    plt.legend()
    plt.savefig(plot_file, dpi=400)

    with open(csv_file, 'w') as out:
        out.write('frac, unconstrained, +nodup, +legal_args, constrained\n')
        for percent in percents:
            row = [str(percent * 100)]
            for model in ['unconstrained', '+nodup', '+legalargs', '+argcands']:
                row.append(str(results[percent][model]['f1']['conll_f1']))
            out.write(', '.join(row) + '\n')


def write_active_set(results: Dict[float, Any],
                     csv_file: str) -> None:
    percents = list(sorted(results.keys()))
    with open(csv_file, 'w') as out:
        out.write('frac, viol0_pc, viol1_pc, viol2_pc, viol3_pc, viol4_pc, viol5_pc, viol6_pc\n')
        for percent in percents:
            row = [str(int(percent * 100))]
            for size in [0, 1, 2, 3, 4, 5, 6]:
                size = str(size)
                if size in results[percent]['+argcands']['active_set']:
                    row.append(str(results[percent]['+argcands']['active_set'][size] * 100))
                else:
                    row.append('0.0')
            out.write(', '.join(row) + '\n')


def main():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    output_dir = f'{root_dir}/output'
    plots_dir = os.path.join(root_dir, 'plots')

    results = load_results(output_dir)

    os.makedirs(plots_dir, exist_ok=True)
    # plot_f1(results, f'{plots_dir}/f1.png', f'{plots_dir}/f1.csv')
    write_active_set(results, f'{plots_dir}/active_set.csv')


if __name__ == '__main__':
    main()
