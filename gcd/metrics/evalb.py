import argparse
import json
import subprocess
import sys
import tempfile
from nltk.tree import Tree
from typing import Dict, List, Tuple


def load_parse_strings_from_jsonl(filename: str, field: str) -> List[str]:
    parses = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            parses.append(data[field])
    return parses


def select_long_enough_inputs(gold_parses: List[str],
                              model_parses: List[str],
                              min_length: int) -> Tuple[List[str], List[str]]:
    gold_filter, model_filter = [], []
    for gold, model in zip(gold_parses, model_parses):
        tree = Tree.fromstring(gold)
        num_tokens = len(tree.leaves())
        if num_tokens >= min_length:
            gold_filter.append(gold)
            model_filter.append(model)
    return gold_filter, model_filter


def add_terminals(parse: str) -> str:
    tokens = []
    for token in parse.split():
        if not token.startswith('(') and not token.endswith(')'):
            tokens.append(f'({token} a)')
        else:
            tokens.append(token)
    return ' '.join(tokens)


def remap_close_parens(parse: str) -> str:
    tokens = []
    for token in parse.split():
        if token.endswith(')'):
            tokens.append(')')
        else:
            tokens.append(token)
    return ' '.join(tokens)


def reformat_parses(gold_parses: List[str],
                    model_parses: List[str]) -> Tuple[List[str], List[str]]:
    # First, add a dummy terminal token
    gold_parses = [add_terminals(parse) for parse in gold_parses]
    model_parses = [add_terminals(parse) for parse in model_parses]

    # Then remap tokens like "NP)" to ")"
    # gold_parses = [remap_close_parens(parse) for parse in gold_parses]
    # model_parses = [remap_close_parens(parse) for parse in model_parses]

    return gold_parses, model_parses


def write_parses_to_file(filename: str, parses: List[str]) -> None:
    with open(filename, 'w') as out:
        for parse in parses:
            out.write(parse + '\n')


def run_evalb(gold_file: str, model_file: str, num_parses: int) -> str:
    # One parse could generate multiple errors, so we set this to some
    # very high number to minimize the chances of evalb crashing. If this number
    # gets too large, evalb will crash for some reason
    num_allowed_errors = int(num_parses * 1e5)
    param_file = 'lib/EVALB/new.prm'
    command = f'lib/EVALB/evalb -p {param_file} -e {num_allowed_errors} {gold_file} {model_file}'.split()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stderr:
        sys.stderr.write(stderr.decode() + '\n')
    return stdout


def parse_metric(line: str) -> Tuple[str, float]:
    name, value = line.split('=')
    name, value = name.strip(), float(value.strip())
    return name, value


def parse_output(output: str) -> Dict[str, float]:
    metrics = {}
    lines = output.split('\n')
    try:
        start_index = lines.index('-- All --')
        for i in range(12):
            name, value = parse_metric(lines[start_index + 1 + i])
            metrics[name] = value
        return metrics
    except ValueError:
        return {}


def run_evalb_on_files(gold_file: str,
                       model_file: str,
                       min_num_tokens: int = 0) -> Dict[str, float]:
    gold_parses = load_parse_strings_from_jsonl(gold_file, 'parse')
    model_parses = load_parse_strings_from_jsonl(model_file, 'prediction')

    gold_parses, model_parses = reformat_parses(gold_parses, model_parses)
    gold_parses, model_parses = select_long_enough_inputs(gold_parses, model_parses, min_num_tokens)

    with tempfile.NamedTemporaryFile() as gold_temp:
        with tempfile.NamedTemporaryFile() as model_temp:
            gold_file = gold_temp.name
            model_file = model_temp.name
            write_parses_to_file(gold_file, gold_parses)
            write_parses_to_file(model_file, model_parses)

            output = run_evalb(gold_file, model_file, len(gold_parses))
            output = output.decode()
            metrics = parse_output(output)
            return metrics


def main(args):
    metrics = run_evalb_on_files(args.gold, args.model)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('gold')
    argp.add_argument('model')
    args = argp.parse_args()
    main(args)
