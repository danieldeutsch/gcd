import argparse
import json
import sys
from typing import Dict, List, Tuple

from gcd.metrics import evalb


def _load_data(filename, field):
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data[field])
    return dataset


def filter_min_num_tokens(tokens: List[str],
                          gold: List[str],
                          model: List[str],
                          min_count: int) -> Tuple[List[str], List[str], List[str]]:
    tokens_filtered, gold_filtered, model_filtered = [], [], []
    for t, g, m in zip(tokens, gold, model):
        if len(t.split()) >= min_count:
            tokens_filtered.append(t)
            gold_filtered.append(g)
            model_filtered.append(m)
    return tokens_filtered, gold_filtered, model_filtered


def is_balanced(parse):
    nonterminals = parse.split()
    paren_count = 0
    # Make sure the phrase openings and closings match
    for nt in nonterminals:
        if nt.startswith('('):
            paren_count += 1
        elif nt.endswith(')'):
            if paren_count == 0:
                return False
            paren_count -= 1
        else:
            continue
    return True


def has_no_empty_phrases(parse):
    nonterminals = parse.split()
    for i in range(len(nonterminals) - 1):
        if nonterminals[i].startswith('(') and nonterminals[i + 1].endswith(')'):
            return False
    return True


def has_correct_num_preterminals(tokens, parse):
    preterminals = 0
    for nt in parse.split():
        if not nt.startswith('(') and not nt.endswith(')'):
            preterminals += 1
    return preterminals == len(tokens.split())


def full_accuracy(gold, model):
    assert len(gold) == len(model)
    total = len(gold)
    correct = sum([g == m for g, m in zip(gold, model)])
    return correct, total, correct / total * 100


def convert_to_normal_trees(tokens, tree):
    # Method assumes there are the same number of tokens and preterminals
    tokens = tokens.split()
    nonterminals = tree.split()
    index = 0
    converted = []
    for nt in nonterminals:
        if not nt.startswith('(') and not nt.startswith(')'):
            converted.append(f'({nt} {tokens[index]})')
            index += 1
        elif nt.endswith(')'):
            converted.append(')')
        else:
            converted.append(nt)
    return ' '.join(converted)


def save_as_normal_trees(tokens, trees, filename):
    with open(filename, 'w') as out:
        for token_str, tree in zip(tokens, trees):
            normal_tree = convert_to_normal_trees(token_str, tree)
            out.write(f'{normal_tree}\n')


def run_validity_metrics(gold_file: str,
                         model_file: str,
                         min_num_tokens: int = 0) -> Dict[str, float]:
    tokens = _load_data(gold_file, 'tokens')
    gold = _load_data(gold_file, 'parse')
    model = _load_data(model_file, 'prediction')

    tokens, gold, model = filter_min_num_tokens(tokens, gold, model, min_num_tokens)

    total, valid = 0, 0
    unbalanced, empty_phrase, preterm = 0, 0, 0
    for t, g, m in zip(tokens, gold, model):
        is_unbalanced = not is_balanced(m)
        has_empty = not has_no_empty_phrases(m)
        has_incorrect_preterm = not has_correct_num_preterminals(t, m)
        is_valid = not any([is_unbalanced, has_empty, has_incorrect_preterm])

        unbalanced += int(is_unbalanced)
        empty_phrase += int(has_empty)
        preterm += int(has_incorrect_preterm)
        valid += int(is_valid)
        total += 1

    valid_percent = valid / total * 100
    unbalanced_percent = unbalanced / total * 100
    empty_phrase_percent = empty_phrase / total * 100
    preterm_percent = preterm / total * 100
    full_correct, _, full_acc = full_accuracy(gold, model)

    return {
        'valid_num': valid,
        'valid_percent': valid_percent,
        'unbalanced_num': unbalanced,
        'unbalanced_percent': unbalanced_percent,
        'empty_phrase_num': empty_phrase,
        'empty_phrase_percent': empty_phrase_percent,
        'incorrect_num_preterminal_num': preterm,
        'incorrect_num_preterminal_percent': preterm_percent,
        'exact_match_num': full_correct,
        'exact_match_percent': full_acc
    }


def main(args):
    metrics = {}
    metrics['format'] = run_validity_metrics(args.gold, args.model)
    metrics['format30'] = run_validity_metrics(args.gold, args.model, min_num_tokens=30)
    try:
        metrics['evalb'] = evalb.run_evalb_on_files(args.gold, args.model)
    except Exception as e:
        sys.stderr.write('Exception during evalb\n')
        sys.stderr.write(str(e) + '\n')
    try:
        metrics['evalb30'] = evalb.run_evalb_on_files(args.gold, args.model, min_num_tokens=30)
    except Exception as e:
        sys.stderr.write('Exception during evalb30\n')
        sys.stderr.write(str(e) + '\n')

    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('gold')
    argp.add_argument('model')
    args = argp.parse_args()
    main(args)
