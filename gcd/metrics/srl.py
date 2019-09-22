import argparse
import json
import os
import tempfile
from allennlp.models.semantic_role_labeler import write_to_conll_eval_file
from collections import defaultdict


def load_srl_data(filename, field=None):
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            if field is not None:
                dataset.append(data[field])
            else:
                dataset.append(data)
    return dataset


def srl_conll_evaluate(pred_file: str, gold_file: str, silent: bool = False, min_length=0):
    """
    Evaluate current model using CoNLL script.

      Args:
        preds: contains the predictions from the model.
      Returns:
        f1 score
        :param silent:
        :param gold_file:
        :param pred_file:

    """
    pred_data = load_srl_data(pred_file)
    gold_data = load_srl_data(gold_file)
    if min_length > 0:
        pred_conll_file = pred_file + ".conll_" + str(min_length)
        gold_conll_file = gold_file + ".conll_" + str(min_length)
    else:
        pred_conll_file = pred_file + ".conll"
        gold_conll_file = gold_file + ".conll"
    assert len(pred_data) == len(gold_data)
    with open(pred_conll_file, mode='w') as pred_conll, open(gold_conll_file, mode='w') as gold_conll:
        for gold, pred in zip(gold_data, pred_data):
            # fields = instance.fields
            try:
                # Most sentences have a verbal predicate, but not all.
                verb_index = gold["target_verb_position"]
            except ValueError:
                verb_index = None

            gold_tags = gold["tags"]
            pred_tags = pred["tags"]
            sentence = gold["words"]
            if min_length > 0:
                if len(sentence) < min_length:
                    continue
            write_to_conll_eval_file(pred_conll, gold_conll,
                                     verb_index, sentence, pred_tags, gold_tags)

    with tempfile.NamedTemporaryFile(mode='r', delete=True) as scores:
        eval_script = "gcd/metrics/srl_perl/bin/srl-eval.pl"
        eval_lib = "gcd/metrics/srl_perl/lib"
        scores_path = scores.name
        # command = f"perl -I {eval_lib} {eval_script} {gold_conll_file} {pred_conll_file} > {scores_path}"
        command = "perl -I %s %s %s %s > %s" % (eval_lib, eval_script, gold_conll_file, pred_conll_file, scores_path)
        # print("running", command)
        os.system(command)
        result = scores.read().split('\n')
        # print(result)
        if not silent:
            for r in result:
                print(r)
        """
        Number of Sentences    :        3248
        Number of Propositions :        3221
        Percentage of perfect props :  68.89

                      corr.  excess  missed    prec.    rec.      F1
        ------------------------------------------------------------
           Overall     4810     997    1081    82.83   81.65   82.24
        ----------
                A0     1803     287     268    86.27   87.06   86.66
                A1     2448     521     525    82.45   82.34   82.40
                A2      450     163     218    73.41   67.37   70.26
                A3       67      16      45    80.72   59.82   68.72
                A4       41      10      24    80.39   63.08   70.69
                A5        1       0       1   100.00   50.00   66.67
        ------------------------------------------------------------
        ------------------------------------------------------------
        """
        conll_f1 = float(result[6].strip().split()[-1])
        perfect_props_percent = float(result[2].strip().split(':')[-1])
        label_f1s = {}
        for r in result[8:]:
            try:
                label_f1 = float(r.strip().split()[-1])
                label = r.strip().split()[0]
                label_f1s[label] = label_f1
            except ValueError:
                break
        return {
            "conll_f1": conll_f1,
            "perfect_props_percent": perfect_props_percent,
            "label_f1s": label_f1s
        }


def count_active_set_size_distribution(pred_file: str):
    counter = defaultdict(int)
    with open(pred_file, 'r') as f:
        for line in f:
            instance = json.loads(line)
            size = len(instance['working_set'])
            counter[size] += 1

    total = sum(counter.values())
    distribution = {size: value / total for size, value in counter.items()}
    return distribution


def main(args):
    assert not args.silent or args.output_file is not None
    conll_f1 = srl_conll_evaluate(gold_file=args.gold,
                                  pred_file=args.model,
                                  silent=args.silent)

    active_set_distribution = count_active_set_size_distribution(args.model)
    metrics = {
        'f1': conll_f1,
        'active_set': active_set_distribution
    }
    if not args.silent:
        print(json.dumps(metrics, indent=2))
    if args.output_file is not None:
        dirname = os.path.dirname(args.output_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(args.output_file, 'w') as out:
            out.write(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--model', type=str, required=True, help='model output jsonl')
    argp.add_argument('--gold', type=str, required=True, help='gold jsonl')
    argp.add_argument('--silent', action="store_true", help='no verbose output')
    argp.add_argument('--scheme', type=str, default="bio", help="bio or biolu")
    argp.add_argument('--output-file', type=str, help='The output file to save the metrics to')
    args = argp.parse_args()
    main(args)
