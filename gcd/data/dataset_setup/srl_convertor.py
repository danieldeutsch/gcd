import argparse
import json

from gcd.data.dataset_readers.srl import SRL_CONLL05Reader


def dump_instances(instances, outfile):
    with open(outfile, "w") as output:
        for instance in instances:
            line = {}
            # this field is only for evaluation purposes bcoz write_to_conll_eval_file requires gold and pred tags.
            line["tags"] = instance["tags"].labels
            metadata = instance["metadata"]
            # Get everything out from the metadata.
            # Write the metadata field such that it has all arguments of text_to_instance method
            line["words"] = metadata["words"]
            line["pos_tags"] = metadata["pos_tags"]
            line["chunk_tags"] = metadata["chunk_tags"]
            line["ner_tags"] = metadata["ner_tags"]
            line["target_verb_lemma"] = metadata["target_verb_lemma"]
            line["target_verb_position"] = metadata["target_verb_position"]
            line["verb_annotation"] = metadata["verb_annotation"]
            line["verb_sense"] = metadata["verb_sense"]
            line["legal_args"] = metadata["legal_args"]
            # this so that the code works even when parse info is not available (during training)
            line["parse"] = ""
            output.write(json.dumps(line) + "\n")


def convert_conll_to_jsonl(gold_file: str, core_args_only: bool, scheme: str = "BIO"):
    conll_reader = SRL_CONLL05Reader(coding_scheme=scheme, core_args_only=core_args_only, tag_label="srl", year='2005')
    instances = conll_reader.read(gold_file)
    outfile = gold_file + ".jsonl"
    dump_instances(instances, outfile)


def main(args):
    convert_conll_to_jsonl(gold_file=args.gold_file, scheme=args.scheme, core_args_only=args.core_args_only)


if __name__ == '__main__':
    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--gold_file', type=str, required=True, help='gold file')
    argp.add_argument('--silent', action="store_true", help='no verbose output')
    argp.add_argument('--scheme', type=str, default="BIO", help="BIO or BIOUL")
    argp.add_argument('--core_args_only', action="store_true", help="only keep A0 to A5")
    args = argp.parse_args()
    main(args)
