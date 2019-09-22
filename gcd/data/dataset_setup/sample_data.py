import argparse
import random
import os


def main(args):
    random.seed(args.seed)

    data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            data.append(line)

    num_instances = int(len(data) * args.sample_percent)
    random.shuffle(data)
    sample = data[:num_instances]

    dirname = os.path.dirname(args.output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(args.output_file, 'w') as out:
        for line in sample:
            out.write(line)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('input_file', help='The file to sample')
    argp.add_argument('sample_percent', type=float, help='The amount to sample between 0.0 and 1.0')
    argp.add_argument('output_file', help='The output sample file')
    argp.add_argument('--seed', type=int, help='The random seed', default=4)
    args = argp.parse_args()
    main(args)
