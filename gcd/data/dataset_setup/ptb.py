import argparse
import json
import re
from nltk.corpus import ptb
from nltk.tree import ParentedTree
from tqdm import tqdm


# Some of the nonterminals are like 'PP-LOC'. If true, this will split
# the label to 'PP'
_split_tags = True

# Some of the nonterminals are '-NONE-', which means there is some word
# absent. This removes those subtrees.
_filter_none = True

# The Grammar as a Foreign Language paper replaced the POS tags in the
# training data to XX because POS tag accuracy is irrelevant to parsing F1.
_replace_pos_tags = True

# The figures in the paper look like it has a different closing parenthese for
# each nonterminal, like ")_NP" and ")_VP". If this variable is true, then these
# are merged into the same ")" token, otherwise they are different for each phrase
_merge_closing_paren = True


def get_fileids(min_section, max_section):
    for fileid in ptb.fileids():
        corpus, section, filename = fileid.split('/')
        if corpus == 'WSJ':
            section = int(section)
            if min_section <= section and section <= max_section:
                yield fileid


def get_tag(label):
    if not _split_tags:
        return label
    # ['-NONE-', '-LRB-', '-RRB-']
    if label.startswith('-'):
        return label
    return re.split('[-\|=]', label)[0]


def drop_none(tree):
    tree = ParentedTree.convert(tree)
    for sub in reversed(list(tree.subtrees())):
        if sub.label() == '-NONE-':
            parent = sub.parent()
            while parent and len(parent) == 1:
                sub = parent
                parent = sub.parent()
            del tree[sub.treeposition()]
    return tree


def flatten(tree, tokens, parse):
    tag = get_tag(tree.label())
    if tree.height() == 2:
        tokens.append(tree[0])
        if _replace_pos_tags:
            parse.append('XX')
        else:
            parse.append(tag)
    else:
        parse.append('(' + tag)
        for node in tree:
            flatten(node, tokens, parse)

        if _merge_closing_paren:
            parse.append(')')
        else:
            parse.append(tag + ')')


def sanity_checks(tokens, parse):
    # Make sure all tokens and nonterminals are not empty
    assert all(tokens) and all(parse)

    # Make sure that there's no empty constituents, like ['(NP', 'NP)']
    for i in range(len(parse) - 1):
        assert not (parse[i].startswith('(') and parse[i + 1].endswith(')'))

    # Make sure there's a preterminal for each token
    preterminals = 0
    for node in parse:
        if not node.startswith('(') and not node.endswith(')'):
            preterminals += 1
    assert len(tokens) == preterminals


def save(fileids, filename):
    with open(filename, 'w') as out:
        for fileid in tqdm(list(fileids)):
            for tree in ptb.parsed_sents(fileid):
                tokens, parse = [], []
                if _filter_none:
                    tree = drop_none(tree)
                flatten(tree, tokens, parse)
                sanity_checks(tokens, parse)
                data = {
                    'tokens': ' '.join(tokens),
                    'parse': ' '.join(parse)
                }
                out.write(json.dumps(data) + '\n')


def main(args):
    global _split_tags
    global _filter_none
    global _replace_pos_tags
    global _merge_closing_paren
    _split_tags = args.split_tags.lower() == 'true'
    _filter_none = args.filter_none.lower() == 'true'
    _replace_pos_tags = args.replace_pos_tags.lower() == 'true'
    _merge_closing_paren = args.merge_closing_paren.lower() == 'true'

    train_fileids = get_fileids(2, 21)
    valid_fileids = get_fileids(22, 22)
    test_fileids = get_fileids(23, 23)

    save(train_fileids, args.train_output)
    save(valid_fileids, args.valid_output)
    save(test_fileids, args.test_output)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('--train-output', required=True)
    argp.add_argument('--valid-output', required=True)
    argp.add_argument('--test-output', required=True)
    argp.add_argument('--split-tags', required=True)
    argp.add_argument('--filter-none', required=True)
    argp.add_argument('--replace-pos-tags', required=True)
    argp.add_argument('--merge-closing-paren', required=True)
    args = argp.parse_args()
    main(args)
