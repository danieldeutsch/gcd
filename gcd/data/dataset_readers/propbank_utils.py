from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
from collections import namedtuple, Counter
from typing import List

from nltk.corpus import PropbankCorpusReader

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def _load_data(filename, field=None):
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            if field is not None:
                dataset.append(data[field])
            else:
                dataset.append(data)
    return dataset


def compute_intersection():
    conll2004 = _load_data(filename="conll2004.train.jsonl")
    conll2005 = _load_data(filename="conll2005.train.jsonl")
    conll2004_dict = {}
    for prop in conll2004:
        key = " ".join(prop["words"]) + " " + str(prop["target_verb_position"])
        conll2004_dict[key] = prop

    seen = set([])
    for prop in conll2005:
        key = " ".join(prop["words"]) + " " + str(prop["target_verb_position"])
        if key in conll2004_dict and key not in seen:
            seen.add(key)
    print(len(seen))


Arg = namedtuple('Argument', ['descr', 'func', 'num'])


class PropbankFramesReader:
    """
    From the propbank-frames/frames/frameset.dtd file:

    roles have a number (or an "M" associated
    with them, for common adjuncts that don't qualify for number argument status).
    Both numbered arguments and adjuncts are labeled with the function tags from the list below

    EXT  extent
    LOC  location
    DIR  direction
    NEG  negation  (not in PREDITOR)
    MOD  general modification
    ADV  adverbial modification
    MNR  manner
    PRD  secondary predication
    REC  recipricol (eg herself, etc)
    TMP  temporal
    PRP  purpose
    PNC  purpose no cause (no longer used)
    CAU  cause
    CXN  constructional pattern (adjectival comparative marker)
    ADJ  adjectival (nouns only)
    COM  comitative
    DIS  discourse
    DSP  direct speech
    GOL  goal
    PAG  prototypical agent (function tag for arg1)
    PPT  prototypical patient (function tag for arg1)
    RCL  relative clause (no longer used)
    SLC  selectional constraint link
    VSP  verb specific (function tag for numbered arguments)
    LVB  light verb (for nouns only)
    """

    def __init__(self, root):
        framefiles = os.listdir(root + "/frames/")
        framefiles = ["frames/" + framefile for framefile in framefiles if not framefile.endswith("frameset.dtd")]
        # print(framefiles)
        self.reader = PropbankCorpusReader(root=root, propfile="",
                                           framefiles=framefiles)

        self.available_verbs: List[str] = []
        for framefile in framefiles:
            baseform = os.path.basename(framefile).split(".")[0]
            # print("baseform", baseform)
            self.available_verbs.append(baseform)
        self.lemma_id2role = {}

    def load_all(self):
        cnt = Counter()
        for verb in self.available_verbs:
            for role in self.reader.rolesets(baseform=verb):
                role_id, role_name = role.get("id"), role.get("name")
                # print(f"role.id {role_id} role.name: {role_name}")
                role_lemma, role_sense = role_id.split(".")
                self.lemma_id2role[(role_lemma, role_sense)] = role
                legal_args = self._get_legal_args_from_role(role=role)
                cnt.update(legal_args)
        # print(cnt)
        # return legal_args

    def get_legal_args(self, verb_lemma: str, sense_id: str) -> List[str]:
        role = self.lemma_id2role[(verb_lemma,sense_id)]
        legal_args = self._get_legal_args_from_role(role)
        return legal_args

    def get_verb_roles(self, verb: str) -> List[str]:
        roles = []

        return roles

    def _get_legal_args_from_role(self, role):
        arguments = []
        for arg in role.findall("roles/role"):
            # print(dir)
            # print(arg.get("descr"), arg.get("f"), arg.get("n"))
            # Sometimes arg.num is either Am or AM. We keep AM
            arguments.append(Arg(arg.get("descr"), arg.get("f"), arg.get("n").upper()))
        legal_args = ["A" + arg.num for arg in arguments]
        return legal_args


if __name__ == "__main__":
    root = "/home1/s/shyamupa/propbank-frames"
    reader = PropbankFramesReader(root=root)
    reader.load_all()
    # for verb in reader.available_verbs:
    #     roles = reader.get_verb_roles(verb)
    #     for verb_lemma, sense_id in roles:
    #         legal_args = reader.get_legal_args(verb_lemma=verb, sense_id=sense_id)
    #         print(legal_args)
    #     cnt.update()
    # parser = argparse.ArgumentParser(description='Short sample app')
    # parser.add_argument('--flag', action="store_true", default=False)
    # parser.add_argument('--path', action="store", dest="b")
    # parser.add_argument('--drop', required=False, action="store", dest="c", type=int)
    # parser.add_argument('--write', action="store_true", dest="write")
    # parser.add_argument('--nolog', action="store_true", default=False)
    # parser.set_defaults(write=True)
    # args = parser.parse_args()
    # args = vars(args)
    # print(args)
    # if args["nolog"]:
    #     logging.disable(logging.DEBUG)
    # logging.debug("unimportant debug msg")
    # logging.info("important msg")
