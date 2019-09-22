from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from typing import Dict

from gcd.inference.automata import Automaton, PDA
from gcd.inference.constraints.parsing.common import \
    EMPTY_STACK_OPEN_SYMBOL, EMPTY_STACK_CLOSE_SYMBOL, \
    OPEN_PAREN_SYMBOL, CLOSE_PAREN_SYMBOL

_STACK_SYMBOLS = [OPEN_PAREN_SYMBOL, CLOSE_PAREN_SYMBOL,
                  EMPTY_STACK_OPEN_SYMBOL, EMPTY_STACK_CLOSE_SYMBOL]


def set_symbol_table(automaton: Automaton,
                     token_to_key: Dict[str, int]) -> Dict[str, int]:
    """
    Sets the symbol table for the parsing constraints. The open and closing
    parens need to be included. If the automaton is a PDA, then they are
    also added to the parentheses table. The full symbol table is returned.
    """
    symbol_table = dict(token_to_key)

    max_key = max(symbol_table.values())
    open_key = max_key + 1
    close_key = max_key + 2
    empty_open_key = max_key + 3
    empty_close_key = max_key + 4
    symbol_table[OPEN_PAREN_SYMBOL] = open_key
    symbol_table[CLOSE_PAREN_SYMBOL] = close_key
    symbol_table[EMPTY_STACK_OPEN_SYMBOL] = empty_open_key
    symbol_table[EMPTY_STACK_CLOSE_SYMBOL] = empty_close_key

    for token, key in symbol_table.items():
        automaton.add_symbol(token, key)

    if isinstance(automaton, PDA):
        automaton.add_paren(open_key, close_key)
        automaton.add_paren(empty_open_key, empty_close_key)

    return symbol_table


def is_stack_token(token: str) -> bool:
    return token in _STACK_SYMBOLS


def is_token_preterminal(token: str) -> bool:
    if token in [START_SYMBOL, END_SYMBOL,
                 DEFAULT_PADDING_TOKEN, DEFAULT_OOV_TOKEN]:
        return False
    if token in _STACK_SYMBOLS:
        return False
    if token.startswith('(') or token.startswith(')'):
        return False
    return True


def is_token_open_paren(token: str) -> bool:
    return token.startswith('(')


def is_token_close_paren(token: str) -> bool:
    return token.endswith(')')
