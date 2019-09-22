from typing import Dict

from gcd.inference.automata import Automaton


def set_symbol_table(automaton: Automaton,
                     token_to_key: Dict[str, int]) -> Dict[str, int]:
    """
    Sets the symbol table for the parsing constraints. The open and closing
    parens need to be included. If the automaton is a PDA, then they are
    also added to the parentheses table. The full symbol table is returned.
    """
    symbol_table = dict(token_to_key)
    for token, key in symbol_table.items():
        automaton.add_symbol(token, key)
    return symbol_table
