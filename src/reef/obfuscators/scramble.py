from reef.obfuscators.base import Obfuscator
from reef.obfuscators.spacy_registry import get_spacy_nlp
from nltk.tokenize.treebank import TreebankWordDetokenizer
from spacy.tokens import Token
import random


def get_nested_dict_from_list(l: list[tuple]) -> dict:
    nested = {}
    for item in reversed(l):
        nested = {item: nested}
    return nested


def deep_update(main_dict, update_dict):
    for k, v in update_dict.items():
        if isinstance(v, dict) and isinstance(main_dict.get(k), dict):
            deep_update(main_dict[k], v)
        else:
            main_dict[k] = v


def linearise_sentence(tree, reverse=False):
    sentence = []

    for (word, direction), children in tree.items():
        left_children = {k: v for k, v in children.items() if k[1] == "L"}
        right_children = {k: v for k, v in children.items() if k[1] == "R"}

        if reverse:
            tmp = left_children
            left_children = right_children
            right_children = tmp

        sentence.extend(linearise_sentence(left_children))
        sentence.append(word)
        sentence.extend(linearise_sentence(right_children))

    return sentence


def shuffle_siblings(tree):
    shuffled_tree = {}

    for (word, direction), children in tree.items():
        shuffled_children = shuffle_siblings(children)
        shuffled_tree[(word, direction)] = shuffled_children

    l_siblings = [(k, v) for k, v in shuffled_tree.items() if k[1] == "L"]
    r_siblings = [(k, v) for k, v in shuffled_tree.items() if k[1] == "R"]

    random.shuffle(l_siblings)
    random.shuffle(r_siblings)

    return dict(l_siblings + r_siblings)


def swap_head_directions(tree):
    swapped_tree = {}

    for (word, direction), children in tree.items():
        new_direction = "L" if direction == "R" else "R"

        swapped_children = swap_head_directions(children)
        swapped_tree[(word, new_direction)] = swapped_children

    return swapped_tree


class ScrambleObfuscator(Obfuscator):
    def spacy_nlp(self, spacy_type: str = "full"):
        if not hasattr(self, "_spacy_nlp"):
            self._spacy_nlp = get_spacy_nlp(spacy_type)
        return self._spacy_nlp

    def obfuscate(self, text: str, algorithm: str = "linear", seed: int = 100) -> str:
        random.seed(seed)
        if algorithm == "linear":
            return self._linear_scramble(text)
        elif algorithm == "hierarchical":
            return self._hierarchical_scramble(text)
        else:
            raise ValueError("Invalid scramble algorithm.")

    def _linear_scramble(self, text: str) -> str:
        words = text.split()
        random.shuffle(words)
        scrambled_words = " ".join(words)
        return scrambled_words

    def _hierarchical_scramble(self, text: str, type: str = "shuffle-siblings") -> str:
        nlp = self.spacy_nlp(spacy_type="full")
        doc = nlp(text)

        d = {}
        for token in doc:
            print("Checking token ", token.text)
            path_to_root = self._get_route_to_root(token)
            print("====")
            print(path_to_root)
            print("====")
            d_from_l = get_nested_dict_from_list(path_to_root)
            deep_update(d, d_from_l)

        if type == "shuffle-siblings":
            shuffled = shuffle_siblings(d)
            linearised = linearise_sentence(shuffled)
        elif type == "head-direction":
            linearised = linearise_sentence(d, reverse=True)
        else:
            raise ValueError("Invalid scramble type.")

        linearised = TreebankWordDetokenizer().detokenize(linearised)
        return linearised

    def _get_route_to_root(
        self, token: Token, curr_pos: int = 0, curr_list: list[tuple] = []
    ) -> list[tuple]:
        """
        Find the path from a token to the eventual root of a sentence.
        This includes head information e.g. 'L' if a token is left-branching or 'R' if right-branching.
        E.g.
        """
        is_root = lambda token, head: token.text == head.text

        curr_head = token.head
        curr_at_root = is_root(token, curr_head)
        direction = "L" if token.i < token.head.i else "R"

        if curr_pos == 0:
            curr_list = []
        curr_list.insert(0, (token.text, direction))

        if curr_at_root:
            return curr_list
        else:
            return self._get_route_to_root(curr_head, curr_pos + 1, curr_list)
