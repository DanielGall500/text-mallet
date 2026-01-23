from tmallet.obfuscators.base import Obfuscator, SpaCyObfuscator
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize
from spacy.tokens import Token, Doc
from typing import Dict
import random


# the linear scrambler does not use SpaCy
class LinearScrambleObfuscator(Obfuscator):
    def obfuscate(
        self,
        text: str,
        config: Dict = {"scramble_within_sentence": False},
        seed: int = 100,
    ) -> str:
        random.seed(seed)

        if "scramble_within_sentence" not in config.keys():
            raise ValueError(
                "Please pass a configuration with the 'algorithm' parameter."
            )

        scramble_within_sentence = config["scramble_within_sentence"]
        if scramble_within_sentence:
            sentences = sent_tokenize(text)
            scrambled_sentences = [self._linear_scramble(s) for s in sentences]
            return " ".join(scrambled_sentences)
        else:
            return self._linear_scramble(text)

    def _linear_scramble(self, text) -> str:
        words = text.split()
        random.shuffle(words)
        scrambled_words = " ".join(words)
        return scrambled_words


class HierarchicalScrambleObfuscator(SpaCyObfuscator):
    def obfuscate(
        self,
        doc: Doc,
        config: Dict = {"algorithm": "scramble-shuffle-siblings"},
        seed: int = 100,
    ) -> str:
        random.seed(seed)

        if "algorithm" not in config.keys():
            raise ValueError(
                "Please pass a configuration with the 'algorithm' parameter."
            )

        algorithm = config["algorithm"]

        if algorithm == "scramble-shuffle-siblings":
            return self._hierarchical_scramble(doc, algorithm=algorithm)
        elif algorithm == "scramble-reverse-head":
            return self._hierarchical_scramble(doc, algorithm=algorithm)
        else:
            raise ValueError(
                "Please provide a valid hierarchical scrambling algorithm."
            )

    def _hierarchical_scramble(
        self, doc: Doc, algorithm: str = "shuffle-siblings"
    ) -> str:
        d = {}
        for token in doc:
            path_to_root = self._get_route_to_root(token)
            d_from_l = get_nested_dict_from_list(path_to_root)
            deep_update(d, d_from_l)

        if algorithm == "shuffle-siblings":
            shuffled = shuffle_siblings(d)
            linearised = linearise_sentence(shuffled)
        elif algorithm == "reverse-head-direction":
            swapped = swap_head_directions(d, swap_probability=0.7)
            linearised = linearise_sentence(swapped, reverse=True)

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


def swap_head_directions(tree, swap_probability: float = 1.0):
    swapped_tree = {}
    for (word, direction), children in tree.items():
        # Randomly decide whether to swap this node's direction
        if random.random() < swap_probability:
            new_direction = "L" if direction == "R" else "R"
        else:
            new_direction = direction

        swapped_children = swap_head_directions(children, swap_probability)
        swapped_tree[(word, new_direction)] = swapped_children
    return swapped_tree


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
