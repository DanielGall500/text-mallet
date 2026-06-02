import random

from sacremoses import MosesDetokenizer
from spacy.tokens import Doc, Token

from tmallet.obfuscators.base import SpaCyObfuscator
from tmallet.obfuscators.structural.config import HierarchicalScrambleConfig

detokenizer = MosesDetokenizer()


class HierarchicalScrambleObfuscator(SpaCyObfuscator):
    """
    Obfuscates text by scrambling the structure based on dependency parsing information.

    Traverses the dependency tree of a sentence and scrambles the
    order of tokens at different levels of the hierarchy (left and right branches).
    It supports different levels of scrambling strength ("weak" or "strong").
    """

    def set_config(self, config: HierarchicalScrambleConfig):
        random.seed(config.seed)
        self.strength = (
            config.strength if isinstance(config.strength, list) else [config.strength]
        )
        self.is_single_obfuscation: bool = len(self.strength) == 1

    def obfuscate(
        self,
        doc: Doc,
    ) -> dict | str:
        """
        Main obfuscation entry point.

        Determines whether to run single or multiple obfuscation passes based on
        the configured strength(s).

        Args:
            doc: The spaCy Doc object to obfuscate.

        Returns:
            A string if only one strength is configured, otherwise a dictionary
            containing the results for all configured strengths.
        """
        if self.is_single_obfuscation:
            return self._obfuscate_single(doc)
        else:
            return self._obfuscate_multi(doc)

    def _obfuscate_single(self, doc: Doc) -> str:
        doc_as_sents = [sent.as_doc() for sent in doc.sents]
        strength = self.strength[0]
        scrambled_sentences = [
            self._hierarchical_scramble(d, strength=strength) for d in doc_as_sents
        ]
        scrambled = " ".join(scrambled_sentences).strip()
        return scrambled

    def _obfuscate_multi(self, doc: Doc) -> dict:
        doc_as_sents = [sent.as_doc() for sent in doc.sents]
        result = {}
        for s in self.strength:
            scrambled_sentences = [
                self._hierarchical_scramble(d, strength=s) for d in doc_as_sents
            ]
            scrambled = " ".join(scrambled_sentences).strip()
            result[s] = scrambled
        return {"scramble-hier": result}

    def _hierarchical_scramble(self, doc: Doc, strength: str) -> str:
        d = {}
        for token in doc:
            path_to_root = self._get_route_to_root(token)
            d_from_l = get_nested_dict_from_list(path_to_root)
            deep_update(d, d_from_l)

        if strength == "weak":
            shuffled = scramble_hier_weak(d)
            linearised = linearise_sentence(shuffled)
        elif strength == "strong":
            swapped = scramble_hier_strong(d, flip_prob=0.5)
            linearised = linearise_sentence(swapped, reverse=True)
        else:
            raise ValueError(f"Invalid hierarchical scramble strength: {strength}.")

        linearised = detokenizer.detokenize(linearised)
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


def scramble_hier_weak(tree):
    shuffled_tree = {}

    for (word, direction), children in tree.items():
        shuffled_children = scramble_hier_weak(children)
        shuffled_tree[(word, direction)] = shuffled_children

    l_siblings = [(k, v) for k, v in shuffled_tree.items() if k[1] == "L"]
    r_siblings = [(k, v) for k, v in shuffled_tree.items() if k[1] == "R"]

    random.shuffle(l_siblings)
    random.shuffle(r_siblings)

    return dict(l_siblings + r_siblings)


def scramble_hier_strong(tree, flip_prob=0.5):
    shuffled_tree = {}
    for (word, direction), children in tree.items():
        shuffled_children = scramble_hier_strong(children, flip_prob)

        # randomly flip direction with flip_prob probability
        new_direction = (
            ("R" if direction == "L" else "L")
            if random.random() < flip_prob
            else direction
        )
        shuffled_tree[(word, new_direction)] = shuffled_children

    l_siblings = [(k, v) for k, v in shuffled_tree.items() if k[1] == "L"]
    r_siblings = [(k, v) for k, v in shuffled_tree.items() if k[1] == "R"]

    random.shuffle(l_siblings)
    random.shuffle(r_siblings)

    return dict(l_siblings + r_siblings)


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
