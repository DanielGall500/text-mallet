import spacy
from spacy.tokens import Doc
from tmallet.utils.spacy_registry import get_spacy_nlp

nlp = get_spacy_nlp(pipeline="ner", prefer_gpu=True)

def get_pos_tags(word_list):
    """
    Takes a list of words and returns a list of universal POS tags.
    """
    spaces = [True] * (len(word_list) - 1) + [False]
    doc = Doc(nlp.vocab, words=word_list, spaces=spaces)
    
    for name, proc in nlp.pipeline:
            doc = proc(doc)
            
    return [token.pos_ for token in doc]
