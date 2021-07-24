from typing import List, Dict, Union

import numpy as np

import nltk
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('tagsets')


def get_named_entities(text: str) -> List[str]:

    continuous_chunk = []
    current_chunk = []

    for i in ne_chunk(pos_tag(word_tokenize(text))):
        if isinstance(i, Tree):
            current_chunk.append(" ".join(token for token, pos in i.leaves()))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            continuous_chunk.append(named_entity)
            current_chunk = []

    named_entity = " ".join(current_chunk)
    continuous_chunk.append(named_entity)

    return continuous_chunk


_general_tags = frozenset(
    {'gVB', 'gNN', 'gPR', 'gWP', 'gRB', 'gJJ'}
)

_raw_tags = frozenset(
    {
        'LS', 'TO', 'VBN', "''",
        'WP', 'UH', 'VBG', 'JJ',
        'VBZ', '--', 'VBP', 'NN',
        'DT', 'PRP', ':', 'WP$',
        'NNPS', 'PRP$', 'WDT',
        '(', ')', '.', ',', '``',
        '$', 'RB', 'RBR', 'RBS',
        'VBD', 'IN', 'FW', 'RP',
        'JJR', 'JJS', 'PDT', 'MD',
        'VB', 'WRB', 'NNP', 'EX',
        'NNS', 'SYM', 'CC', 'CD', 'POS'
    }
)

_tagset = (
    *_raw_tags,
    *_general_tags
)


def count_part_of_speechs(text: str) -> Dict[str, Union[int, float]]:

    total_count  = dict.fromkeys(_tagset, 0)
    max_in_sent  = dict.fromkeys(_tagset, 0)
    min_in_sent  = dict.fromkeys(_tagset, 0)
    mean_in_sent = dict.fromkeys(_tagset, 0)

    for word, pos in nltk.pos_tag(nltk.word_tokenize(text)):
        total_count[pos] += 1
        general_tag = f'g{pos[:2]}'
        if general_tag in _general_tags:
            total_count[general_tag] += 1

    sentences = nltk.sent_tokenize(text)
    num_sentences = len(sentences)
    num_words = []

    for sentence in map(nltk.word_tokenize, sentences):
        cur_sentence_stat = dict.fromkeys(_tagset, 0)
        num_words.append(len(sentence))
        for word, pos in nltk.pos_tag(sentence):
            cur_sentence_stat[pos] += 1
            general_tag = f'g{pos[:2]}'
            if general_tag in _general_tags:
                cur_sentence_stat[general_tag] += 1
        for tag in _tagset:
            max_in_sent[tag]   = max(max_in_sent[tag], cur_sentence_stat[tag])
            min_in_sent[tag]   = min(min_in_sent[tag], cur_sentence_stat[tag])
            mean_in_sent[tag] += cur_sentence_stat[tag] / num_sentences

    res = {}
    for k, v in total_count.items():
        res[f'TOTAL_{k}'] = v
    for k, v in max_in_sent.items():
        res[f'MAX_{k}'] = v
    for k, v in min_in_sent.items():
        res[f'MIN_{k}'] = v
    for k, v in mean_in_sent.items():
        res[f'MEAN_{k}'] = v

    num_words = np.array(num_words)
    res['NUM_SENTENCES'] = num_sentences
    res['MEAN_NUM_WORDS'] = num_words.mean()
    res['STD_NUM_WORDS'] = (((num_words - num_words.mean()) ** 2).sum() / len(num_words)) ** 0.5 
    return res
