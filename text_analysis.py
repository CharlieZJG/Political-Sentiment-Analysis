import nltk
from nltk.corpus import wordnet
from textblob import TextBlob
from profanity_check import predict_prob
import jieba.posseg as psg
from profanityfilter import ProfanityFilter
"""
Extract names from text
"""
def get_human_names(text):
    person_list = []
    person_names = person_list
    tokens = nltk.tokenize.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    sentt = nltk.ne_chunk(pos, binary = False)

    person = []
    name = ""
    for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
        for leaf in subtree.leaves():
            person.append(leaf[0])
        if len(person) > 1: #avoid grabbing lone surnames
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
        person = []

    for person in person_list:
        person_split = person.split(" ")
        for name in person_split:
            if wordnet.synsets(name):
                if(name in person):
                    person_names.remove(person)
                    break
    return person_names

"""
Extract noun phrases
"""
def extract_nouns(text):

    blob = TextBlob(text)
    return blob.sentences


"""
check profanity
"""
def check_profanity(text):
    return predict_prob(text)


"""
get human names in Chinese
"""

def cn_get_human_names(text):
    list = []
    for word in text:
        res = psg.cut(word)
        for item in res:
            if item.flag == 'nr':
                list.append(item.word)
    return list

"""
filter profanty
"""

def filter_profanity(text):
    pf = ProfanityFilter()
    tx = pf.censor(text)
    return tx
"""
Count Chinese characters
"""
def hans_count(str):
    hans_total = 0
    for s in str:
        if '\u4e00' <= s <= '\u9fef':
            hans_total += 1
    return hans_total