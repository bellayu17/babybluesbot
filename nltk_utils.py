import nltk
import json
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
# 1. First step is to tokenize the sentence, so that key words would be left
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
# 2. Second step is to find the stem words of each key words, so that the model can find patterns easier
def stem(word):
    return stemmer.stem(word.lower())
# 3. Third step is transform words to 0 and 1
def bag_of_words(tokenized_sentence, words):
    tokenized_sentence = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w, in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1
    return bag



