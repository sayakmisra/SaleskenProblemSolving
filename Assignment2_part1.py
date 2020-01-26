# Here we are using word-embeddings from pretrained word2vec models. Since the dataset is very small, other NLP encoding techniques
# like: bag of words, TF-IDF or creating a word2veec model from this dataset, will not yeild good results.

import numpy as np
import gensim
from scipy import spatial

file_sentences = []

with open ('list_of_sentences.txt') as fp:
    line = fp.readline()
    file_sentences.append(line)
    while fp.readline():
       line = fp.readline()
       file_sentences.append(line)

# Using the word embeddings of word2vec model, pre-trained om Google news dataset.
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
index2word_set = set(model.wv.index2word)

# Getting the sentence-embeddings, by simply doing the average of word-embeddings of each word.
def average_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec


simmilar_sentences = []
for sentence1 in file_sentences:
    sent1_embedding = average_feature_vector(sentence1, model=model, num_features=300, index2word_set=index2word_set)
    max_simmilarity = 0
    simmilar_sentence = []
    for sentence2 in file_sentences:
        if sentence1 != sentence2:
            sent2_embedding = average_feature_vector(sentence2, model=model, num_features=300, index2word_set=index2word_set)
            # Calculating the cosine differnce of two sntence embeddings.
            simmilarity = 1 - spatial.distance.cosine(sent1_embedding, sent2_embedding)
            if simmilarity > max_simmilarity:
                max_simmilarity = simmilarity
                simmilar_sentence = [sentence1,sentence2]
                
    simmilar_sentences.append(simmilar_sentence)            
    print("simmilar sentence:",simmilar_sentence)


print("The most simmilar sentences:",simmilar_sentences)