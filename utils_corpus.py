# Librerias

import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

tok = TweetTokenizer()

# Leer datos de train.data
def obtener_data(fichero):
    with open (fichero, 'r') as f:
        data = f.readlines()
        return data

# Obtener tuits y labels
def obtener_tuits_labels_train(data):
    tuits = []
    labels = []
    for line in data:
        sentence = re.split(r'\t+',  line)
        if ( len(sentence) > 2):		# para evitar lineas vacías
            tuits.append(sentence[2])
            label = sentence[-1]
            label = label[:-1]
            labels.append(float(label))
    return tuits, labels
    
def obtener_tuits_labels_test(data):
    tuits = []
    labels = []
    for line in data:
        sentence = re.split(r'\t+',  line)
        if ( len(sentence) > 2):		# para evitar lineas vacías
            tuits.append(sentence[2])
            label = sentence[1]
            labels.append(float(label))
    return tuits, labels


# Obtener labels
def obtener_labels_data(data):
    data_labels = []
    for line in data:
        sentence = re.split(r'\t+',  line)
        if ( len(sentence) > 2):
            label = sentence[-1]
            label = label[:-1]
            data_labels.append(float(label))
    return data_labels
    
    
# Tokenizar el corpus    
def obtener_corpus_tokenizado(data):
    corpus = []
    for line in data:
        token = tok.tokenize(line)
        sentence = " ".join(token)
        corpus.append(sentence)
    return corpus
    
# Quitar las stopwords de un corpus
def obtener_corpus_stopwords(data):
    corpus =  []
    for line in data:
        token = [word for word in line.split() if word not in stopwords.words('english')]
        sentence = " ".join(token)
        corpus.append(sentence)
    return corpus
    
