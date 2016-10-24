# Librerias

import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

##### LEER DATA #####
def obtener_data(fichero):
    with open (fichero, 'r') as f:
        data = f.readlines()
        return data

def obtener_train():
    data = obtener_data('train.data')
    tweets = []
    labels = []
    for line in data:
        sentence = re.split(r'\t+',  line)
        if ( len(sentence) > 2):
            tweets.append(sentence[2])
            labels.append(float( sentence[-1]))
    return tweets, labels

def obtener_test():
    data = obtener_data('test.data')    
    tweets = []
    labels = []
    for line in data:
        sentence = re.split(r'\t+',  line)
        if ( len(sentence) > 2):		# para evitar lineas vacias
            tweets.append(sentence[2])
            labels.append(float(sentence[1]))
    return tweets, labels

##### TOKENIZAR ##### 
emoticons_str = r"""
    (?:
        [:=;] 
        [oO\-]?
        [D\)\]\(\]/\\OpP]
        [:D]
        [:-O]
    )"""
    
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

## Si no pertenece a un emoticono se pasa a minusculas
def tokenize(data):
    corpus = []
    for line in data:
        tokens = TweetTokenizer().tokenize(line)
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
        sentence = " ".join(tokens)
        corpus.append(sentence)
    return corpus
    
# Quitar las stopwords de un corpus
def quitar_stopwords(data):
    corpus =  []
    for line in data:
        token = [word for word in line.split() if word not in stopwords.words('english')]
        sentence = " ".join(token)
        corpus.append(sentence)
    return corpus
