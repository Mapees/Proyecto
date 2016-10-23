import utils_corpus
import utils_recursos
import utils_train
import utils_emoticonos
import time
import numpy as np
import re

from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_predict, cross_val_score

data_train = utils_corpus.obtener_data('train.data');
data_test = utils_corpus.obtener_data('test.data');

train, train_labels = utils_corpus.obtener_tuits_labels_train(data_train);
test, test_labels = utils_corpus.obtener_tuits_labels_test(data_test);

# Obtener corpus tokenizados y sin stopwords
corpus_train_tokenizado = utils_corpus.obtener_corpus_tokenizado(train)
corpus_train_token_stop = utils_corpus.obtener_corpus_stopwords(corpus_train_tokenizado)

corpus_test_tokenizado = utils_corpus.obtener_corpus_tokenizado(test)
corpus_test_token_stop = utils_corpus.obtener_corpus_stopwords(corpus_test_tokenizado)

bag_train, bag_test = utils_recursos.obtener_vector_bag(corpus_train_tokenizado, corpus_test_tokenizado);

# LIWC
diccionario_liwc, vector_inicial_liwc, categorias_liwc = utils_recursos.obtener_diccionario_liwc()

# EMOLEX
diccionario_emolex, vector_inicial_emolex = utils_recursos.obtener_diccionario_emolex()

# ------------------------------
#     VECTORES TOKENIZADOS
# ------------------------------
# Positivo - negativo
vector_train_pn_liwc, vector_train_pn_liwc_norm = utils_recursos.obtener_vector_pn_liwc(corpus_train_tokenizado, diccionario_liwc)
vector_test_pn_liwc, vector_test_pn_liwc_norm = utils_recursos.obtener_vector_pn_liwc(corpus_test_tokenizado, diccionario_liwc)

vector_train_pn_emolex, vector_train_pn_emolex_norm = utils_recursos.obtener_vector_pn_emolex(corpus_train_tokenizado, diccionario_emolex)
vector_test_pn_emolex, vector_test_pn_emolex_norm = utils_recursos.obtener_vector_pn_emolex(corpus_test_tokenizado, diccionario_emolex)

# Completos
vector_train_liwc, vector_train_liwc_norm = utils_recursos.obtener_vector(corpus_train_tokenizado, diccionario_liwc, vector_inicial_liwc)
vector_test_liwc, vector_test_liwc_norm = utils_recursos.obtener_vector(corpus_test_tokenizado, diccionario_liwc, vector_inicial_liwc)

vector_train_emolex, vector_train_emolex_norm = utils_recursos.obtener_vector(corpus_train_tokenizado, diccionario_emolex, vector_inicial_emolex)
vector_test_emolex, vector_test_emolex_norm = utils_recursos.obtener_vector(corpus_test_tokenizado, diccionario_emolex, vector_inicial_emolex)


# ------------------------------
#     VECTORES ENTRENAMIENTO
# ------------------------------
# Tokenizados
# LIWC + EMOLEX (positivo y negativo)
vector_train_pn_liwc_emo = utils_recursos.duplicar_vectores(vector_train_pn_liwc, vector_train_pn_emolex)
vector_test_pn_liwc_emo = utils_recursos.duplicar_vectores(vector_test_pn_liwc, vector_test_pn_emolex)

vector_train_pn_liwc_emo_norm = utils_recursos.duplicar_vectores(vector_train_pn_liwc_norm, vector_train_pn_emolex_norm)
vector_test_pn_liwc_emo_norm = utils_recursos.duplicar_vectores(vector_test_pn_liwc_norm, vector_test_pn_emolex_norm)

# LIWC + EMOLEX (completo)
vector_train_liwc_emo = utils_recursos.duplicar_vectores(vector_train_liwc, vector_train_emolex)
vector_train_liwc_emo_norm = utils_recursos.duplicar_vectores(vector_train_liwc_norm, vector_train_emolex_norm)

vector_test_liwc_emo = utils_recursos.duplicar_vectores(vector_test_liwc, vector_test_emolex)
vector_test_liwc_emo_norm = utils_recursos.duplicar_vectores(vector_test_liwc_norm, vector_test_emolex_norm)


# ------------------------------
#         EMOTICONOS
# ------------------------------
# Train
vector_inicial_train_emoticonos = utils_emoticonos.obtener_diccionario_emoticonos(corpus_train_tokenizado)
vector_emoticonos_train, vector_emoticonos_train_norm = utils_emoticonos.obtener_vector(corpus_train_tokenizado, vector_inicial_train_emoticonos)

vector_inicial_train_emoticonos_stop = utils_emoticonos.obtener_diccionario_emoticonos(corpus_train_token_stop)
vector_emoticonos_train_stop, vector_emoticonos_train_norm_stop = utils_emoticonos.obtener_vector(corpus_train_token_stop, vector_inicial_train_emoticonos_stop)

# Test
vector_inicial_test_emoticonos = utils_emoticonos.obtener_diccionario_emoticonos(corpus_test_tokenizado)
vector_emoticonos_test, vector_emoticonos_test_norm = utils_emoticonos.obtener_vector(corpus_test_tokenizado, vector_inicial_train_emoticonos)

vector_inicial_test_emoticonos_stop = utils_emoticonos.obtener_diccionario_emoticonos(corpus_test_token_stop)
vector_emoticonos_test_stop, vector_emoticonos_test_norm_stop = utils_emoticonos.obtener_vector(corpus_test_token_stop, vector_inicial_train_emoticonos_stop)

# VECTORES ENTRENAMIENTO
# Tokenizado
vector_emoticonos_completo_train_norm = utils_recursos.duplicar_vectores(vector_emoticonos_train_norm, vector_train_liwc_emo_norm)
vector_emoticonos_completo_test_norm = utils_recursos.duplicar_vectores(vector_emoticonos_test_norm, vector_test_liwc_emo_norm)


def experimentosPositivoNegativo():

    num_entrenamiento = 1
    titulo = "LIWC (posemo y negemo)"
    utils_train.entrenar_svm(num_entrenamiento, titulo, vector_train_pn_liwc_norm, train_labels, vector_test_pn_liwc_norm, test_labels)

    num_entrenamiento = 2
    titulo = "LIWC (completo)"
    utils_train.entrenar_svm(num_entrenamiento, titulo, vector_train_liwc_norm, train_labels, vector_test_liwc_norm, test_labels)

    num_entrenamiento = 3
    titulo = "EMOLEX (positive y negative)"
    utils_train.entrenar_svm(num_entrenamiento, titulo, vector_train_pn_emolex_norm, train_labels, vector_test_pn_emolex_norm, test_labels)    

    num_entrenamiento = 4
    titulo = "EMOTICONOS + Recursos tokenizados"
    utils_train.entrenar_svm(num_entrenamiento, titulo, vector_emoticonos_completo_train_norm, train_labels, vector_emoticonos_completo_test_norm, test_labels)
  
    

def experimentoBagOfWords():

    vectorizer = CountVectorizer()
    train_vect = vectorizer.fit_transform(train)
    test_vect = vectorizer.transform(test)

    clasifier = svm.SVR()
    clasifier.fit(train_vect, train_labels)
    prediction = clasifier.predict(test_vect)

    print("ENTRENAMIENTO: 5 - BAG OF WORDS + Data tokenizado")
    print('Result MSE: {}'.format(mean_squared_error(test_labels, prediction)))
    print('Result Coseno: {}'. format(cosine_similarity(prediction.reshape(1, -1), test_labels)))
    
    vectBowEmoPN = CountVectorizer()
    train_bowEmoPN = vectBowEmoPN.fit_transform(train)
    test_bowEmoPN = vectBowEmoPN.transform(test)

    train_bowEmoPN = np.hstack(bag_train, vector_train_pn_emolex_norm)
    test_bowEmoPN = np.hstack(bag_test, vector_test_pn_emolex_norm)
    
    clas_bowEmoPN = svm.SVR()
    clas_bowEmoPN.fit(train_bowEmoPN, train_labels)
    pre_bowEmoPN = clas_bowEmoPN.predict(test_bowEmoPN)
    
    print("ENTRENAMIENTO: 6 - BAG OF WORDS + EMOLEX p/n")
    print('Result MSE: {}'.format(mean_squared_error(test_labels, pre_bowEmoPN)))
    print('Result Coseno: {}'. format(cosine_similarity(pre_bowEmoPN.reshape(1, -1), test_labels)))
    
    vectBowEmo = CountVectorizer()
    train_bowEmo = vectBowEmo.fit_transform(train)
    test_bowEmo = vectBowEmo.transform(test)

    train_bowEmo = np.hstack(bag_train, vector_train_emolex_norm)
    test_bowEmo = np.hstack(bag_test, vector_test_emolex_norm)
    
    clas_bowEmo = svm.SVR()
    clas_bowEmo.fit(train_bowEmo, train_labels)
    pre_bowEmo = clas_bowEmo.predict(test_bowEmo)
    
    print("ENTRENAMIENTO: 7 - BAG OF WORDS + Todas las categorias")
    print('Result MSE: {}'.format(mean_squared_error(test_labels, pre_bowEmo)))
    print('Result Coseno: {}'. format(cosine_similarity(pre_bowEmo.reshape(1, -1), test_labels)))

def experimentosTodasCategorias():

    num_entrenamiento = 8
    titulo = "EMOLEX (completo)"
    utils_train.entrenar_svm(num_entrenamiento, titulo, vector_train_emolex_norm, train_labels, vector_test_emolex_norm, test_labels)
    
    num_entrenamiento = 9
    titulo = "LIWC (completo)"
    utils_train.entrenar_svm(num_entrenamiento, titulo, vector_train_liwc_norm, train_labels, vector_test_liwc_norm, test_labels)


# Split corpus categorical
def experimentoSplitCorpusEmolexPN():
    tweet_cat = {'irony':[], 'sarcasm': [], 'not': [],'other': []}
    test_cat = {'irony':[], 'sarcasm': [], 'not': [],'other': []}

    for post, tweet in enumerate(test):
        pattern_irony= re.findall(r'#[iI][rR][oO][nN][yY]? | #[iI][rR][oO]?', tweet)
        pattern_sarcasm = re.findall(r'#[sS][aA][rR][cC][aA][sS][mM]? | #[sS][aA][rR][cC]? | *#[sS][aA][rR][cC]*', tweet)
        pattern_not = re.findall(r'#[nN][oO][tT]? | *#[nN]?', tweet)
    
        if pattern_irony:
            tweet_cat['irony'].append(tweet)
            test_cat['irony'].append(tweet)
        elif pattern_sarcasm:
            tweet_cat['sarcasm'].append(tweet)
            test_cat['sarcasm'].append(tweet)
        elif pattern_not:
            tweet_cat['not'].append(tweet)
            test_cat['not'].append(tweet)
        else:
            tweet_cat['other'].append(tweet)
            test_cat['other'].append(tweet)


    vect_irony = CountVectorizer()
    train_irony = vect_irony.fit_transform(train)
    test_irony = vect_irony.transform(tweet_cat['irony'])

    class_irony = svm.SVR()
    class_irony.fit(train_irony, train_labels)
    predict_irony = class_irony.predict(bag_test)

    print("ENTRENAMIENTO: - Split IRONY BoW + Emolex  p/n")
    print('Result MSE: {}'.format(mean_squared_error(test_cat['irony'], predict_irony)))
    print('Result Coseno: {}'. format(cosine_similarity(predict_irony.reshape(1, -1), test_irony)))
    
    print("ENTRENAMIENTO: - Split IRONY BoW + Emolex  p/n")
    print("ENTRENAMIENTO: - Split SARCASM BoW + Emolex  p/n")
    print("ENTRENAMIENTO: - Split NOT BoW + Emolex  p/n")
    print("ENTRENAMIENTO: - Split OTHERS BoW + Emolex  p/n")
    


experimentosPositivoNegativo()
experimentoBagOfWords()
experimentosTodasCategorias()

experimentoSplitCorpusEmolexPN()