import utils_corpus
import utils_recursos
import utils_train
import utils_emoticonos
import numpy as np
import scipy.sparse as sp
import re

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#from sklearn.model_selection import cross_val_predict, cross_val_score

# Obtener datos
print('Obteniendo datos...')
train, train_labels = utils_corpus.obtener_train();
test, test_labels = utils_corpus.obtener_test();

# Tokenizar
print('Tokenizando el corpus...')
# tokenize_tweet
# tokenize_word_token
train_token = utils_corpus.tokenize_tweet(train);
test_token = utils_corpus.tokenize_tweet(test);

# Obtener corpus tokenizados y sin stopwords
#print('Quitando stop words...')
#train_token = utils_corpus.quitar_stopwords(train_token)
#test_token = utils_corpus.quitar_stopwords(test_token)

# Recursos - LIWC y EMOLEX
print('Obteniendo diccionarios LIWC y Emolex...')
dic_liwc, vector_ini_liwc, categorias_liwc = utils_recursos.obtener_diccionario_liwc()
dic_emolex, vector_ini_emolex = utils_recursos.obtener_diccionario_emolex()

################################
## Obtener datos entrenamientos
# BoW
print('Obteniendo BoW...')
bag_train, bag_test = utils_recursos.obtener_vector_bag(train_token, test_token);
print('Obteniendo TFIDF...')
train_tf, test_tf = utils_recursos.obtener_vector_tfidf(train_token, test_token);


######     VECTORES     ######
print('Obteniendo vectores LIWC...')
# Todas las categorias
vect_train_liwc, vect_train_liwc_norm = utils_recursos.obtener_vector(train_token, dic_liwc, vector_ini_liwc)
vect_test_liwc, vect_test_liwc_norm = utils_recursos.obtener_vector(test_token, dic_liwc, vector_ini_liwc)
# Posemo y negemo
vect_train_liwc_pn, vect_train_liwc_pn_norm = utils_recursos.obtener_vector_pn_liwc(train_token, dic_liwc)
vect_test_liwc_pn, vect_test_liwc_pn_norm = utils_recursos.obtener_vector_pn_liwc(test_token, dic_liwc)

print('Obteniendo vectores Emolex...')
# Todas las categorias
vect_train_emolex, vect_train_emolex_norm = utils_recursos.obtener_vector(train_token, dic_emolex, vector_ini_emolex)
vect_test_emolex, vect_test_emolex_norm = utils_recursos.obtener_vector(test_token, dic_emolex, vector_ini_emolex)
# Positivo y negativo
vect_train_emolex_pn, vect_train_emolex_pn_norm = utils_recursos.obtener_vector_pn_emolex(train_token, dic_emolex)
vect_test_emolex_pn, vect_test_emolex_pn_norm = utils_recursos.obtener_vector_pn_emolex(test_token, dic_emolex)

print('Obteniendo vectores de Emoticonos...')
# Train
vector_inicial_train_emoticonos = utils_emoticonos.obtener_diccionario_emoticonos(train_token)
vector_emoticonos_train, vector_emoticonos_train_norm = utils_emoticonos.obtener_vector(train_token, vector_inicial_train_emoticonos)

# Test
vector_inicial_test_emoticonos = utils_emoticonos.obtener_diccionario_emoticonos(test_token)
vector_emoticonos_test, vector_emoticonos_test_norm = utils_emoticonos.obtener_vector(test_token, vector_inicial_train_emoticonos)
print('Transformando vectores de entrenamiento...')
# Vectores Emolex: vect_train_emolex - vect_train_emolex_pn
# LIWC + EMOLEX
vect_train_liwc_emo = np.hstack((vect_train_liwc, vect_train_emolex))
vect_train_liwc_emo_norm =  np.hstack((vect_train_liwc_norm, vect_train_emolex_norm))

vect_test_liwc_emo = np.hstack((vect_test_liwc, vect_test_emolex))
vect_test_liwc_emo_norm = np.hstack((vect_test_liwc_norm, vect_test_emolex_norm))

# Vectores LIWC: vect_train_liwc - vect_train_liwc_pn
# LIWC + EMOLEX  - P/N
vect_train_liwc_emo_pn = np.hstack((vect_train_liwc_pn, vect_train_emolex_pn))
vect_train_liwc_emo_norm_pn = np.hstack((vect_train_liwc_pn_norm, vect_train_emolex_pn_norm))

vect_test_liwc_emo_pn = np.hstack((vect_test_liwc_pn, vect_test_emolex_pn))
vect_test_liwc_emo_norm_pn = np.hstack((vect_test_liwc_pn_norm, vect_test_emolex_pn_norm))


print('Comenzando a entrenar...')

def experimentosPositivoNegativo():

    print("ENTRENAMIENTO: 1 - LIWC (posemo y negemo)")
    print("Normal:")
    utils_train.clas_svr(vect_train_liwc_pn, train_labels, vect_test_liwc_pn, test_labels)
    print("Normalizado:")
    utils_train.clas_svr(vect_train_liwc_norm, train_labels, vect_test_liwc_norm, test_labels)

    print("ENTRENAMIENTO: 2 - EMOLEX (positive y negative)")
    print("Normal:")    
    utils_train.clas_svr(vect_train_emolex_pn, train_labels, vect_test_emolex_pn, test_labels)  
    print("Normalizado:")    
    utils_train.clas_svr(vect_train_emolex_pn_norm, train_labels, vect_test_emolex_pn_norm, test_labels)   

    print("ENTRENAMIENTO: 3 - EMOLEX  + LIWC (positive y negative)")
    print("Normal:")        
    utils_train.clas_svr(vect_train_liwc_emo_pn, train_labels, vect_test_liwc_emo_pn, test_labels)  
    print("Normalizado:")    
    utils_train.clas_svr(vect_train_liwc_emo_norm_pn, train_labels, vect_test_liwc_emo_norm_pn, test_labels) 

def experimentosTodasCategorias():

    print("ENTRENAMIENTO: 4 - LIWC (completo)")
    print("Normal:")       
    utils_train.clas_svr(vect_train_liwc, train_labels, vect_test_liwc, test_labels)
    print("Normalizado:")
    utils_train.clas_svr(vect_train_liwc_norm, train_labels, vect_test_liwc_norm, test_labels)
    
    print("ENTRENAMIENTO: 5 - EMOLEX (completo)")
    print("Normal:")       
    utils_train.clas_svr(vect_train_emolex, train_labels, vect_test_emolex, test_labels)   
    print("Normalizado:")
    utils_train.clas_svr(vect_train_emolex_norm, train_labels, vect_test_emolex_norm, test_labels)   
    
    print("ENTRENAMIENTO: 6 - EMOLEX + LIWC (completo)")
    print("Normal:")       
    utils_train.clas_svr(vect_train_liwc_emo, train_labels, vect_test_liwc_emo, test_labels)  
    print("Normalizado:")
    utils_train.clas_svr(vect_train_liwc_emo_norm, train_labels, vect_test_liwc_emo_norm, test_labels)   

def experimentosBagOfWords():

    print("ENTRENAMIENTO: 14 - BAG OF WORDS")
    utils_train.clas_svr(bag_train, train_labels, bag_test, test_labels)
    
    bow_train_liwc = sp.hstack((bag_train, vect_train_liwc))
    bow_test_liwc = sp.hstack((bag_test, vect_test_liwc))
    print("ENTRENAMIENTO: 15 - BAG OF WORDS + LIWC")
    utils_train.clas_svr(bow_train_liwc, train_labels, bow_test_liwc, test_labels)
    
    bow_train_liwc_pn = sp.hstack((bag_train, vect_train_liwc_pn))
    bow_test_liwc_pn = sp.hstack((bag_test, vect_test_liwc_pn))
    print("ENTRENAMIENTO: 16 - BAG OF WORDS + LIWC P/N")
    utils_train.clas_svr(bow_train_liwc_pn, train_labels, bow_test_liwc_pn, test_labels)
    
    bow_train_emo = sp.hstack((bag_train, vect_train_emolex))
    bow_test_emo = sp.hstack((bag_test, vect_test_emolex))
    print("ENTRENAMIENTO: 17 - BAG OF WORDS + Emolex")
    utils_train.clas_svr(bow_train_emo, train_labels, bow_test_emo, test_labels)
    
    bow_train_emo_pn = sp.hstack((bag_train, vect_train_emolex_pn))
    bow_test_emo_pn = sp.hstack((bag_test, vect_test_emolex_pn))
    print("ENTRENAMIENTO: 18 - BAG OF WORDS + Emolex P/N")
    utils_train.clas_svr(bow_train_emo_pn, train_labels, bow_test_emo_pn, test_labels)
    
    bow_train_liwc_emo = sp.hstack((bag_train, vect_train_liwc_emo))
    bow_test_liwc_emo = sp.hstack((bag_test, vect_test_liwc_emo))
    print("ENTRENAMIENTO: 19 - BAG OF WORDS + LIWC + Emolex")
    utils_train.clas_svr(bow_train_liwc_emo, train_labels, bow_test_liwc_emo, test_labels)
    
    bow_train_liwc_emo_pn = sp.hstack((bag_train, vect_train_liwc_emo_pn))
    bow_test_liwc_emo_pn = sp.hstack((bag_test, vect_test_liwc_emo_pn))
    print("ENTRENAMIENTO: 20 - BAG OF WORDS + LIWC + Emolex + P/N")
    utils_train.clas_svr(bow_train_liwc_emo_pn, train_labels, bow_test_liwc_emo_pn, test_labels)

# Split corpus categorical
tweets_cat = {'irony':[], 'sarcasm': [], 'not': [],'other': []}
test_cat = {'irony':[], 'sarcasm': [], 'not': [],'other': []}
test_labels_split = {'irony':[], 'sarcasm': [], 'not': [],'other': []}
# Emolex
vector_test_split_emo_pn = {'irony':[], 'sarcasm': [], 'not': [],'other': []}
vector_test_split_emo = {'irony':[], 'sarcasm': [], 'not': [],'other': []}
# Liwc
vector_test_split_liwc_pn = {'irony':[], 'sarcasm': [], 'not': [],'other': []}
vector_test_split_liwc = {'irony':[], 'sarcasm': [], 'not': [],'other': []}
# Emoticonos
vector_test_split_emot = {'irony':[], 'sarcasm': [], 'not': [],'other': []}
def dividir_tweets_hashtag():

    print("Separando el test por hashtag...")
    
    for post, tweet in enumerate(test_token):
        pattern_irony= re.findall(r'#iro\w+', tweet)
        pattern_sarcasm = re.findall(r'#sarcas\w+', tweet)
        pattern_not = re.findall(r'#not? | #n?', tweet)
    
        if pattern_irony:
            tweets_cat['irony'].append(tweet)
            test_cat['irony'].append(post)
            vector_test_split_emo_pn['irony'].append(vect_test_emolex_pn[post])
            vector_test_split_emo['irony'].append(vect_test_emolex[post])
            vector_test_split_liwc_pn['irony'].append(vect_test_liwc_pn[post])
            vector_test_split_liwc['irony'].append(vect_test_liwc[post])
            test_labels_split['irony'].append(test_labels[post])
            vector_test_split_emot['irony'].append(vector_emoticonos_train[post])
        elif pattern_sarcasm:
            tweets_cat['sarcasm'].append(tweet)
            test_cat['sarcasm'].append(post)
            vector_test_split_emo_pn['sarcasm'].append(vect_test_emolex_pn[post])
            vector_test_split_emo['sarcasm'].append(vect_test_emolex[post])
            vector_test_split_liwc_pn['sarcasm'].append(vect_test_liwc_pn[post])
            vector_test_split_liwc['sarcasm'].append(vect_test_liwc[post])
            test_labels_split['sarcasm'].append(test_labels[post])
            vector_test_split_emot['sarcasm'].append(vector_emoticonos_train[post])
        elif pattern_not:
            tweets_cat['not'].append(tweet)
            test_cat['not'].append(post)
            vector_test_split_emo_pn['not'].append(vect_test_emolex_pn[post])
            vector_test_split_emo['not'].append(vect_test_emolex[post])
            vector_test_split_liwc_pn['not'].append(vect_test_liwc_pn[post])
            vector_test_split_liwc['not'].append(vect_test_liwc[post])
            test_labels_split['not'].append(test_labels[post])
            vector_test_split_emot['not'].append(vector_emoticonos_train[post])
        else:
            tweets_cat['other'].append(tweet)
            test_cat['other'].append(post)
            vector_test_split_emo_pn['other'].append(vect_test_emolex_pn[post])
            vector_test_split_emo['other'].append(vect_test_emolex[post])
            vector_test_split_liwc_pn['other'].append(vect_test_liwc_pn[post])
            vector_test_split_liwc['other'].append(vect_test_liwc[post])
            test_labels_split['other'].append(test_labels[post])
            vector_test_split_emot['other'].append(vector_emoticonos_train[post])
    
def experimentosEmoticonos():
    
    
    print("ENTRENAMIENTO: 7 - Emoticonos + BoW ")       
    bow_train_e = sp.hstack((train_tf, vector_emoticonos_train))
    bow_test_e = sp.hstack((test_tf, vector_emoticonos_test))
    utils_train.clas_svr(bow_train_e, train_labels, bow_test_e, test_labels)
    
    print("ENTRENAMIENTO: 8 - Emoticonos + IRONY ")    
    bow_train_emo_i, bow_test_emo_i = utils_recursos.obtener_vector_tfidf(train_token, tweets_cat['irony'])
    bow_train_emo = sp.hstack((bow_train_emo_i, vector_emoticonos_train))
    bow_test_categ_emo = sp.hstack((bow_test_emo_i, vector_test_split_emot['irony'])) 
    utils_train.clas_svr(bow_train_emo, train_labels, bow_test_categ_emo, test_labels_split['irony'])
        
    print("ENTRENAMIENTO: 9 - Emoticonos + SARCASM")    
    bow_train_emo_i, bow_test_emo_i = utils_recursos.obtener_vector_tfidf(train_token, tweets_cat['sarcasm'])
    bow_train_emo = sp.hstack((bow_train_emo_i, vector_emoticonos_train))
    bow_test_categ_emo = sp.hstack((bow_test_emo_i, vector_test_split_emot['sarcasm'])) 
    utils_train.clas_svr(bow_train_emo, train_labels, bow_test_categ_emo, test_labels_split['sarcasm'])
    
    print("ENTRENAMIENTO: 10 - Emoticonos + NOT")    
    bow_train_emo_i, bow_test_emo_i = utils_recursos.obtener_vector_tfidf(train_token, tweets_cat['not'])
    bow_train_emo = sp.hstack((bow_train_emo_i, vector_emoticonos_train))
    bow_test_categ_emo = sp.hstack((bow_test_emo_i, vector_test_split_emot['not'])) 
    utils_train.clas_svr(bow_train_emo, train_labels, bow_test_categ_emo, test_labels_split['not'])
    
    print("ENTRENAMIENTO: 11 - Emoticonos + OTHER")
    bow_train_emo_i, bow_test_emo_i = utils_recursos.obtener_vector_tfidf(train_token, tweets_cat['other'])
    bow_train_emo = sp.hstack((bow_train_emo_i, vector_emoticonos_train))
    bow_test_categ_emo = sp.hstack((bow_test_emo_i, vector_test_split_emot['other'])) 
    utils_train.clas_svr(bow_train_emo, train_labels, bow_test_categ_emo, test_labels_split['other'])
        
def experimentosSplitIrony(tipo):
    
    if(tipo == "bow") :
        train, bow_test_categ = utils_recursos.obtener_vector_bag(train_token, tweets_cat['irony'])
    else:
        train, bow_test_categ = utils_recursos.obtener_vector_tfidf(train_token, tweets_cat['irony'])
        
    print("Tipo: ".format(tipo))
    bow_train_emo = sp.hstack((train, vect_train_emolex))
    bow_test_categ_emo = sp.hstack((bow_test_categ, vector_test_split_emo['irony']))   
    print("ENTRENAMIENTO: 21 - Split IRONY BoW + Emolex")
    utils_train.clas_svr(bow_train_emo, train_labels, bow_test_categ_emo, test_labels_split['irony'])
              
    bow_train_emo_pn = sp.hstack((train, vect_train_emolex_pn))
    bow_test_categ_emo_pn = sp.hstack((bow_test_categ, vector_test_split_emo_pn['irony']))
    print("ENTRENAMIENTO: 22 - Split IRONY BoW + Emolex   p/n")
    utils_train.clas_svr(bow_train_emo_pn, train_labels, bow_test_categ_emo_pn, test_labels_split['irony'])
    
    bow_train_liwc = sp.hstack((train, vect_train_liwc))
    bow_test_categ_liwc = sp.hstack((bow_test_categ, vector_test_split_liwc['irony']))   
    print("ENTRENAMIENTO: 23 - Split IRONY BoW + LIWC")
    utils_train.clas_svr(bow_train_liwc, train_labels, bow_test_categ_liwc, test_labels_split['irony'])
              
    bow_train_liwc_pn = sp.hstack((train, vect_train_liwc_pn))
    bow_test_categ_liwc_pn = sp.hstack((bow_test_categ, vector_test_split_liwc_pn['irony']))
    print("ENTRENAMIENTO: 24 - Split IRONY BoW + LIWC   p/n")
    utils_train.clas_svr(bow_train_liwc_pn, train_labels, bow_test_categ_liwc_pn, test_labels_split['irony'])
    
    bow_train_liwc_emo = sp.hstack((train, vect_train_liwc_emo))
    aux = sp.hstack((bow_test_categ, vector_test_split_liwc['irony']))
    bow_test_categ_liwc_emo = sp.hstack((aux, vector_test_split_emo['irony']))
    print("ENTRENAMIENTO: 25 - Split IRONY BoW + LIWC + EMOLEX ")
    utils_train.clas_svr(bow_train_liwc_emo, train_labels, bow_test_categ_liwc_emo, test_labels_split['irony'])
    
    bow_train_liwc_emo_pn = sp.hstack((train, vect_train_liwc_emo_pn))
    aux_pn = sp.hstack((bow_test_categ, vector_test_split_liwc_pn['irony']))
    bow_test_categ_liwc_emo_pn = sp.hstack((aux_pn, vector_test_split_emo_pn['irony']))
    print("ENTRENAMIENTO: 26 - Split IRONY BoW + LIWC + EMOLEX p/n")
    utils_train.clas_svr(bow_train_liwc_emo_pn, train_labels, bow_test_categ_liwc_emo_pn, test_labels_split['irony'])
    
def experimentosSplitSarcasm(tipo):
    
    if(tipo == "bow") :
        train_b, bow_test_categ = utils_recursos.obtener_vector_bag(train_token, tweets_cat['sarcasm'])
    else:
        print("B")
        train_b, bow_test_categ = utils_recursos.obtener_vector_tfidf(train_token, tweets_cat['sarcasm'])



    bow_train_emo = sp.hstack((train_b, vect_train_emolex))
    bow_test_categ_emo = sp.hstack((bow_test_categ, vector_test_split_emo['sarcasm']))   
    print("ENTRENAMIENTO: 27 - Split SARCASM BoW + Emolex")
    utils_train.clas_svr(bow_train_emo, train_labels, bow_test_categ_emo, test_labels_split['sarcasm'])
              
    bow_train_emo_pn = sp.hstack((train_b, vect_train_emolex_pn))
    bow_test_categ_emo_pn = sp.hstack((bow_test_categ, vector_test_split_emo_pn['sarcasm']))
    print("ENTRENAMIENTO: 28 - Split SARCASM BoW + Emolex   p/n")
    utils_train.clas_svr(bow_train_emo_pn, train_labels, bow_test_categ_emo_pn, test_labels_split['sarcasm'])
    
    bow_train_liwc = sp.hstack((train_b, vect_train_liwc))
    bow_test_categ_liwc = sp.hstack((bow_test_categ, vector_test_split_liwc['sarcasm']))   
    print("ENTRENAMIENTO: 29 - Split SARCASM BoW + LIWC")
    utils_train.clas_svr(bow_train_liwc, train_labels, bow_test_categ_liwc, test_labels_split['sarcasm'])
              
    bow_train_liwc_pn = sp.hstack((train_b, vect_train_liwc_pn))
    bow_test_categ_liwc_pn = sp.hstack((bow_test_categ, vector_test_split_liwc_pn['sarcasm']))
    print("ENTRENAMIENTO: 30 - Split SARCASM BoW + LIWC   p/n")
    utils_train.clas_svr(bow_train_liwc_pn, train_labels, bow_test_categ_liwc_pn, test_labels_split['sarcasm'])
    
    bow_train_liwc_emo = sp.hstack((train_b, vect_train_liwc_emo))
    aux = sp.hstack((bow_test_categ, vector_test_split_liwc['sarcasm']))
    bow_test_categ_liwc_emo = sp.hstack((aux, vector_test_split_emo['sarcasm']))
    print("ENTRENAMIENTO: 31 - Split SARCASM BoW + LIWC + EMOLEX ")
    utils_train.clas_svr(bow_train_liwc_emo, train_labels, bow_test_categ_liwc_emo, test_labels_split['sarcasm'])
    
    bow_train_liwc_emo_pn = sp.hstack((train_b, vect_train_liwc_emo_pn))
    aux_pn = sp.hstack((bow_test_categ, vector_test_split_liwc_pn['sarcasm']))
    bow_test_categ_liwc_emo_pn = sp.hstack((aux_pn, vector_test_split_emo_pn['sarcasm']))
    print("ENTRENAMIENTO: 32 - Split SARCASM BoW + LIWC + EMOLEX p/n")
    utils_train.clas_svr(bow_train_liwc_emo_pn, train_labels, bow_test_categ_liwc_emo_pn, test_labels_split['sarcasm'])


def experimentosSplitNot(tipo):

    
    if(tipo == "bow") :
        train_b, bow_test_categ = utils_recursos.obtener_vector_bag(train_token, tweets_cat['not'])

    else:
        print("B")
        train_b, bow_test_categ = utils_recursos.obtener_vector_tfidf(train_token, tweets_cat['not'])
  
    bow_train_emo = sp.hstack((train_b, vect_train_emolex))
    bow_test_categ_emo = sp.hstack((bow_test_categ, vector_test_split_emo['not']))   
    print("ENTRENAMIENTO: 33 - Split NOT BoW + Emolex")
    utils_train.clas_svr(bow_train_emo, train_labels, bow_test_categ_emo, test_labels_split['not'])
              
    bow_train_emo_pn = sp.hstack((train_b, vect_train_emolex_pn))
    bow_test_categ_emo_pn = sp.hstack((bow_test_categ, vector_test_split_emo_pn['not']))
    print("ENTRENAMIENTO: 34 - Split NOT BoW + Emolex   p/n")
    utils_train.clas_svr(bow_train_emo_pn, train_labels, bow_test_categ_emo_pn, test_labels_split['not'])
    
    bow_train_liwc = sp.hstack((train_b, vect_train_liwc))
    bow_test_categ_liwc = sp.hstack((bow_test_categ, vector_test_split_liwc['not']))   
    print("ENTRENAMIENTO: 35 - Split NOT BoW + LIWC")
    utils_train.clas_svr(bow_train_liwc, train_labels, bow_test_categ_liwc, test_labels_split['not'])
              
    bow_train_liwc_pn = sp.hstack((train_b, vect_train_liwc_pn))
    bow_test_categ_liwc_pn = sp.hstack((bow_test_categ, vector_test_split_liwc_pn['not']))
    print("ENTRENAMIENTO: 36 - Split NOT BoW + LIWC   p/n")
    utils_train.clas_svr(bow_train_liwc_pn, train_labels, bow_test_categ_liwc_pn, test_labels_split['not'])
    
    bow_train_liwc_emo = sp.hstack((train_b, vect_train_liwc_emo))
    aux = sp.hstack((bow_test_categ, vector_test_split_liwc['not']))
    bow_test_categ_liwc_emo = sp.hstack((aux, vector_test_split_emo['not']))
    print("ENTRENAMIENTO: 37 - Split NOT BoW + LIWC + EMOLEX ")
    utils_train.clas_svr(bow_train_liwc_emo, train_labels, bow_test_categ_liwc_emo, test_labels_split['not'])
    
    bow_train_liwc_emo_pn = sp.hstack((train_b, vect_train_liwc_emo_pn))
    aux_pn = sp.hstack((bow_test_categ, vector_test_split_liwc_pn['not']))
    bow_test_categ_liwc_emo_pn = sp.hstack((aux_pn, vector_test_split_emo_pn['not']))
    print("ENTRENAMIENTO: 38 - Split NOT BoW + LIWC + EMOLEX p/n")
    utils_train.clas_svr(bow_train_liwc_emo_pn, train_labels, bow_test_categ_liwc_emo_pn, test_labels_split['not'])


def experimentosSplitOthers(tipo):

    
    
    if(tipo == "bow") :
        train_b, bow_test_categ = utils_recursos.obtener_vector_bag(train_token, tweets_cat['other'])

    else:
        print("B")
        train_b, bow_test_categ = utils_recursos.obtener_vector_tfidf(train_token, tweets_cat['other'])
  
    bow_train_emo = sp.hstack((train_b, vect_train_emolex))
    bow_test_categ_emo = sp.hstack((bow_test_categ, vector_test_split_emo['other']))   
    print("ENTRENAMIENTO: 39 - Split OTHERS BoW + Emolex")
    utils_train.clas_svr(bow_train_emo, train_labels, bow_test_categ_emo, test_labels_split['other'])
              
    bow_train_emo_pn = sp.hstack((train_b, vect_train_emolex_pn))
    bow_test_categ_emo_pn = sp.hstack((bow_test_categ, vector_test_split_emo_pn['other']))
    print("ENTRENAMIENTO: 40 - Split OTHERS BoW + Emolex   p/n")
    utils_train.clas_svr(bow_train_emo_pn, train_labels, bow_test_categ_emo_pn, test_labels_split['other'])
    
    bow_train_liwc = sp.hstack((train_b, vect_train_liwc))
    bow_test_categ_liwc = sp.hstack((bow_test_categ, vector_test_split_liwc['other']))   
    print("ENTRENAMIENTO: 41 - Split OTHERS BoW + LIWC")
    utils_train.clas_svr(bow_train_liwc, train_labels, bow_test_categ_liwc, test_labels_split['other'])
              
    bow_train_liwc_pn = sp.hstack((train_b, vect_train_liwc_pn))
    bow_test_categ_liwc_pn = sp.hstack((bow_test_categ, vector_test_split_liwc_pn['other']))
    print("ENTRENAMIENTO: 42 - Split OTHERS BoW + LIWC   p/n")
    utils_train.clas_svr(bow_train_liwc_pn, train_labels, bow_test_categ_liwc_pn, test_labels_split['other'])
    
    bow_train_liwc_emo = sp.hstack((train_b, vect_train_liwc_emo))
    aux = sp.hstack((bow_test_categ, vector_test_split_liwc['other']))
    bow_test_categ_liwc_emo = sp.hstack((aux, vector_test_split_emo['other']))
    print("ENTRENAMIENTO: 43 - Split OTHERS BoW + LIWC + EMOLEX ")
    utils_train.clas_svr(bow_train_liwc_emo, train_labels, bow_test_categ_liwc_emo, test_labels_split['other'])
    
    bow_train_liwc_emo_pn = sp.hstack((train_b, vect_train_liwc_emo_pn))
    aux_pn = sp.hstack((bow_test_categ, vector_test_split_liwc_pn['other']))
    bow_test_categ_liwc_emo_pn = sp.hstack((aux_pn, vector_test_split_emo_pn['other']))
    print("ENTRENAMIENTO: 44 - Split OTHERS BoW + LIWC + EMOLEX p/n")
    utils_train.clas_svr(bow_train_liwc_emo_pn, train_labels, bow_test_categ_liwc_emo_pn, test_labels_split['other'])
    


    
def experimentosTfIdf():



    print("ENTRENAMIENTO: 45 - TFIDF")
    utils_train.clas_svr(train_tf, train_labels, test_tf, test_labels)
    
    tfidf_train_liwc = sp.hstack((train_tf, vect_train_liwc))
    tfidf_test_liwc = sp.hstack((test_tf, vect_test_liwc))
    print("ENTRENAMIENTO: 46 - TFIDF + LIWC")
    utils_train.clas_svr(tfidf_train_liwc, train_labels, tfidf_test_liwc, test_labels)
    
    tfidf_train_liwc_pn = sp.hstack((train_tf, vect_train_liwc_pn))
    tfidf_test_liwc_pn = sp.hstack((test_tf, vect_test_liwc_pn))
    print("ENTRENAMIENTO: 47 - TFIDF + LIWC P/N")
    utils_train.clas_svr(tfidf_train_liwc_pn, train_labels, tfidf_test_liwc_pn, test_labels)
    
    tfidf_train_emo = sp.hstack((train_tf, vect_train_emolex))
    tfidf_test_emo = sp.hstack((test_tf, vect_test_emolex))
    print("ENTRENAMIENTO: 48 - TFIDF + Emolex")
    utils_train.clas_svr(tfidf_train_emo, train_labels, tfidf_test_emo, test_labels)
    
    tfidf_train_emo_pn = sp.hstack((train_tf, vect_train_emolex_pn))
    tfidf_test_emo_pn = sp.hstack((test_tf, vect_test_emolex_pn))
    print("ENTRENAMIENTO: 49 - TFIDF + Emolex P/N")
    utils_train.clas_svr(tfidf_train_emo_pn, train_labels, tfidf_test_emo_pn, test_labels)
    
    tfidf_train_liwc_emo = sp.hstack((train_tf, vect_train_liwc_emo))
    tfidf_test_liwc_emo = sp.hstack((test_tf, vect_test_liwc_emo))
    print("ENTRENAMIENTO: 50 - TFIDF + LIWC + Emolex")
    utils_train.clas_svr(tfidf_train_liwc_emo, train_labels, tfidf_test_liwc_emo, test_labels)
    
    tfidf_train_liwc_emo_pn = sp.hstack((train_tf, vect_train_liwc_emo_pn))
    tfidf_test_liwc_emo_pn = sp.hstack((test_tf, vect_test_liwc_emo_pn))
    print("ENTRENAMIENTO: 51 - TFIDF + LIWC + Emolex + P/N")
    utils_train.clas_svr(tfidf_train_liwc_emo_pn, train_labels, tfidf_test_liwc_emo_pn, test_labels)    






def experimentosBaseline():
       
    print("ENTRENAMIENTO: Tweet Tokenize - Sin quitar stop words")
    # SVM    
    # Result MSE: 4.606085530671697
    # Result Coseno: [[ 0.54898976]]
    # Naive
  
    #print("ENTRENAMIENTO: Word Tokenize - Sin quitar stop words")
    # SVM    
    # Result MSE: 4.606250091766936
    # Result Coseno: [[ 0.54898417]]
    
    #print("ENTRENAMIENTO: Tweet Tokenize - Quitando stop words")
    # SVM 
    # Result MSE: 4.615252360203416
    # Result Coseno: [[ 0.54828179]]
    # Naive

    
    #print("ENTRENAMIENTO: Word Tokenize - Quitando stop words")
    # SVM 
    # Result MSE: 4.636833596778402
    # Result Coseno: [[ 0.54776495]]
    
    #utils_train.clas_svr(bag_train, train_labels, bag_test, test_labels)
    # Error
    #raise TypeError('A sparse matrix was passed, but dense '
    #TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array.
    #utils_train.clas_naive(bag_train, train_labels, bag_test, test_labels)
    utils_train.clas_svc(bag_train, train_labels, bag_test, test_labels)
    
    
def gridSearch():
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],  'C': [1, 10, 100, 1000]},
                         {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
                         ]

    scores = ['precision']
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='%s_macro' % score)
        # Error
        #raise ValueError("Unknown label type: %r" % y_type)
        #ValueError: Unknown label type: 'continuous'
        clf.fit(bag_train, np.array(train_labels))
        
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = bag_test, clf.predict(test_labels)
        print(classification_report(y_true, y_pred))
        print()

# ARTICULO
#experimentosPositivoNegativo()
#experimentosTodasCategorias()


#experimentosBagOfWords()

#dividir_tweets_hashtag()
#experimentosEmoticonos()
#experimentosSplitIrony("bow")
#experimentosSplitSarcasm()
#experimentosSplitNot()
#experimentosSplitOthers()

#experimentosTfIdf();
#experimentosSplitIrony("otro")
#experimentosSplitSarcasm("otro")
#experimentosSplitNot("otro")
#experimentosSplitOthers("otro")


# PROYECTO
# Nos quedamos con tweet tokenizer y con stopwords

experimentosBaseline();

# Error grid search
#gridSearch();