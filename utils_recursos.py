# Librerias
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

# ------------------------------
#     UTILIDADES RECURSOS
# ------------------------------

# ------------------------------
#            LIWC
# ------------------------------

# Leer categorias, palabras entre % y %

def obtener_diccionario_liwc():
    dic_liwc = {}
    vector_liwc = {}
    categorias_liwc = {}
    with open ('LIWC2007_English.dic', 'r') as f:
        cont = False
        for line in f:
            if line.startswith('%'):
                if cont == True:
                    cont = False
                else:
                    cont = True
            elif cont == True:
                l = line.split()
                categorias_liwc[l[0]] = l[1]
                vector_liwc[l[1]] = 0
            else:
                l = line.split()
                for i in l[1:-1]:
                    dic_aux = {}
                    categoria = categorias_liwc[i]
                    if categoria in dic_liwc:
                        dic_aux = dic_liwc[categoria]
                    dic_aux[l[0]] = 1
                    dic_liwc[categoria] = dic_aux
        return dic_liwc, vector_liwc, categorias_liwc
                                                        
def obtener_vector_pn_liwc(data, diccionario): 
    vector_resultante = []
    vector_resultante_normalizado = []
    posemo = diccionario['posemo']
    negemo = diccionario['negemo']
    for line in data:
        sentence = line.split()
        positivo = 0
        negativo = 0
        for i in sentence:
            palabra = i.lower()
            if (palabra in posemo):
                positivo = positivo + posemo[palabra]
            elif (palabra in negemo):
                negativo = negativo + negemo[palabra]
        vector_tmp = []
        vector_norm = []
        vector_tmp.extend([positivo, negativo])
        vector_norm.extend([positivo / len(sentence), negativo / len(sentence)])
        vector_resultante.append(vector_tmp)
        vector_resultante_normalizado.append(vector_norm)
    return np.array(vector_resultante), np.array(vector_resultante_normalizado)

# ------------------------------
#           EMOLEX
# ------------------------------
def obtener_diccionario_emolex():
    dic_emolex = {}
    vector_emolex = {}
    with open ('emolex.txt', 'r') as f:
        for line in f:
            dic_aux = {}
            l = line.split()
            palabra = l[0]
            categoria = l[1]
            if l[2] == "1":
                if categoria in dic_emolex:
                    dic_aux = dic_emolex[categoria]
                dic_aux[palabra] = 1
                dic_emolex[categoria] = dic_aux
                if categoria not in vector_emolex:
                    vector_emolex[categoria] = 0
        return dic_emolex, vector_emolex

def obtener_vector_pn_emolex(data, diccionario): 
    vector_resultante = []
    vector_resultante_normalizado = []
    posemo = diccionario['positive']
    negemo = diccionario['negative']
    for line in data:
        positivo = 0
        negativo = 0
        sentence = line.split()
        for i in sentence:
            palabra = i.lower()
            if (palabra in posemo):
                positivo = positivo + posemo[palabra]
            elif (palabra in negemo):
                negativo = negativo + negemo[palabra]
        vector_tmp = []
        vector_norm = []
        vector_tmp.extend([positivo, negativo])
        vector_norm.extend([positivo / len(sentence), negativo / len(sentence)])
        vector_resultante.append(vector_tmp)
        vector_resultante_normalizado.append(vector_norm)
    return np.array(vector_resultante), np.array(vector_resultante_normalizado)
 


# ------------------------------
#       OTRAS UTILIDADES
# ------------------------------
def obtener_vector(data, diccionario, vector_inicial):
    vector_resultante = []
    vector_resultante_normalizado = []
    for line in data:
        count_tmp = vector_inicial.copy()
        sentence = line.split()
        for i in sentence:
            palabra = i.lower()
            for categoria in diccionario:
                dic_aux = diccionario[categoria]        
                if (palabra in dic_aux):
                    valor = count_tmp[categoria]
                    valor = valor + dic_aux[palabra]
                    count_tmp[categoria] = valor             
            
        vector=[]
        vector_temp_norm = []
        for key in count_tmp:
            vector.append(count_tmp[key])
            vector_temp_norm.append(count_tmp[key]/ len(sentence))
        vector_resultante.append(vector)
        vector_resultante_normalizado.append(vector_temp_norm)
    return np.array(vector_resultante), np.array(vector_resultante_normalizado)
    
    
def obtener_categorias_diccionario(diccionario):
    categorias = []    
    for key in diccionario.keys():
        categorias.append(key)
    return categorias
    
def crear_vector_por_categoria(diccionario):
    count_diccionario = {}
    for key in diccionario.keys():
        count_diccionario[key] = 0
    return count_diccionario
    
def duplicar_vectores(a, b):
    vector_resultado = []
    tam = len(a)
    for i in range(tam):        
        v1 = a[i]
        v2 = b[i]
        v = []
        for x in v1:
            v.append(x)
        for y in v2:
            v.append(y)
        vector_resultado.append(v)
    return vector_resultado

# ------------------------------
#       BAG of WORDS
# ------------------------------
def obtener_vector_bag(train, test):
    vectorizer = CountVectorizer()
    train_bag = vectorizer.fit_transform(train)
    test_bag = vectorizer.transform(test)
   # print ("En los {} tweets de entrenamiento habian {} palabras distintas ".format(train.shape[0], train.shape[1]))
    return train_bag, test_bag

def obtener_vector_tfidf(train, test):
    transformer = TfidfTransformer()
    train_tf =  transformer.fit_transform(train)
    test_tf = transformer.transform(test)
    return train_tf, test_tf
   
