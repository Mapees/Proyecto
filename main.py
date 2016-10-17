#Librerias

import utils_corpus
import utils_recursos
import utils_train

# ------------------------------
#     LEER   TRAIN - TEST
# ------------------------------

# Data
data_train = utils_corpus.obtener_data('train.data');
data_test = utils_corpus.obtener_data('test.data');

# Tuis y labels
train, train_labels = utils_corpus.obtener_tuits_labels_train(data_train)
test, test_labels = utils_corpus.obtener_tuits_labels_test(data_test)

# ------------------------------
#            CORPUS
# ------------------------------

# Obtener corpus tokenizados y sin stopwords
corpus_train_tokenizado = utils_corpus.obtener_corpus_tokenizado(train)
corpus_test_tokenizado = utils_corpus.obtener_corpus_tokenizado(test)

# ------------------------------
#          RECURSOS
# ------------------------------

# LIWC
diccionario_liwc, vector_inicial_liwc, categorias_liwc = utils_recursos.obtener_diccionario_liwc()

# EMOLEX
diccionario_emolex, vector_inicial_emolex = utils_recursos.obtener_diccionario_emolex()


# ------------------------------
#       DATA - BAG OF WORDS
# ------------------------------

bag_train, bag_test = utils_recursos.obtener_vector_bag(corpus_train_tokenizado, corpus_test_tokenizado)

# ------------------------------
#         ENTRENAMIENTOS
# ------------------------------

print("BAG OG WORDS")
num_entrenamiento = 1
titulo = "BAG OF WORDS + Data tokenizado"
utils_train.entrenar_gaussiana2(num_entrenamiento, titulo, bag_train, train_labels, bag_test, test_labels)

# ------------------------------
#         CROSS VALIDATION
# ------------------------------
from sklearn.cross_validation import train_test_split
from sklearn import svm

X_train, X_test, y_train, y_test = train_test_split(corpus_train_tokenizado, train_labels, test_size = 0.1, random_state=0)
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print('Score: {}'.format(clf.score(X_test, y_test)))

# ------------------------------
#           TF-IDF
# ------------------------------

tfidf_train, tfidf_test = utils_recursos.obtener_vector_tfidf(corpus_train_tokenizado, corpus_test_tokenizado)
print("BAG OG WORDS")
num_entrenamiento = 2
titulo = "BAG OF WORDS + Data tokenizado"
utils_train.entrenar_gaussiana2(num_entrenamiento, titulo, tfidf_train, train_labels, tfidf_test, test_labels)

# ------------------------------
#  SARCASM, IRONY, NOT, OTHER
# ------------------------------
# Script.sh = Separa en los diferentes archivos.
# Tenemos separados los tweets en 4 archivos: irony.data, sarcasm, not y otros
# En los archivos irony_labels, sarcasm_labels,_ not_labels y otros_labels se tienen los ids de los tweets
