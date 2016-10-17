# Librerias

from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import time
from sklearn.naive_bayes import GaussianNB
import numpy

# ------------------------------
#   UTILIDADES ENTRENAMIENTOS
# ------------------------------

def entrenar_svm(num_entrenamiento, titulo, vector_train, labels_train, vector_test, labels_test):
    print("ENTRENAMIENTO: ", num_entrenamiento, " - ", titulo)
    clasifier = svm.SVR()
    t0 = time.time()
    clasifier.fit(vector_train, labels_train)
    t1 = time.time()
    prediction = clasifier.predict(vector_test)
    t2 = time.time()
    time_train = t1-t0
    time_predict = t2-t1
    print ('Tamaño del vector (filas x columnas): {} x {} '.format(len(vector_train), len(vector_train[0])))
    print('Time train: {} - Time predict: {}'.format(time_train, time_predict))
    print('Result MSE: {}'.format(mean_squared_error(labels_test, prediction)))
    print('Result Coseno: {}'.format(cosine_similarity(prediction.reshape(1,-1), numpy.array(labels_test))))
    print("\n")

def entrenar_gaussiana(num_entrenamiento, titulo, vector_train, labels_train, vector_test, labels_test):
    print("ENTRENAMIENTO: ", num_entrenamiento, " - ", titulo)
    clasifier = GaussianNB()
    t0 = time.time()
    clasifier.fit(vector_train.toarray(), numpy.array(labels_train).astype(int))
    t1 = time.time()
    prediction = clasifier.predict(vector_test.toarray())
    t2 = time.time()
    time_train = t1-t0
    time_predict = t2-t1
#    print ('Tamaño del vector (filas x columnas): {} x {} '.format(len(vector_train), len(vector_train[0])))
    print('Time train: {} - Time predict: {}'.format(time_train, time_predict))
    print('Result MSE: {}'.format(mean_squared_error(numpy.array(labels_test).astype(int), prediction)))
    
    print("\n")

def entrenar_gaussiana2(num_entrenamiento, titulo, vector_train, labels_train, vector_test, labels_test):
    print("ENTRENAMIENTO: ", num_entrenamiento, " - ", titulo)
    clasifier = svm.SVR()
    t0 = time.time()
    clasifier.fit(vector_train.toarray(), numpy.array(labels_train).astype(int))
    t1 = time.time()
    prediction = clasifier.predict(vector_test.toarray())
    t2 = time.time()
    time_train = t1-t0
    time_predict = t2-t1
#    print ('Tamaño del vector (filas x columnas): {} x {} '.format(len(vector_train), len(vector_train[0])))
    print('Time train: {} - Time predict: {}'.format(time_train, time_predict))
    print('Result MSE: {}'.format(mean_squared_error(numpy.array(labels_test).astype(int), prediction)))
    print('Result Coseno: {}'.format(cosine_similarity(prediction.reshape(1,-1), numpy.array(labels_test))))
    print("\n")
