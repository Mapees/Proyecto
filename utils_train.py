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

def clas_svm(train, labels_train, test, labels_test):
    clasifier = svm.SVR()
    clasifier.fit(train, labels_train)
    prediction = clasifier.predict(test)
    print('Result MSE: {}'.format(mean_squared_error(labels_test, prediction)))
    print('Result Coseno: {}'.format(cosine_similarity(prediction.reshape(1,-1), numpy.array(labels_test))))
    print('\n')
