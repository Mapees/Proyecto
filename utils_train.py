# Librerias


from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.svm import SVC
import numpy

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

# ------------------------------
#   UTILIDADES ENTRENAMIENTOS
# ------------------------------
# svm
def clas_svr(train, labels_train, test, labels_test):
    clasifier = SVR()
    clasifier.fit(train, labels_train)
    prediction = clasifier.predict(test)
    print('Result MSE: {}'.format(mean_squared_error(labels_test, prediction)))
    print('Result Coseno: {}'.format(cosine_similarity(prediction.reshape(1,-1), numpy.array(labels_test))))
    print('\n')

# Naive Bayes
def clas_naive(train, labels_train, test, labels_test):
    clasifier = GaussianNB()
    clasifier.fit(train, labels_train)
    prediction = clasifier.predict(test)
    print('Result MSE: {}'.format(mean_squared_error(labels_test, prediction)))
    print('Result Coseno: {}'.format(cosine_similarity(prediction.reshape(1,-1), numpy.array(labels_test))))
    print('\n')  
    
def clas_svc(train, labels_train, test, labels_test):
    clasifier = SVC()
    clasifier.fit(train, labels_train)
    prediction = clasifier.predict(test)
    print('Result MSE: {}'.format(mean_squared_error(labels_test, prediction)))
    print('Result Coseno: {}'.format(cosine_similarity(prediction.reshape(1,-1), numpy.array(labels_test))))
    print('\n')  
    


def iter_svc(train, labels_train, test, labels_test):
	# Anyadir tolerancia y max iters si no obtiene resultados: 'max_iter': [100], 'tol':[2, 3]
    tuned_parameters = [{'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'degree':[2, 3], 'random_state': ['None']},
                         {'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'degree':[2, 3], 'random_state': ['None']},
                         {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000], 'degree':[2, 3], 'random_state': ['None']}
                         ]
    
    scores = ['precision']
    
    clasifier = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % scores)
    clasifier.fit(train, labels_train)
    print("Best parameters set found on development set:")
    print()
    print(clasifier.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clasifier.cv_results_['mean_test_score']
    stds = clasifier.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clasifier.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()
    
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test, clasifier.predict(labels_test)
    print(classification_report(y_true, y_pred))
    print()
	
def iter_decision_tree_regresor(train, labels_train, test, labels_test):
	# criterion (default="mse")
    tuned_parameters=[{'max_features': ['int', 'float', 'string', 'None', 'auto', 'sqrt', 'log2'], 'max_depth': [2, 3], 'min_samples_split': [2]}]
    scores = ['precision']
    
    clasifier = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=5, scoring='%s_macro' % scores)
    clasifier.fit(train, labels_train)
	
	

