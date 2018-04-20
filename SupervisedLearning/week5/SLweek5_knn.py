import numpy as np
from sklearn import datasets, cross_validation as cv
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def to_file(answer, num):
	with open('knn_answer'+str(num)+'.txt', 'w') as fout:
		fout.write(str(answer))

def e_metric(x, y):
	return np.sqrt( np.sum((x - y)**2) )

if __name__ == "__main__":
	digits = load_digits()
	X_train, X_test, y_train, y_test = cv.train_test_split(digits.data, digits.target, test_size=0.75)
	y_pred_knn = []
	for test_value in X_test:
		ind_min_metric = 0
		min_metric = e_metric(test_value, X_train[0])
    
		for index, train_value in enumerate(X_train):
			metric = e_metric(test_value, train_value)
			if metric < min_metric:
				min_metric = metric
				ind_min_metric = index
            
		y_pred_knn.append(y_train[ind_min_metric])
	knn_err = 1 - accuracy_score(y_test, y_pred_knn)
	to_file(knn_err,1)
	rf_clf = RandomForestClassifier(n_estimators=1000)
	rf_clf.fit(X_train, y_train)
	y_pred_rf = rf_clf.predict(X_test)
	rf_err = 1 - accuracy_score(y_test, y_pred_rf)
	to_file(rf_err,2)