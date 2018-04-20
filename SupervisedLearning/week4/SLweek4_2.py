from sklearn import datasets, cross_validation as cv
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.metrics import mean_squared_error as MSE
from xgboost import XGBRegressor as XGBR
from sklearn.linear_model import LinearRegression as LR
from matplotlib import pyplot as plt
import numpy as np

def to_file(answer, num):
	print(answer)
	with open('gbanswer_'+str(num)+'.txt', 'w') as file:
		file.write(str(answer))

def square_grad(original_y, y_pred):
    grad = original_y - y_pred
    return grad

def gbm_predict(X, ests, coef):
    return [sum([coef * algo.predict([x])[0] for algo, coef in zip(ests, coef)]) 
                for x in X]

def function(Xt, Xy, Yt, Yy,weight, dynamic = False):
	for i in range(50):
		estimator = DTR(max_depth=5, random_state=42)
		if i == 0:
			estimator.fit(X_train, y_train)
		else:
			estimator.fit(X_train, square_grad(y_train, gbm_predict(X_train, ests, coef)))
		ests.append(estimator)
		if (dynamic):
			coef.append(weight/(1 + i))
		else:
			coef.append(weight)
	return MSE(y_test, gbm_predict(X_test, ests, coef))

def drawplot(a, cnts, err, title):
	plt.subplot(a)
	plt.grid(True)
	plt.plot(cnts, err)
	plt.xlabel('n_estimators')
	plt.ylabel('MSE')
	plt.title(title)

def get_errors(Xt, Xy, Yt, Yy, counts, Xtype = True):
	errors = []
	for cnt in counts:
		if (Xtype):
			xgb_reg = XGBR(n_estimators=cnt).fit(X_train, y_train)
		else:
			xgb_reg = XGBR(max_depth=cnt).fit(X_train, y_train)
		mse = MSE(y_test, xgb_reg.predict(X_test))
		errors.append(mse)
	return errors

if __name__ == '__main__':
	data = datasets.load_boston()
	X_train, X_test, y_train, y_test = cv.train_test_split(data.data, data.target, test_size=0.25)
	print(type(data.target[0]))
	print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
	ests, coef = [], []
	to_file(function(X_train, X_test, y_train, y_test,0.9),1)
	to_file(function(X_train, X_test, y_train, y_test,0.9, dynamic = True),2)
	counts = np.arange(1, 100, 10)
	err = get_errors(X_train, X_test, y_train, y_test, counts)
	drawplot(121, counts, err, 'MSE and n_estimators relationship')
	err = get_errors(X_train, X_test, y_train, y_test, counts, Xtype = False)
	drawplot(122, counts, err,'MSE and max_depth relationship')
	plt.tight_layout(-1)
	plt.show()
	to_file('2 3',3)
	linreg = LR().fit(X_train, y_train)
	rmse = mean_squared_error(y_test, linreg.predict(X_test)) ** 0.5
	to_file(rmse,4)



