from sklearn import datasets as ds, cross_validation, tree as tr, ensemble as ens, grid_search, model_selection as ms
import matplotlib.pyplot as plt
import numpy as np

def to_file(answer, num):
    with open('answer_'+str(num)+'.txt', 'w') as file:
        file.write(str(answer))

def check_quality(object,X,Y):
	object.fit(X,Y)
	quality = ms.cross_val_score(object, X, Y, cv = 10)
	print('Quality = ', quality, 'Quality means = ', quality.mean())
	return quality.mean()

def drawplot(grid, object, label):
	plt.plot(grid, object)
	plt.xlabel(label)
	plt.ylabel('Quality')
	plt.show()

def function(i, item):
	if i == 0:
		return ens.RandomForestClassifier(n_estimators=item)
	elif i == 1:
		return ens.RandomForestClassifier(max_features=item)
	else: 
		return ens.RandomForestClassifier(max_depth=item)

if __name__ == '__main__':
	data = ds.load_digits()
	X, Y = data.data, data.target
	classes = [ tr.DecisionTreeClassifier(), ens.BaggingClassifier(n_estimators=100), ens.BaggingClassifier(n_estimators=100, max_features=int(np.sqrt(X.shape[1]))), ens.BaggingClassifier(base_estimator=tr.DecisionTreeClassifier(max_features=int(np.sqrt(X.shape[1]))), n_estimators=100), ens.RandomForestClassifier()]
	for i in range(len(classes)):
		to_file(check_quality(classes[i],X,Y),i+1)
	labels = ['Esimators', 'Max features', 'Max Depth']
	grid = [ [5, 10, 15, 30, 50, 100, 150, 200, 300, 500], [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60], [2, 4, 6, 8, 10, 20, 30, 50, 100] ]
	Qarray = [[],[],[]]
	for i in range(len(grid)):
		for item in grid[i]:
			scores = ms.cross_val_score(function(i,item), X, Y, cv=5)
			Qarray[i].append(scores.mean())
	for i in range(len(Qarray)):
			drawplot(grid[i],Qarray[i],labels[i])
	to_file('2 3 4 7', 6)

