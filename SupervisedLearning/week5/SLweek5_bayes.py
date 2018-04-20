from sklearn import datasets
from sklearn import cross_validation
from sklearn import naive_bayes
import pandas as pd
import numpy as np

def to_file(answer, num):
	with open('bayes_answer'+str(num)+'.txt', 'w') as fout:
		fout.write(str(answer))

def get_ME(X,Y):
	return cross_validation.cross_val_score(naive_bayes.BernoulliNB(), X, Y).mean(), cross_validation.cross_val_score(naive_bayes.MultinomialNB(), X, Y).mean(), cross_validation.cross_val_score(naive_bayes.GaussianNB(), X, Y).mean()

def terminate(X,Y):
	m1,m2,m3 = get_ME(X,Y)
	print('BernoulliNB: {0}\n'.format(m1),'MultinomialNB: {0}\n'.format(m2),'GaussianNB: {0}'.format(m3))
	return max(m1,m2,m3)

if __name__ == '__main__':
	pack = [ datasets.load_digits(), datasets.load_breast_cancer() ]
	X, Y, M, N = pack[0].data, pack[0].target, pack[1].data, pack[1].target
	to_file(terminate(M,N),1)
	to_file(terminate(X,Y),2)
	to_file('3 4',3)

