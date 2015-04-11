#!/usr/bin/python

import sklearn, sklearn.linear_model, sklearn.ensemble, sklearn.naive_bayes, sklearn.neighbors
import random
from sklearn.preprocessing import OneHotEncoder

print "hello"



def transform(x):
	x = x.split(',')
	cls = -1
	
	if len(x) > 12:
		#print len(x)
		cls = x[1]
		del x[1]
	del x[2]
	del x[2]
	x[-1] = x[-1].replace('\r', "")
	if 'female' in x[2]:
		x[2] = 0
	else:
		x[2] = 1
		
	id = x[0]

	del x[0]
	
	for i,xx in enumerate(x):
		if x[i] == "":
			x[i] = -1
	

	print x

	
	if len(x) < 9:
		print "problem:", x
		return []
		
# 	for i in range(len(x)):
# 		try:
# 			#x[i] = float(x[i])
# 		except:
# 			pass
	try:
		return (int(id), x, int(cls))
	except:
		return (id, x, cls)
		
encoders = [sklearn.preprocessing.OneHotEncoder() for x in range(len([1,0,0,0,0,1,1,1,1]))]

def one_hot_encode_features(all_data, all_test, which = [1,0,0,0,0,1,0,1,1]):
	global encoders
	tmp = [list(elem) for elem in all_data]
	tmpt = [list(elem) for elem in all_test]
	curd = map(lambda x: x[1], all_data)
	curt = map(lambda x: x[1], all_test)
	#print curd
	encoded = []
	for idx, d in enumerate(which):
		if which[idx]==0:
			continue
		to_add = []
		group = map(lambda x: [hash(x[idx])%10000000], curd)
		group_test = map(lambda x: [hash(x[idx])%10000000], curt)
		
		print idx, group[0:2]
		encoders[idx].fit(group+group_test)
		
		to_add = encoders[idx].transform(group).toarray()
		for ii, t in enumerate(tmp):
			#print(to_add[idx])
			t[1]+=to_add[ii]

		to_addn = encoders[idx].transform(group_test).toarray()
		for ii, t in enumerate(tmpt):
			#print(to_add[idx])
			t[1]+=to_addn[ii]
	
	for idx, t in enumerate(tmp):
		for i in reversed(range(len(which))):
			if which[i] == 1:
				del t[1][i]	
		#print t[1]		
		t[1] = map( float, t[1])

	for idx, t in enumerate(tmpt):
		for i in reversed(range(len(which))):
			if which[i] == 1:
				del t[1][i]	
		#print t[1]		
		t[1] = map( float, t[1])
	
	
		
	return tmp, tmpt
data = []
test = []
with file('train.csv','r') as f:
	data = f.read()

with file('test.csv','r') as f:
	test = f.read()	

data = filter(lambda x: len(x)>2, map(transform, filter(lambda x: len(x) > 3, data.split('\n')))[1:])
random.shuffle(data)

test = filter(lambda x: len(x)>2,  map(transform, filter(lambda x: len(x) > 3, test.split('\n')))[1:])


(data, test) = one_hot_encode_features(data, test)


(train_data, train_class) = (map(lambda x:x[1], data), map(lambda x: x[2], data))
(ids, test_data) = (map(lambda x:x[0], test), map(lambda x:x[1], test))
# for d in train_data:
# 	print d

print train_data[1]



models = [sklearn.linear_model.LogisticRegression(penalty='l1', C=.03), sklearn.ensemble.RandomForestClassifier(n_estimators=150), sklearn.naive_bayes.BernoulliNB(), sklearn.ensemble.AdaBoostClassifier(), sklearn.neighbors.KNeighborsClassifier()]

print len(train_data), len(train_class)

for mm in models:
	mm.fit (train_data[:-300], train_class[:-300])


stack = map(list, zip(*[mm.predict(train_data) for mm in models]))

print stack

m = sklearn.ensemble.GradientBoostingClassifier(n_estimators=150)
m.fit(stack[-300:-150], train_class[-300:-150])


print m.score(stack[-150:], train_class[-150:])

test_stack = map(list, zip(*[mm.predict(test_data) for mm in models]))

results =  zip(ids, m.predict(test_stack))

with file('submit.csv','w') as f:
	f.write("PassengerId,Survived\n")
	to_write = ""
	for r in results:
		to_write+=(str(r[0])+","+str(r[1])+"\n")
	f.write(to_write[:-1])


