'''
Train a Decision Tree Classifier to predict the “Very Late Adopter” class using the customer dataset from Q1-a1_dataset.dat
'''

import sys
import optparse
import numpy as np
from sklearn import tree
from sklearn.preprocessing import LabelEncoder as label
import collections, numpy
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle

data=[]
def processText(line):
	line = line.strip().split(',')
	data.append(line)

parser = optparse.OptionParser()
parser.add_option('-f', '--file', dest='fileName', help="read from file")
parser.add_option('-v', '--verbose', action="store_true", dest='verb', help="pruning the target")

(options,others) = parser.parse_args()
usingFile = False
prune = False

if options.fileName is None:
	print "DEBUG: the user did not enter the -f option"
else:
	print "DEBUG: the user entered the -f option"
	usingFile = True

if options.verb is None:
	print "DEBUG: the user did not enter the -v option, Q2a"
else:
	print "DEBUG: the user entered the -v option, Q2b-pruning the target for 50%-50% sample size "
	prune = True


if (usingFile == True):
	print options.fileName
	file = open(options.fileName, 'r')
	for line in file:
		processText(line)
else:
	for line in sys.stdin.readlines():
		processText(line)

data=np.array(data)
#print data.shape (4662,10)

#normalize data into 2-class

for i in range(0,data.shape[0]):
	data[i][0] = int(data[i][0])	
	if data[i][1] == 'M':
		data[i][1] = 1
	else: 
		data[i][1] = 0
	data[i][2] = int(data[i][2])

	if data[i][3] == 'Married':
		data[i][3] = 1
	else: 
		data[i][3] = 0
	if data[i][4] == 'Prepaid':
		data[i][4] = 0
	elif data[i][4] == 'Low': 
		data[i][4] = 1
	elif data[i][4] == 'Medium': 
		data[i][4] = 2
	else:
		data[i][4] = 3
	if data[i][5] == 'Automatic':
		data[i][5] = 1
	else: 
		data[i][5] = 0
	if data[i][6] == 'No Contract':
		data[i][6] = 0
	elif data[i][6] == '12 Months': 
		data[i][6] = 1
	elif data[i][6] == '24 Months': 
		data[i][6] = 2
	else: 
		data[i][6] = 3
	if data[i][7] == 'Y':
		data[i][7] = 1
	else: 
		data[i][7] = 0
	if data[i][8] == 'Y':
		data[i][8] = 1
	else: 
		data[i][8] = 0
	if data[i][9] == 'Very Late':
		data[i][9] = 1
	else: 
		data[i][9] = 0

#print data

target = data[:,-1] #give the entire last column
feature = np.delete(data,-1,axis=1) #delete the last column. the rest are features
#print target, feature

testX=[]
trainingX=[]
testY=[]
trainingY=[]
for i in range(0,data.shape[0]):
	if i % 10 == 0:
		testX.append(feature[i])
		testY.append(target[i])
	else:
		trainingX.append(feature[i])
		trainingY.append(target[i])
testX=np.array(testX)
trainingX=np.array(trainingX)
testY=np.array(testY)
trainingY=np.array(trainingY)
#print trainingX.shape (4195,9)
#print trainingY.shape (4195,)
#print testY.shape (467,)

#create a tree and test with test data

clf=tree.DecisionTreeClassifier()
clf.fit(trainingX,trainingY)

correct=0
incorrect=0
predictions=clf.predict(testX) 	
for i in range(0,predictions.shape[0]):
	if predictions[i] == testY[i]:
		correct += 1
	else:
		incorrect += 1
accuracy=float(correct)/(correct+incorrect)
#print correct, incorrect, accuracy (333,134,0.700)


#Q2b-prune the target
if options.verb == True:
		#count the 0s and 1s in target (0:3535, 1:1127)
	print "Before pruning, this is the 0s and 1s in the target: "
	print collections.Counter(target)
	
		#shuffle data first; then sort data; then cut the first 2408 rows; then shuffle again

	data=shuffle(data,random_state=0)
	#print data
	sortdata=data[np.argsort(data[:, 9])]
	sortdata=np.delete(sortdata,numpy.s_[0:2408],axis=0)
	sortdata=shuffle(sortdata,random_state=0)
	#print sortdata

	#print sortdata.shape (2254,10)

		#recreate target and feature, training and test data	
	target = sortdata[:,-1] 
	feature = np.delete(sortdata,-1,axis=1) 
	testX=[]
	trainingX=[]
	testY=[]
	trainingY=[]
	for i in range(0,sortdata.shape[0]):
		if i % 10 == 0:
			testX.append(feature[i])
			testY.append(target[i])
		else:
			trainingX.append(feature[i])
			trainingY.append(target[i])
	testX=np.array(testX)
	trainingX=np.array(trainingX)
	testY=np.array(testY)
	trainingY=np.array(trainingY)
	#print trainingX.shape (2028,9)
	#print trainingY.shape (2028,)
	#print testY.shape (226,)
	print "After pruning, this is the 0s and 1s in the target: "
	print collections.Counter(target)



max_depth = [None,2,4,8,16]
max_leaf_nodes = [2,4,8,16,32,64,128,256]

for a in max_depth:
	accuracy_list=[]
	#clean up the accuracy list for every depth
	for j in max_leaf_nodes:
		clf=tree.DecisionTreeClassifier(max_depth=a,max_leaf_nodes=j)
		clf.fit(trainingX,trainingY)

		correct=0
		incorrect=0
		predictions=clf.predict(testX) 
		for i in range(0,predictions.shape[0]):
			if predictions[i] == testY[i]:
				correct += 1
			else:
				incorrect += 1
		accuracy=float(correct)/(correct+incorrect)
		accuracy_list.append(accuracy)

	plt.plot(max_leaf_nodes,accuracy_list)
	plt.xlabel('max leaf nodes')
	plt.ylabel('accuracy')
	plt.title('max leaf nodes vs accuracy')
	plt.show()
