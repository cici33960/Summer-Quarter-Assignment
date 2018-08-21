#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 20:46:54 2018

@author: elenaxu
"""

import json
import random
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance
import numpy as np

f = open('titanic.json.txt', 'r')
d = json.load(f)

# Create data-set arrays from the JSON encoded data with the following 5 features:
# create array of zeros: # of independent variable x: 5, # of target variable y: length of raw data(d)
data = [[0 for x in range(0, 6)] for y in range(0, len(d))]

#Replace zeros with data values
for i in range(0, len(d)):
    #1
    if d[i]['Age'] == '':
        data[i][0] = None
    else:
        data[i][0] = float(d[i]['Age'])
    #2
    data[i][1] = float(d[i]['Fare'])
    #3
    data[i][2] = float(d[i]['SiblingsAndSpouses']) + float(d[i]['ParentsAndChildren'])
    #4
    if d[i]['Embarked'] == 'C':
        data[i][3] = 1.
    elif d[i]['Embarked'] == 'Q':
        data[i][3] = 2.
    elif d[i]['Embarked'] == 'S':
        data[i][3] = 3.
    else:
        data[i][3] = None
    #5
    if d[i]['Sex'] == 'male':
        data[i][4] = 0
    elif d[i]['Sex'] == 'female':
        data[i][4] = 1
    else:
        data[i][4] = None
    #target variable 
    data[i][5] = float(d[i]['Survived'])
print 'Raw data has',len(data), 'rows'

print 'Start to remove missing values'
cleandata = []
for xlist in data:
    if None not in xlist:
        cleandata.append(xlist)
print 'cleaned data has ', len(cleandata), 'rows'
print (len(cleandata)/float(len(data)))*100,'% of the total'

#normalize data
data=np.array(cleandata)

max_data=np.amax(data, axis=0)
min_data=np.amin(data, axis=0)

for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            data[i][j]= (data[i][j]-min_data[j]) / (max_data[j]-min_data[j])


#Dendrogram: Justify a cutting threshold and state the resulting number of clusters. 
#only row 0-4 are graphed independent variable
plt.figure(1)
Z = linkage(data[:, 0:5],method='ward',metric='euclidean')
dendrogram(Z)

plt.title('hierarchical clustering dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.axhline(y=9, color='black')
plt.savefig('Q2_Dendogram')
plt.clf()
print 'Cutting threshold: distance = 9, with 3 clusters'

#Start k-means clustering
plt.figure(2)
k=3
centroid = []
#initialize the cluster centers randomly #5 x-variables only
for i in range(k):
    centroid.append(data[random.randint(0, len(data)), 0:5]) 


#def dist(a, b):
#    return np.sqrt(np.sum((a-b)**2))

#plot the centroid

epoch=0
iteration=0

old_cluster1 = []
old_cluster2 = []
old_cluster3 = []

while epoch == 0 and iteration < 100:
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for row in data:
        dist1 = distance.euclidean(row[0:5], centroid[0])
        dist2 = distance.euclidean(row[0:5], centroid[1])
        dist3 = distance.euclidean(row[0:5], centroid[2])
        if dist1 < dist2 and dist1<dist3:
            cluster1.append(row)
        elif dist2 < dist1 and dist2<dist3:
            cluster2.append(row)
        elif dist3 < dist1 and dist3<dist2:
            cluster3.append(row)
            
            
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)
    cluster3 = np.array(cluster3)
   
    #check if clusters still changing 
    if np.array_equal (cluster1, old_cluster1) and np.array_equal (cluster2, old_cluster2) and np.array_equal (cluster3, old_cluster3):
        epoch += 1
        
    #recalculate the centroids
    #centroid=[]
    centroid[0] = [np.mean(cluster1[:, 0:5])]
    centroid[1] = [np.mean(cluster2[:, 0:5])]
    centroid[2] = [np.mean(cluster3[:, 0:5])]
    
    old_cluster1 = cluster1
    old_cluster2 = cluster2
    old_cluster3 = cluster3

    if iteration <10 or epoch == 1:
        featureNames = ['Age','Fare','Companions Count', 'Embarked Location', 'Sex', 'Survived']
        for i in range(k):
            for j in range(k):
                if i < j:
                    for n in range(len(cluster1)):
                        if cluster1[n][5] == 1:
                            plt.plot(cluster1[n][i], cluster1[n][j], 'gx')
                        else:
                            plt.plot(cluster1[n][i], cluster1[n][j], 'gs')
                    
                    for n in range(len(cluster2)):
                        if cluster2[n][5] == 1:
                            plt.plot(cluster2[n][i], cluster2[n][j], 'bx')
                        else:
                            plt.plot(cluster2[n][i], cluster2[n][j], 'bs')
                    
                    for n in range(len(cluster3)):
                        if cluster3[n][5] == 1:
                            plt.plot(cluster3[n][i], cluster3[n][j], 'yx')
                        else:	
                            plt.plot(cluster3[n][i], cluster3[n][j], 'ys')
                    
                    plt.plot(centroids[0][i], centroids[0][j], 'ko', label='cluster1 centroid')
                    plt.plot(centroids[1][i], centroids[1][j], 'k*', label='cluster2 centroid')
                    plt.plot(centroids[2][i], centroids[2][j], 'ks', label='cluster3 centroid')
                    plt.title(featureNames[i]+' vs. '+featureNames[j]+' iteration%s ' % iteration)
                    plt.xlabel(featureNames[i])
                    plt.ylabel(featureNames[j])
                    plt.legend()
                    #plt.show()
                    plt.savefig('%s_vs_%s_iteration_%i.png' % (featureNames[i], featureNames[j], iteration))
                    plt.clf()
	iteration += 1
    
#who survived in each cluster
survived1 = 0
survived2 = 0
survived3 = 0

for row in cluster1:
    if row[5] == 1:
        survived1 += 1
for row in cluster2:
    if row[5] == 1:
        survived2 += 1
for row in cluster3:
    if row[5] == 1:
        survived3 += 1

print str((survived1/float(len(cluster1))*100)) + '% survived in cluster 1'
print str((survived2/float(len(cluster2))*100)) + '% survived in cluster 2'
print str((survived3/float(len(cluster3))*100)) + '% survived in cluster 3'
