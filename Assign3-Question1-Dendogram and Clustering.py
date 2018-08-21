from sklearn import cluster, datasets
from matplotlib import pyplot as plt
import random
import csv
import numpy as np

with open('A3_Q1_dataset.csv') as csvfile:
    readcsv=csv.reader(csvfile, delimiter=',')
    dataset=[]
    for point in readcsv:
        for i in range(len(point)):
            point[i]=float(point[i])
            dataset.append(point)
        
x = dataset

def normalization(x,max_x,min_x):
    x = (x-min_x)/(max_x - min_x)
    return x

x = normalization (x,np.max(x),np.min(x))
x=np.array(x)

        
#target variable y is not used in unsupervised learning!

#perform hierarchial clustering
from scipy.cluster.hierarchy import dendrogram, linkage
plt.figure(1)
#save the clustering results in variable z
Z = linkage(x, method='ward', metric='euclidean')
#create a dendrogram from the result of the hierarchial clustering
dendrogram(Z) 

plt.title("Hierarchial Clustering Dendrogram")
plt.xlabel("sample index")
plt.ylabel("distance")
plt.savefig('Q1a_Dendrogram')

print 'declare cluster centers (list of 3): decide to cut it into 3 clusters: between the clusters that have distance 20 and 80 (because of the long distance, the similarity seems weak)'



k=3
centroid = []
#initialize the cluster centers randomly
for i in range(k):
    centroid.append( x[random.randint(0,x.shape[0])] )


plt.figure(2)

#plot all data points x-axis=0th column;y=axis=1st column
#plt.scatter(x[:,0], x[:,1])  #[all rows, 0th column]

def dist(a, b):
    return np.sqrt(np.sum((a-b)**2))

#plot the centroid
epoch=0
iteration=0

old_cluster1 = []
old_cluster2 = []
old_cluster3 = []

while epoch == 0 and iteration <= 100:
    cluster1 = []
    cluster2 = []
    cluster3 = []
    for i in range(len(x)):
        dist1 = dist(x[i], centroid[0])
        dist2 = dist(x[i], centroid[1])
        dist3 = dist(x[i], centroid[2])
        if dist1 < dist2 and dist1<dist3:
            cluster1.append(x[i])
        elif dist2 < dist1 and dist2<dist3:
            cluster2.append(x[i])
        elif dist3 < dist1 and dist3<dist2:
            cluster3.append(x[i])
            
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)
    cluster3 = np.array(cluster3)
   
    #recalculate the centroids
    if np.array_equal (cluster1, old_cluster1) and np.array_equal (cluster2, old_cluster2) and np.array_equal (cluster3, old_cluster3):
        epoch += 1
    
    centroid=[]
    centroid.append([np.mean(cluster1[:, 0]), np.mean(cluster1[:, 1])])
    centroid.append([np.mean(cluster2[:, 0]), np.mean(cluster2[:, 1])])
    centroid.append([np.mean(cluster3[:, 0]), np.mean(cluster3[:, 1])])
    #centroid = [cluster1_mean, cluster2_mean, cluster3_mean]
    
    old_cluster1 = cluster1
    old_cluster2 = cluster2
    old_cluster3 = cluster3

    #since the iteration doesn't seem go more than 100, or rarely stop right at 0,5,10,100
    #therefore, I have changed this a little 

    if iteration in (0,1,2,3,4,5,6,7,8,9,10) or epoch == 1:
        plt.scatter(cluster1[:, 0], cluster1[:, 1], c='b', marker='x')
        plt.scatter(cluster2[:, 0], cluster2[:, 1], c='g', marker='x')
        plt.scatter(cluster3[:, 0], cluster3[:, 1], c='y', marker='x')
        for i in range(k):
            plt.scatter(centroid[i][0] ,centroid[i][1], s=100, c='r')
        plt.title('Clustering after iteration %s' % iteration)
        plt.savefig('Q1b_Iteration %s' % iteration)
        plt.show()
	iteration += 1


