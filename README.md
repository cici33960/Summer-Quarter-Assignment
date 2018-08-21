# Summer-Quarter-Assignment
About CommandLines and Python

Assignment 1

Q1-clean up a dataset by using Linux CommandLines

Q2-replace words in Python

Q3-define a function to count all characters read in from standard input

Q4-count certain characters in Python

Q5-create a beta function in Python

Assignment 2

Question1

Import and use the Boston housing dataset to fit a linear regression model using Gradient Descent. Create a linear model, def a function to compute the result of the model, and def an RMSE function for this purpose. Partition the Boston dataset into training and holdout (test) with a ratio of 90% to 10%, respectively. Terminate learning after a maximum of 10 epochs, and create a plot of the RMS error of your model after each epoch. Repeat the experiment for each of the following learning rates:0.00001, 0.0001, 0.001, 0.01, 0.1, 1.Save the images.

Hint : For extending the two-feature case to any number of features, notice:
the update rule:
b0 = b0 - learning_rate*error
can be re-written as:
b0 = b0 - learning_rate*error*x0
where x0 = 1 (by convention)
Use this observation to derive expressions for the higher order coefficients. 

Question2

Train a Decision Tree Classifier to predict the “Very Late Adopter” class using the customer dataset from Assignment 1 (a1_dataset.dat. Your program reads in data from a csv file or other structured data file. Your program takes 10% of the data and sets it aside as a test set. Use the rest as a training set for all parts of this question. Use your data processing skills to turn the problem into a 2-class n-feature classification (class prediction) problem, for example, modify the target vector to indicate 1 for “Very Late Adopter” and 0 to mean “not a Very Late Adopter”.
 
a) Train the model on the entire input data set (which is biased towards samples that are not “Very Late Adopter” - verify this for yourself using grep or something), 
b) also with pruning the input data so that there is a 50:50 ratio of samples of each target class: “Very Late Adopter” OR not “Very Late Adopter” For each mix of data don’t forget to reserve 10% for the testing set.
 
Plot the performance (Best Accuracy) on the test set when meta-parameter max_leaf_nodes is varied:
max_leaf_nodes: 2, 4, 8, 16, 32, 64, 128, 256 and max_depth is left to None (default). Similarly plot the performance when varying max_depth in the same way, using max_depth:use this as learning rate(one max_depth is one graph, with different leaf nodes) 2, 4, 8, 16 (10 Plots total). Leaving max_leaf_nodes = None (default) 

Assignment 3

Dendrogram and Clustering-1

a) Perform hierarchical clustering analysis using the tools presented in class. Justify a cutting threshold and state the resulting number of clusters. Plot the dendrogram. 
b) Complete the k-means code we began in class. Using the number of clusters (k) you arrived at in the previous step, recompute the centroid of each of your clusters until termination criteria are reached or 100 iterations, whichever happens first. Plot the cluster centroids after: 0 iterations (initial guess); 5 iterations; 10 iterations, 100 iterations (or last iteration if the algorithm completed early) 

Dendrogram and Clustering-Titanic

a) Your program MUST read the provided data (titanic.json) and analyze the data within the provided file.
 
b) Create data-set arrays from the JSON encoded data with the following 5 features:
-age
-fare
-combined Sibling/Spouse count and Parent/Children count into “companions count” feature column
-embarked location (C = Cherbourg, Q = Queenstown, S = Southampton - think about how to encode this)
-sex
Think about how to handle missing values, and how to handle the survived “target” directly in the data or to use it to to colour or otherwise distinguish survivors and non survivors in the plots. 
 
c) Perform hierarchical clustering analysis using the tools presented in class. Justify a cutting threshold and state the resulting number of clusters. Plot the dendrogram.

d) Use the k-means code completed in Q1 to cluster the passengers aboard the Titanic.  
Plot the cluster centroids after: 0 iterations (initial guess); 5 iterations; 10 iterations, 100 iterations (or last iteration if the algorithm completed early).
When plotting you may consider colouring survivors and non-survivors within your clusters. You may consider and justify ways to simplify the dataset further.


