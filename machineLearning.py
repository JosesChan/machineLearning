from cmath import sqrt
import numpy
import numpy as np
import numpy.linalg as linalg
import matplotlib as plt
import matplotlib.pyplot as plt
import pandas
import seaborn

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate, train_test_split

polyTest = pandas.read_csv("Task1 - dataset - pol_regression.csv")
dataframe = pandas.read_csv("Task2 - dataset - dog_breeds.csv")
hivDataset = pandas.read_csv("Task3 - dataset - HIV RVG.csv")
dataset = dataframe.values

# Section 1.1: Implementation of Polynomial Regression (10%).
# You are asked to implement the polynomial regression algorithm. To do so, you are required to
# use the following function:

def pol_regression(features_train, y_train, degree):
    X = getPolynomialDataMatrix(features_train, degree)
    XX = X.transpose().dot(X)
    if (degree == 0):
        average = numpy.average(y_train)
        meanList = numpy.ones(1,)
        w = numpy.multiply(meanList, average)
        return w
    w = numpy.linalg.solve(XX, X.transpose().dot(y_train))
    return w  

# Section 1.2: Regress a polynomial of the following degrees: 0, 1, 2, 3, 6, 10 (10%).
# After implementing the pol_regression function, use this function to regress the data set given in
# the .csv file of task 1 as part of this assignment. Regress a polynomial of the following degrees: 0,
# 1, 2, 3, 6, 10. Plot the resulting polynomial in the range of [-5, 5] for the inputs x. In addition, plot
# the training points. Interpret and elaborate on your results. Which polynomial function would you
# choose?


x_values = polyTest['x']
y_values = polyTest['y']
x_values_sorted = numpy.sort(x_values)

degreesList = [0, 1, 2, 3, 6, 10]

def getPolynomialDataMatrix(x, degree):
    X = numpy.ones(x.shape)
    for i in range(1,degree + 1):
        X = numpy.column_stack((X, x ** i))
    return X
    
# evaluatate polynomial coefficients
def polynomialCoefficients(x, coefficients):
    o = len(coefficients)
    y = 0
    for i in range(o):
        y += coefficients[i]*x**i
    return y


plt.plot(x_values, y_values, "ro")

w0 = pol_regression(x_values,y_values,0)
coefficients0 = polynomialCoefficients(x_values_sorted,w0)
plt.plot(x_values, coefficients0, 'b')

w1 = pol_regression(x_values,y_values,1)
coefficients1 = polynomialCoefficients(x_values_sorted,w1)
plt.plot(x_values_sorted, coefficients1, 'g')

w2 = pol_regression(x_values,y_values,2)
coefficients2 = polynomialCoefficients(x_values_sorted,w2)
plt.plot(x_values_sorted, coefficients2, 'y')

w3 = pol_regression(x_values,y_values,3)
coefficients3 = polynomialCoefficients(x_values_sorted,w3)
plt.plot(x_values_sorted, coefficients3, 'm')

w6 = pol_regression(x_values,y_values,6)
coefficients6 = polynomialCoefficients(x_values_sorted,w6)
plt.plot(x_values_sorted, coefficients6, 'r')

w10 = pol_regression(x_values,y_values,10)
coefficients10 = polynomialCoefficients(x_values_sorted,w10)
plt.plot(x_values_sorted, coefficients10, 'c')
    
plt.xlim((-5, 5))
plt.legend(('training points', '$x^{0}$', '$x^{1}$', '$x^{2}$', '$x^{3}$','$x^{6}$', '$x^{10}$'), loc = 'lower right')
plt.show()
plt.savefig('polynomial.png')


# Section 1.3: Evaluation (10%).
# Using the returned results of the pol_regression function from the previous section, you need now
# to evaluate the performance of your algorithm. To do so, you first need to implement the following
# function:

# may output with imaginary units
def eval_pol_regression(parameters, x, y, degree):
    # Sum squared error
    xTest = getPolynomialDataMatrix(x, degree)
    yTest = xTest.dot(parameters)
    error = y-yTest
    sse = error.dot(error)
    mse = sse/len(x)
    rmse = sqrt(mse)
    return (rmse)

# rmse0 = eval_pol_regression(w0, x_values, y_values, 0)
rmse1 = eval_pol_regression(w1, x_values, y_values, 1)
rmse2 = eval_pol_regression(w2, x_values, y_values, 2)
rmse3 = eval_pol_regression(w3, x_values, y_values, 3)
rmse6 = eval_pol_regression(w6, x_values, y_values, 6)
rmse10 = eval_pol_regression(w10, x_values, y_values, 10)



# The eval_pol_regression function takes as arguments the parameters computed from the
# pol_regression function and evaluates the algorithm’s performance on the input x and output y
# data (again 1D numpy arrays). The last argument of the function again specifies the degree of the
# polynomial. In this function, you need to compute the root mean squared error (rmse) of the
# polynomial given by the parameters’ vector. You should avoid using pre-built functions like
# numpy.polyval when evaluating a polynomial.
# However, this time, split the same dataset provided into 70% train and 30% test set. You can use
# the “train_test_split” function in sklearn library since you are not asked here to implement
# polynomial regression functions.
# Once again you need to train your polynomial functions that you developed in the previous section
# with degrees 0, 1, 2, 3, 6 and 10 on split training data.
# Evaluate the training set rmse and the test set rmse of all the given degrees. Plot both rmse values
# using the degree of the polynomial as x-axis of the plot. Interpret your results. Which degree would
# you now choose? Are there any degrees of the polynomials where you can clearly identify over
# and underfitting? Explain your findings in great detail. 


xTrainDataT1, xTestDataT1, yTrainDataT1, yTestDataT1 = train_test_split(polyTest['x'], polyTest['y'], train_size = 0.70, test_size = 0.30)

# #training
# w0 = pol_regression(x_values,y_values,0)
# coefficients0 = polynomialCoefficients(x_values_sorted,w0)

# w1 = pol_regression(x_values,y_values,1)
# coefficients1 = polynomialCoefficients(x_values_sorted,w1)

# w2 = pol_regression(x_values,y_values,2)
# coefficients2 = polynomialCoefficients(x_values_sorted,w2)

# w3 = pol_regression(x_values,y_values,3)
# coefficients3 = polynomialCoefficients(x_values_sorted,w3)

# w6 = pol_regression(x_values,y_values,6)
# coefficients6 = polynomialCoefficients(x_values_sorted,w6)

# w10 = pol_regression(x_values,y_values,10)
# coefficients10 = polynomialCoefficients(x_values_sorted,w10)

# # testing
# w0 = pol_regression(x_values,y_values,0)
# coefficients0 = polynomialCoefficients(x_values_sorted,w0)

# w1 = pol_regression(x_values,y_values,1)
# coefficients1 = polynomialCoefficients(x_values_sorted,w1)

# w2 = pol_regression(x_values,y_values,2)
# coefficients2 = polynomialCoefficients(x_values_sorted,w2)

# w3 = pol_regression(x_values,y_values,3)
# coefficients3 = polynomialCoefficients(x_values_sorted,w3)

# w6 = pol_regression(x_values,y_values,6)
# coefficients6 = polynomialCoefficients(x_values_sorted,w6)

# w10 = pol_regression(x_values,y_values,10)
# coefficients10 = polynomialCoefficients(x_values_sorted,w10)

# Task 2 (30%): In this task, several data for specific dog breeds have been collected. You will need
# to download the Task2 - dataset - dog_breeds.csv file. The data include four features, which are
# height, tail length, leg length, and nose circumference for each dog breed. The purpose of this
# experiment is to figure out whether these features could be used to cluster the dog breeds into
# three groups/clusters.
# Section 2.1: Implementation of the K-Means Clustering (15%).
# Implement the (simple) K-Means Clustering algorithm by creating the following functions
# • The function compute_euclidean_distance() calculates the distance of two vectors, e.g.,
# Euclidean distance
# • The function initialise_centroids() randomly initializes the centroids
# • The function kmeans() clusters the data into k groups 

# calculates the distance between samples
def compute_euclidean_distance(vec_1, vec_2):
    length = len(vec_1)
    euclidDistance = 0
    for i in range(length):
        euclidDistance += pow(vec_1[i] - vec_2[i],2)#power of 2
    return np.sqrt(euclidDistance)

# intialise center of clusters
def initialise_centroids(dataset, k):
    centroids = np.zeros([k,4]) 
    for i in range(k):
        randomInt = np.random.randint(0,len(dataset))
        centroids[i] = dataset[randomInt]    
    return centroids


# cluster into k amount of groups, calculating average distance between centroids and data
def kmeans(dataset, k):
    
    # initialise centroids and other variables
    centroids = initialise_centroids(dataset, k) #creating the first centroids
    distance = np.zeros(k)
    clusters = np.zeros(len(dataset)) 
    datapoints = [] # all summation values for error function

    # number of iterations for recalculation
    for i in range(12):
        distance, errorFunction, clusters, sumMinCentroids = recalculateClusters(distance, centroids, clusters, dataset, k)
    
        # append closest distance, to get error function
        datapoints.append(errorFunction)
        # recalculate centroids
        centroids = recalculateCentroids(sumMinCentroids, clusters, dataset, k)
    return centroids, clusters, datapoints



# meanCentroid holds the average distance of each centroid
# clusters holds the available types of clusters data can be categorized under
# using the parameters, it recalculates centroid locations
def recalculateCentroids(sumMinCentroids, clusters, dataset, k):
    # variable to store centroid locations
    newCentroids = np.zeros([k,4])
    # recalculate each centroid
    for selectedCentroid in range(k):
        for features in range(4):
            # average of each centroid and assigns as new centroid
            newCentroids[selectedCentroid][features] = (sumMinCentroids[selectedCentroid][features])/(len(dataset[clusters==selectedCentroid]))
    return newCentroids

# distance represents the distance between centroid and its closest datapoint
def recalculateClusters(distance, centroids, clusters, dataset, k):
    sumMinCentroids = np.zeros([k,4])
    errorFunction = 0
    # for entire dataset, find closest datapoints
    for currentDataPoint in range(len(dataset)):    
        for selectedCentroid in range(k):
            
            distance[selectedCentroid] = compute_euclidean_distance(dataset[currentDataPoint], centroids[selectedCentroid])
            
        # find datapoint with minimum distance through index values
        closest = np.where(distance == np.min(distance))[0][0]
            
        # post processing
        errorFunction += distance[closest]  
        clusters[currentDataPoint] = closest      
        sumMinCentroids[closest] += dataset[currentDataPoint]
    return distance, errorFunction, clusters, sumMinCentroids

# plots objective/error function graph
def plotErrorFunction(datapoints):
    plt.plot(range(len(datapoints)), datapoints, 'o-', label='Error Function')
    plt.xlabel("Iteration Step")
    plt.ylabel("Objective Function")
    plt.legend()
    plt.show()

# plot cluster graphs
def plotCluster(dataset, centroids, clusters, x, y, labels):
    #only showing column 0 1 or 0 and 2 (height and tail-length/ leg-length)
    plt.scatter(dataset[:,x],dataset[:,y], c = clusters)
    plt.scatter(centroids[:,x], centroids[:,y], c = 'r')
    plt.xlabel(labels[x])
    plt.ylabel(labels[y])
    plt.show()    


# centroids, clusters, datapoints = kmeans(dataset, 2)
# plotErrorFunction(datapoints)
# plotCluster(dataset,centroids,clusters,0,1,dataframe.columns) 
# plotCluster(dataset,centroids,clusters,0,2,dataframe.columns) 
# plotCluster(dataset,centroids,clusters,0,3,dataframe.columns) 


# centroids, clusters, datapoints = kmeans(dataset, 3)
# plotErrorFunction(datapoints)
# plotCluster(dataset,centroids,clusters,0,1,dataframe.columns) 
# plotCluster(dataset,centroids,clusters,0,2,dataframe.columns) 
# plotCluster(dataset,centroids,clusters,0,3,dataframe.columns) 


# HIV (human immunodeficiency virus) is a virus that attacks the body's immune system. If HIV is not
# treated, it can lead to AIDS (acquired immunodeficiency syndrome). In this task, you are required
# to develop machine learning models to predict HIV infection from measured features. The dataset
# “Task3 - dataset - HIV RVG.csv” provided consists of clinical records of both HIV patients and
# controls (healthy). The data has 5 retinal vascular geometry (RVG) features extracted from retinal
# images; the RVG features are described in [1].
# Image
# number
# Bifurcation
# number
# Artery (1)/
# Vein (2) Alpha Beta Lambda Lambda1 Lambda2 Participant
# Condition
# 1 1 1 0.60009863 2.14118476 0.77466034 1.15678779 0.89611761 Patient
# 1 2 1 0.82261224 1.83585755 0.90697974 1.00362703 0.91026938 Patient
# Each image consists of a set of bifurcations. A bifurcation can be either artery or vein. The
# extracted features from each bifurcation are Alpha, Beta, Lambda, Lambda1 and Lmbda2.
# Participant condition is either patient or control.
# [1] Al-Diri, B. and Hunter, A., 2009. Automated measurements of retinal bifurcations. In World
# Congress on Medical Physics and Biomedical Engineering, September 7-12, 2009, Munich,
# Germany (pp. 205-208). Springer, Berlin, Heidelberg.
# The class membership of each row is stored in the field “Status”. Status refers to the health
# condition of patients, or in other words, we consider this to be our label/annotation for the sake
# of all implementations. Unit of measurement or range of values of each feature are not relevant.
# However, features can be at different scales and/or measured in different units. Our task is to
# develop a set of classification models for automatically classifying patients as healthy or HIV
# infected, based on the RVG features. No prior knowledge of the domain problem is needed or
# assumed to fulfil the requirements of this assessment whatsoever.

# Section 3.1: Data import, summary, pre-processing and visualisation (10%)
# As a first step, you need to load the data set from the .xlsx file into a Python IDE. You should then
# provide a statistical summary of the dataset (i.e. mean values, standard deviations, min/max values
# for each feature). In data pre-processing, you need to think about whether you should normalise
# the data before starting training/testing any model and justify your decision.
# To visualise the data, you need to generate two plots. The first one shall be a box plot, which will
# include the two classes (“Status”), i.e. control/patient, in the x-axis and the “Alpha” in the y-axis.
# The second one shall be a density plot for the feature “Beta”, with the graphs of both classes
# appearing in the same plot. What information can be obtained from each of these two plots? Can
# one use any of these two plots to identify outliers? If yes, please elaborate.
# Please include your explanation of implementation alongside the plots.


# box plot of status classes (Participant Condition) and alpha
seaborn.boxplot(x=hivDataset["Participant Condition"], y=hivDataset["Alpha"])

# density plot 
seaborn.displot(x=hivDataset["Beta"], kind = "kde", hue=hivDataset["Participant Condition"])

plt.show()

patientHivDataset = hivDataset.loc[hivDataset['Participant Condition'] == "Patient"]
controlHivDataset = hivDataset.loc[hivDataset["Participant Condition"] == "Control"]

featureNames = hivDataset.columns.values
featureNames.drop(columns=["Image number","Bifurcation number","Artery (1)/ Vein (2)","Participant Condition"])

# display sum statistics
for i in featureNames:
    sumStatisticsTable = {"Participant Condition": ["Patient","Control"], "Minimum": [patientHivDataset[i].min(), controlHivDataset[i].min()], 
    "Maximum": [patientHivDataset[i].max(), controlHivDataset[i].max()], "Median": [patientHivDataset[i].median(), controlHivDataset[i].median()], 
    "Mean": [patientHivDataset[i].mean(),controlHivDataset[i].mean()], "Variance": [controlHivDataset[i].var(), controlHivDataset[i].var()], 
    "Standard Devation": [patientHivDataset[i].std(), controlHivDataset[i].std()]} 
    print(pandas.DataFrame(data=sumStatisticsTable))

# if multiple mode values, display all
for i in featureNames:
    print("Patient Mode Values ", featureNames[i])
    print(patientHivDataset.mode())

    print("Control Mode Values ", featureNames[i])
    print(controlHivDataset.mode())

# normalise the dataset
normalisedHivDataset=(hivDataset-hivDataset.min())/(hivDataset.max()-hivDataset.min())


# Section 3.2 Designing algorithms (15%)
# You will now design an artificial neural network (ANN) classifier for classifying patients as patient
# or control, based on their RVG features. You will use the provided data set to train the model. To
# design an ANN, use a fully connected neural network architecture with two hidden layers; use the
# sigmoid function as the non-linear activation function for the hidden layers and logistic function
# for the output layer; set the number of neurons in each hidden layer to 500. Now randomly choose
# 90% of the data as training set, and the rest 10% as test set. Train the ANN using the training data,
# and calculate the accuracy, i.e. the fraction of properly classified cases, using the test set. Please
# report how you split the data into training and test sets. In addition, please report the steps
# undertaken to train the ANN in detail and present the accuracy results.
# You will try different numbers of epochs to monitor how accuracy changes as the algorithm keeps
# learning, which you can plot using the number of epochs in the ‘x’ axis and the accuracy in ‘y’ axis
# [it should show the change of accuracy as the number of epochs increases].
# Now use the same training and test data to train a random forests classifier with 1000 trees. The
# minimum number of samples required to be at a leaf node has two options, i.e. 5 and 10. Please
# report the steps for training random forests for both options and show their accuracy results on
# the test set.

# x are features to be fed into the classifiers
# y are labels that are to be predicted by the classifiers
xHivDataset = normalisedHivDataset.values
xHivDataset = normalisedHivDataset.iloc[: , :-1]
yHivDataset =  normalisedHivDataset["Participant Condition"].values

xTrainData, xTestData, yTrainData, yTestData = train_test_split(xHivDataset, yHivDataset, train_size = 0.90, test_size = 0.10)

# variable to hold epoch amount
epochAmount = 200

# create multilayer perceptron with 2 hidden layers of size 500 each. Set epoch amount
mlpClassifier = MLPClassifier(hidden_layer_sizes=(500,500), activation="logistic", max_iter=epochAmount)

mlpClassifier.fit(xTrainData, yTrainData)

mlpClassifier.predict(xTestData)

print(mlpClassifier.score(xTestData, yTestData))

forestClassifier = RandomForestClassifier(n_estimators=1000, min_samples_leaf=5)

forestClassifier.fit(xTrainData, yTrainData)

forestClassifier.predict(xTestData)

print(forestClassifier.score(xTestData, yTestData))


# Section 3.3: Model selection (15%)
# You have designed ANN and random forests classifiers with almost fixed model parameters (e.g.
# number of neuros for ANN, number of trees for random forests, etc). The performance of the
# model could vary when those model parameters are changed. You would like to understand, which
# set of parameters are preferable, and also to select the best set of parameters given a range of
# options for ANN and random forests. To do so, one method is to employ a cross-validation (CV)
# process. In this task, you are asked to use a 10-fold CV. As a first step, randomly split the data into
# 10 folds of nearly equal size, and report the steps undertaken to do this.
# For ANN, create three ANN classifiers with 50, 500, and 1000 neuros in each hidden layer
# (remember there are two hidden layers), respectively. Similarly, create three random forest
# classifiers with 50, 500, and 10000 trees, respectively, and set the “minimum number of samples
# required to be at a leaf node” =10 for all of them.
# Please do the following tasks for both methods:
# 1) Use the 10-fold CV method to choose the best number of neurons or number of trees for
# ANN and random forests respectively.
# a) Report the processes involved when applying CV to each model.
# b) Report the mean accuracy results for each set of parameters, i.e. for different number of
# neurons and different number of trees accordingly.
# c) Which parameters should we use for each of the two methods, i.e. specifically for ANN
# and random forests?
# 2) Until now, you have selected the best parameters for each method, but we have not decided
# the best model yet. With the results you have had so far,
# a) which method is best, ANN or random forest?
# b) Please discuss and justify your choice, reflecting upon your knowledge thus far. 

# Create number of folds
folds = 10
# first n_samples%n_splits folds have the size of n_samples//n_splits+1, the rest are size of n_samples//n_splits
kfolds = KFold(n_splits=folds)

mlpClassifier1 = MLPClassifier(hidden_layer_sizes=(50,50), activation="logistic", max_iter=epochAmount)
mlpCV1 = cross_validate(mlpClassifier1, xHivDataset, yHivDataset, cv=kfolds, n_jobs=-1)
print(mlpCV1["test_score"].mean())
mlpClassifier2 = MLPClassifier(hidden_layer_sizes=(500,500), activation="logistic", max_iter=epochAmount)
mlpCV2 = cross_validate(mlpClassifier2, xHivDataset, yHivDataset, cv=kfolds, n_jobs=-1)
print(mlpCV2["test_score"].mean())
mlpClassifier3 = MLPClassifier(hidden_layer_sizes=(1000,1000), activation="logistic", max_iter=epochAmount)
mlpCV3 = cross_validate(mlpClassifier3, xHivDataset, yHivDataset, cv=kfolds, n_jobs=-1)
print(mlpCV3["test_score"].mean())

forestClassifier1 = RandomForestClassifier(n_estimators=50, min_samples_leaf=10)
forestCV1 = cross_validate(forestClassifier1, xHivDataset, yHivDataset, cv=kfolds, n_jobs=-1)
print(forestCV1["test_score"].mean())
forestClassifier2 = RandomForestClassifier(n_estimators=500, min_samples_leaf=10)
forestCV2 = cross_validate(forestClassifier2, xHivDataset, yHivDataset, cv=kfolds, n_jobs=-1)
print(forestCV2["test_score"].mean())
forestClassifier3 = RandomForestClassifier(n_estimators=1000, min_samples_leaf=10)
forestCV3 = cross_validate(forestClassifier3, xHivDataset, yHivDataset, cv=kfolds, n_jobs=-1)
print(forestCV3["test_score"].mean())