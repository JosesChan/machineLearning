from array import array
from doctest import OutputChecker
from tkinter import Y
import numpy
import numpy.linalg as linalg
import matplotlib
import pandas

polyTest = pandas.read_csv('Task1 - dataset - pol_regression.csv')


# Section 1.1: Implementation of Polynomial Regression (10%).
# You are asked to implement the polynomial regression algorithm. To do so, you are required to
# use the following function:

def pol_regression(features_train, y_train, degree):
    coefficients = numpy.polyfit(features_train, y_train, degree)
    polynomialcoefficient = numpy.poly1d(coefficients)    
    return polynomialcoefficient

# Section 1.2: Regress a polynomial of the following degrees: 0, 1, 2, 3, 6, 10 (10%).
# After implementing the pol_regression function, use this function to regress the data set given in
# the .csv file of task 1 as part of this assignment. Regress a polynomial of the following degrees: 0,
# 1, 2, 3, 6, 10. Plot the resulting polynomial in the range of [-5, 5] for the inputs x. In addition, plot
# the training points. Interpret and elaborate on your results. Which polynomial function would you
# choose?

x_train = polyTest['x']
y_train = polyTest['y']

degreesList = [0, 1, 2, 3, 6, 10]


def getPolynomialDataMatrix(x, degree):
    X = numpy.ones(x.shape)
    for i in range(1,degree + 1):
        X = numpy.column_stack((X, x ** i))
    return X

def getWeightsForPolynomialFit(x,y,degree):
    X = getPolynomialDataMatrix(x, degree)

    XX = X.transpose().dot(X)
    w = numpy.linalg.solve(XX, X.transpose().dot(y))
    #w = np.linalg.inv(XX).dot(X.transpose().dot(y))

    return w

plt.figure()
plt.plot(x_train,y_train, 'k')
plt.plot(x_train,y_train, 'bo')

w0 = getWeightsForPolynomialFit(x_train,y_train,0)
Xtest0 = getPolynomialDataMatrix(polyTest['x'], 0)
ytest0 = Xtest0.dot(w0)
plt.plot(x_train, ytest0, 'b')

w1 = getWeightsForPolynomialFit(x_train,y_train,1)
Xtest1 = getPolynomialDataMatrix(polyTest['x'], 1)
ytest1 = Xtest1.dot(w1)
plt.plot(x_train, ytest1, 'g')

w2 = getWeightsForPolynomialFit(x_train,y_train,2)
Xtest2 = getPolynomialDataMatrix(polyTest['x'], 2)
ytest2 = Xtest2.dot(w2)
plt.plot(x_train, ytest2, 'y')

w3 = getWeightsForPolynomialFit(x_train,y_train,3)
Xtest3 = getPolynomialDataMatrix(polyTest['x'], 3)
ytest3 = Xtest3.dot(w3)
plt.plot(x_train, ytest3, 'm')

w6 = getWeightsForPolynomialFit(x_train,y_train,6)
Xtest6 = getPolynomialDataMatrix(polyTest['x'], 6)
ytest6 = Xtest6.dot(w6)
plt.plot(x_train, ytest6, 'r')

w10 = getWeightsForPolynomialFit(x_train,y_train,10)
Xtest10 = getPolynomialDataMatrix(polyTest['x'], 10)
ytest10 = Xtest10.dot(w10)
plt.plot(x_train, ytest10, 'c')
    
plt.ylim((-5, 5))
plt.legend(('training points', 'ground truth', '$x^{0}$', '$x^{1}$', '$x^{2}$', '$x^{3}$', '$x^{10}$'), loc = 'lower right')

plt.savefig('polynomial1.png')


## errors on test dataset Bashir
error0 = polyTest['y']-ytest0
SSE0 = error0.dot(error0)

error1 = polyTest['y']-ytest1
SSE1 = error1.dot(error1)

error2 = polyTest['y']-ytest2
SSE2 = error2.dot(error2)

error3 = polyTest['y']-ytest3
SSE3 = error3.dot(error3)

error6 = polyTest['y']-ytest6
SSE6 = error6.dot(error6)

error10 = polyTest['y']-ytest10
SSE10 = error10.dot(error10)

SSE0, SSE1, SSE2, SSE3, SSE6, SSE10


plt.scatter(features_train, y_train)

plt.clf()
plt.plot(x_train, y_train, 'bo')
plt.plot(polyTest['x'],polyTest['y'], 'g')
plt.legend(('training points', 'ground truth'))
#plt.hold(True)
plt.savefig('trainingdata.png')
plt.show()



# There is no need to split the data here. Just simply treat the whole data as training data. The plots for
# each degree can be either in the same graph (preferable) or in different ones.


# Section 1.3: Evaluation (10%).
# Using the returned results of the pol_regression function from the previous section, you need now
# to evaluate the performance of your algorithm. To do so, you first need to implement the following
# function:

def eval_pol_regression(inputX,outputY):
    #calculate r-squared
    yhat = polynomialCoeffs(features_train)
    ybar = numpy.sum(y_train)/len(y_train)
    sumSquareRegression = numpy.sum((yhat-ybar)**2)
    sumSquareTotal = numpy.sum((y_train - ybar)**2)
    polyResults['r_squared'] = sumSquareRegression / sumSquareTotal























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


# = train_test_split()

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


# Section 2.2: clustering the dog breed data (15%).
# After implementing the kmeans() function, use it to cluster the dog breed data. You need to run
# your k-means algorithm with all the four features as input. You should not slice the data to have
# two features for calculating Euclidean distance in k-means. Having said that, you need to cluster
# the data based on the four features instead of a portion of them. Use two different values for the
# parameter k: 2 and 3.
# Draw the following three plots for the K-Means clustering results:
# a) The first plot is a scatter plot, where x axis is the “height” feature and y axis is the “tail length”
# feature; Use different colours for the different cluster data points.
# b) The second plot is also a scatter plot, where x axis is the “height” feature and y axis is the “leg
# length” feature; As before use different colours to depict the data points from different
# clusters.
# c) The third plot is a line plot, where x axis is the ‘iteration step’ and y axis is the ‘objective
# function value’. The plot should show a decreasing trend! You must plot the objective
# function (error function) at each iteration when you train the model.

# Please generate the above three requested plots for k = 2 and do the same for k =3. The plotting
# tasks are just to display your clustering results in a 2-D space (height-length) for one k value. Since
# there are two k values, you should run your k-means ONLY TWICE, no matter how many plots you
# need to produce. Anyway, you will need to produce three plots for each k value, and they should
# not suggest you to re-run your algorithm for each plot.
# Note: You need to implement the simple K-Means clustering following the above steps by yourself,
# group work is not allowed! You are allowed to use numpy, pandas and matplotlib libraries for
# processing and plotting data. However, other data science and machine learning libraries such as
# scikit-learn or other built-in libraries that have implemented the k-means algorithm are prohibited
# in this coursework. 



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