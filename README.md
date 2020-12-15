# Introduction-to-Machine-Learning-Assignments

## Assignment 1
Parse website makemytrip.com or goibibo.com to collect all the
listed hotel details by using beautiful soup library.

## Assignment 2 Q5

Download the COVID -19 data of India for the month of May, 2020 and design a predictor for the number of deaths on a particular day. Hence, predict the number of deaths on  April 20, 2020 and June 10th , 2020. Verify your prediction with the actual number of deaths and hence calculate the accuracy of prediction.

## Assignment 2 Q6

Download the housing price data set of Windsor City of Canada ( provided on my website link). Design a housing price predictor taking only floor area (plot size), number of bedrooms, and number of bathrooms into considerations. Out of total 546 data , you may take 70% for designing the predictor and 30% for validating the design. The predictor design should be done using the following methods:

a) Normal equations  with  and without regularization and compare their performances in terms of % error in prediction. ( only allowed to use NumPy library of Python.no other functions/libraries are allowed ).

b) Design Predictor using Batch Gradient Descent Algorithm, Stochastic Gradient Algorithm and mini batch Gradient Descent algorithms (determining minibatch size is your choice- here it could be 10, 20, 30 etc.) with and without feature scaling and compare their performances in terms of % error in prediction.(only allowed to use NumPy library of Python, no other functions/libraries are allowed).

c)Design Predictor using Batch Gradient Descent Algorithm, Stochastic Gradient Algorithm and mini batch Gradient Descent algorithms (determining minibatch size is your choice- here it could be 10, 20, 30 etc.) with and without regularization and compare their performances in terms of % error in prediction.(only allowed to use the NumPy library of Python, no other functions/libraries are allowed).

d)Implement the LWR algorithm on the Housing Price data set with different tau values.	Find out the tau value which will provide the best fit predictor and hence compare its results with a) , b) and c) above.	

## Assignment 3

Implementing Box-Muller Transformation algorithm in Python (use NumPy library only)

## Assignment 4

1. Our task is to build a Logistic Regression based classification model that estimates an applicant's probability of getting admission to an institution based on the scores from those two examinations whose data have been provided here (you may use 70% data for training and 30% for testing).

a) Design a Predictor with two basic features which are given using Batch Gradient Descent Algorithm, Stochastic Gradient Algorithm and mini batch Gradient Descent algorithms (determining minibatch size is your choice- here it could be 10, 20, 30 etc.) with and without feature scaling and compare their performances in terms of % error in prediction.(only allowed to use NumPy library of Python, no other functions/libraries are allowed).  

b) Inject more features from the data set  in the model ( at least 6-9) and repeat (a)  

c) Add regularization term and repeat (b). Submit comparative analyses of your results. 

2. After gaining experience of solving problem No 1) Design a classifier using logistic regression on Cleveland Medical data set for heart disease diagnosis. The processed dataset with some 13 features have been given with a label that a patient has a heart disease (1) or not (0).

## Assignment 5 on GDA(Gaussian Discriminant Analysis)

1. Using raw data set as given, create three more features, and from there develop a GDA
model. Thereafter, utilize the same to predict whether a Microchip component will be
accepted or rejected. May use 70% data for training and 30 % data for testing.

2. Using the same data set and features, and same 70% of the data for training and 30%
for testing, now use Box-Muller transformation to create a new data set having Gaussian
distribution within the range of the given data set and create a new Gaussian Discriminant
Analysis (GDA) model. Thereafter, utilize the model to predict where a component will be
accepted or rejected using the testing data.

3. Compare the performance of GDA in both the above cases and write a comparative analysis
report with results.

## Assignment 6

Design a Naïve Bayes classifier for filtering Spam and Ham (Normal) messages. Make a comparative
study on the performance of all the three models of Naïve Bayes classifier.

## Assignment 7 SVM

Use the dataset of heart disease provided on my assignment folder of the course with the
following pre-processing and instructions:

1. Use only two features for simplicity- age ( data in column #1) and trestbps (on
admission to the hospital, data in column #4, i.e resting blood pressure in
mm/Hg)

2. Modify the last column (# 14) from 1 –heart disease & 0 –no heart disease to
Y(i)= {1 and -1}.

3. Apply feature scaling methods to the data of Col# 1 and Col# 4.

4. Use 70% data for training and 30% for testing.

## Assignment 7 ANN

Implement Perceptron training algorithms for AND, OR, NAND and NOR gates.How will you verify your trained algorithms? Justify your solution.

## Assignment 8 ANN

Using two inputs and one output X-NOR data, train a Neural Network using Back Propagation Algorithm.Also explain how you will test the network.

## Assignment 9 ANN

A Bidirectional Associative Memory(BAM)  is required to store the following  M =4 pairs of patterns:

Set A:  X1 =[1 1 1 1 1 1 ]T,  X2 =[-1 -1 -1 -1 -1 -1 ]T, X3 =[1 -1 -1 1 1 1 ]T, X4 =[1 1 -1 -1 -1 -1 ]T

Set B:  Y1=[1 1 1]T, Y2=[-1 -1 -1]T, Y3=[-1 1 1]T, Y4=[1 -1 1]T

Using BAM algorithm, train a W matrix for BAM which can retrieve all the above mentioned 4 pairs. 

Hence test the level of weight corrections of the BAM with examples.

## Assignment 10 Kohonen

Consider a Kohonen network with 100 neurons arranged in the form of a two-dimensional lattice with 10 rows and 10 columns . 

The network is required to classify two-dimensional input vectors such that each neuron in the network should respond only to the input vectors occurring in its region. 

Train the network with 1500 two-dimensional input vectors generated randomly in a square region in the interval between -1 and +1. Select initial synaptic weights randomly in the same interval  (-1 and +1  ) and take the learning rate parameter α is equal to 0.1.

Test the performance of the self organizing neurons using the following

Input vectors:

X1=[0.1  0.8]T,  X2=[0.5  -0.2]T, X3=[-0.8  -0.9]T, X4=[-0.0.6  0.9]T.


## Decision Tree Assignment

### Problem 1

Consider two features, age and heart disease to  create a decision tree with gini impurity.

### Problem 2

Consider two features, slope and heart disease to create a decision tree with Information gain.

## Assignment on PCA

Using the face dataset:
https://drive.google.com/drive/u/2/folders/1XGdUi0w_FcHcnlQt9mU5y_PNLyFhCW9V
design a face recognition system using Python.

### Use of following libraries are allowed

1. Numpy, Scipy for matrix multiplication , finding SVD or Eigen vector etc.
2. Open CV- Python library for inputting/reading images etc.

### For Step by step directions, refer:

https://docs.google.com/a/iiita.ac.in/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxnY25hbmRpfGd4OjQzYjVhZTAwYTdmYjBmNjc

Take 60% data as training set and 40 % data as test set, evaluate your classifier on the following
Factors:

1. Change the value of k and then, see how it changes the classification accuracy. Plot
a graph between accuracy and k value to show the comparative study.
2. Add imposters (who do not belong to the training set) into the test set and then
recognize it as the not enrolled person.






