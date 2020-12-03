import numpy as np

hyperParameter = 100000 #Taking lambda as 1 lakh

def cost(X, Y, theta):
    hypo = np.dot(X, theta)
    err = (hypo - Y) ** 2
    cost = np.mean(err)
    return cost

#WITHOUT REGULARIZATION

def batch_gradient_descent(trainingX, trainingY, theta, numIters, learningRate):
    m = trainingX.shape[0]
    historyList = []
    for i in range(numIters):
        gradFactor = (1 /m) * (np.dot(trainingX, theta) - trainingY).dot(trainingX)
        theta = theta - learningRate * gradFactor

        historyList.append(cost(trainingX, trainingY, theta))

    return historyList , theta


def stochastic_gradient_descent(trainingX, trainingY, theta, numIters, learningRate):
    m = trainingX.shape[0]
    errList = []
    for i in range(numIters):
        for j in range(m):
            idx = np.random.randint(m)
            X = trainingX[idx:idx+1, :]
            Y = trainingY[idx:idx+1]

            gradFactor = (1/m)*(np.dot(X, theta) - Y).dot(X)
            theta = theta - learningRate * gradFactor

        errList.append(cost(trainingX,trainingY, theta))

    return errList, theta


def mini_batch_gradient_descent(trainingX,trainingY,theta,numIters,learningRate):
    m=trainingX.shape[0]
    batchSize = 45
    errList = []
    for i in range(0,numIters):
        for j in range(0,(int)(m/45)):
            left = 45*j
            right = 45*j + 45
            if right>m:
                right = m
            X = trainingX[left:right]
            Y = trainingY[left:right]
            hypo = np.dot(X,theta)
            err = hypo - Y
            XTranspose = np.transpose(X)
            gradFactor = np.dot(XTranspose,err)/m
            theta = theta - learningRate*gradFactor
        errList.append(cost(trainingX,trainingY,theta))
    return errList,theta

#WITH REGULARIZATION

def batch_gradient_descent_reg(trainingX, trainingY, theta, numIters, learningRate):
    m = trainingX.shape[0]
    historyList = []
    for i in range(numIters):
        gradFactor = (1 /m) * (np.dot(trainingX, theta) - trainingY).dot(trainingX)
        theta = (1-(learningRate*hyperParameter)/m)*theta - learningRate * gradFactor #SEE THE CHANGE IN THIS LINE

        historyList.append(cost(trainingX, trainingY, theta))

    return historyList , theta


def stochastic_gradient_descent_reg(trainingX, trainingY, theta, numIters, learningRate):
    m = trainingX.shape[0]
    errList = []
    for i in range(numIters):
        for j in range(m):
            idx = np.random.randint(m)
            X = trainingX[idx:idx+1, :]
            Y = trainingY[idx:idx+1]

            gradFactor = (1/m)*(np.dot(X, theta) - Y).dot(X)
            theta =(1-(learningRate*hyperParameter)/m)*theta - learningRate * gradFactor #SEE THE CHANGE IN THIS LINE

        errList.append(cost(trainingX,trainingY, theta))

    return errList, theta


def mini_batch_gradient_descent_reg(trainingX,trainingY,theta,numIters,learningRate):
    m=trainingX.shape[0]
    batchSize = 45
    errList = []
    for i in range(0,numIters):
        for j in range(0,(int)(m/45)):
            left = 45*j
            right = 45*j + 45
            if right>m:
                right = m
            X = trainingX[left:right]
            Y = trainingY[left:right]
            hypo = np.dot(X,theta)
            err = hypo - Y
            XTranspose = np.transpose(X)
            gradFactor = np.dot(XTranspose,err)/m
            theta = (1-(learningRate*hyperParameter)/m)*theta - learningRate*gradFactor #SEE THE CHANGE IN THIS LINE
        errList.append(cost(trainingX,trainingY,theta))
    return errList,theta
