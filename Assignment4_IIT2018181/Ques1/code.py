import numpy as np

hyperParameter = 100000

def cost(trainingX,trainingY,theta):
    z = np.dot(trainingX,theta)
    hypo = 1.0/(1.0 + np.exp(-z))
    err = (hypo-trainingY)**2
    cost = np.mean(err)
    return cost

def batch_gradient_descent(trainingX,trainingY,theta,numIters,learningRate):
    m = trainingX.shape[0]
    historyList = []
    for i in range(numIters):
        z = np.dot(trainingX,theta)
        hypo = 1.0/(1.0 + np.exp(-z))
        gradFactor = (1/m)*(hypo-trainingY).dot(trainingX)
        theta = theta - learningRate*gradFactor
        
        historyList.append(cost(trainingX,trainingY,theta))
        
    return historyList,theta

def stochastic_gradient_descent(trainingX, trainingY, theta, numIters, learningRate):
    m = trainingX.shape[0]
    errList = []
    for i in range(numIters):
        for j in range(m):
            idx = np.random.randint(m)
            X = trainingX[idx:idx+1, :]
            Y = trainingY[idx:idx+1]

            z = np.dot(X,theta)
            hypo = 1.0/(1.0 + np.exp(-z))
            gradFactor = (1/m)*(hypo-Y).dot(X)
            theta = theta - learningRate*gradFactor

        errList.append(cost(trainingX,trainingY, theta))

    return errList, theta


def mini_batch_gradient_descent(trainingX,trainingY,theta,numIters,learningRate):
    m=trainingX.shape[0]
    batchSize = 10
    errList = []
    for i in range(0,numIters):
        for j in range(0,(int)(m/10)):
            left = 10*j
            right = 10*j + 10
            if right>m:
                right = m
            X = trainingX[left:right]
            Y = trainingY[left:right]
            z = np.dot(X,theta)
            hypo = 1.0/(1.0 + np.exp(-z))
            err = hypo - Y
            XTranspose = np.transpose(X)
            gradFactor = np.dot(XTranspose,err)/m
            theta = theta - learningRate*gradFactor
        errList.append(cost(trainingX,trainingY,theta))
    return errList,theta

# WITH REGULARIZATION

def batch_gradient_descent_reg(trainingX,trainingY,theta,numIters,learningRate):
    m = trainingX.shape[0]
    historyList = []
    for i in range(numIters):
        z = np.dot(trainingX,theta)
        hypo = 1.0/(1.0 + np.exp(-z))
        gradFactor = (1/m)*(hypo-trainingY).dot(trainingX)
        theta = (1-(learningRate*hyperParameter)/m)*theta - learningRate*gradFactor
        
        historyList.append(cost(trainingX,trainingY,theta))
        
    return historyList,theta

def stochastic_gradient_descent_reg(trainingX, trainingY, theta, numIters, learningRate):
    m = trainingX.shape[0]
    errList = []
    for i in range(numIters):
        for j in range(m):
            idx = np.random.randint(m)
            X = trainingX[idx:idx+1, :]
            Y = trainingY[idx:idx+1]

            z = np.dot(X,theta)
            hypo = 1.0/(1.0 + np.exp(-z))
            gradFactor = (1/m)*(hypo-Y).dot(X)
            theta = (1-(learningRate*hyperParameter)/m)*theta - learningRate*gradFactor

        errList.append(cost(trainingX,trainingY, theta))

    return errList, theta


def mini_batch_gradient_descent_reg(trainingX,trainingY,theta,numIters,learningRate):
    m=trainingX.shape[0]
    batchSize = 10
    errList = []
    for i in range(0,numIters):
        for j in range(0,(int)(m/10)):
            left = 10*j
            right = 10*j + 10
            if right>m:
                right = m
            X = trainingX[left:right]
            Y = trainingY[left:right]
            z = np.dot(X,theta)
            hypo = 1.0/(1.0 + np.exp(-z))
            err = hypo - Y
            XTranspose = np.transpose(X)
            gradFactor = np.dot(XTranspose,err)/m
            theta = (1-(learningRate*hyperParameter)/m)*theta - learningRate*gradFactor
        errList.append(cost(trainingX,trainingY,theta))
    return errList,theta

