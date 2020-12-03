import numpy as np
    
def LWR(xConst, trainingX, trainingY, tau):
    xConst = np.r_[1, xConst]
    trainingX = np.c_[np.ones(len(trainingX)),trainingX]
    
    xTranspose = trainingX.T * kernel(xConst, trainingX, tau)
    theta = np.linalg.pinv(xTranspose @ trainingX) @ xTranspose @ trainingY
    
    return xConst @ theta, theta
def kernel(xConst, trainingX, tau):
    return np.exp(np.sum((trainingX - xConst) ** 2, axis=1) / (-2 * tau * tau))

    
    