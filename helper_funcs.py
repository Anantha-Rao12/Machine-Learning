import numpy as np
from numpy import sqrt


def mean(x: list) -> float:
    """given x can be an array or list output is the mean value"""
    mean_x = sum(x)/len(x)
    return mean_x


def mean_normalise(x: list) -> list:
    """Returns the mean normalized list for a given input list"""
    normalised_x = [xi - mean(x) for xi in x]
    return normalised_x


def std_dev(x: list) -> float:
    """Returns the standard deviation value for an input list"""
    norm_x = mean_normalise(x)
    var_x = sum(list(map(lambda x: x**2, norm_x)))
    return sqrt(var_x/len(x))


def mean_normalization(x: list) -> np.ndarray: 
    """Returns the mean normalized array divided by the std.dev of the entries"""
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    return (x - mean)/std


def r_sqr(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns the R2 or the coefficient of determination 
    for given Y and predicted Y arrays"""
    num = sum((y-y_pred)**2)
    den = sum((y-mean(y))**2)
    r_sqr = 1 - (num/den)
    return r_sqr[0]


def train_test_split(X: np.ndarray, Y: np.ndarray, test_split,shuffle=True):
    """Given the Independent (X) and Dependent (Y) variables, 
    perform train-test split by random shuffling(default)"""
    try:
        dataset = np.hstack((X, Y.reshape(X.shape[0], 1)))  #create the dataset by concating X,Y
        shuffled_dataset = np.random.permutation(dataset)  #shuffle the dataset
        testset_length = np.int(len(shuffled_dataset)*test_split)  # length of test set
        trainset_length = len(shuffled_dataset) - testset_length   # length of training set
        X_train = shuffled_dataset[:trainset_length, :-1]
        Y_train = shuffled_dataset[:trainset_length, -1]
        X_test = shuffled_dataset[trainset_length:, :-1]
        Y_test = shuffled_dataset[trainset_length:, -1]
        
        return X_train, Y_train, X_test, Y_test   
        
    except:
        print('Error !') # need to specify the kind and methods to change input
        print('Shape of Input X and Y arrays dont match !')
        
        return None
