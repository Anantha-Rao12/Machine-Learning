from numpy import sqrt

def mean(x):
    # given x can be an array or list
    # output is the mean value 
    mean_x = sum(x)/len(x)
    return mean_x

def mean_normalise(x):
    # given x can be an array or list
    # output is the same list that is normalised to the mean
    normalised_x = [xi - mean(x) for xi in x]
    return normalised_x

def std_dev(x):
    # given x can be an array or list
    # output is the standard deviation value
    norm_x = mean_normalise(x)
    var_x = sum(list(map(lambda x : x**2,norm_x)))
    std_x = sqrt(var_x/len(x))  # use the square root function provided by numpy
    return std_x

def mean_normalization(x): 
    # given x is a 2D numpy array
    # output is the same list that is normalised to the mean/std 
    data_array = np.array(x)
    mean = np.mean(x,axis=0,keepdims=True)
    std = np.std(x,axis=0,keepdims=True)

    norm_data_array = (x - mean)/std
    return norm_data_array

def r_sqr(y,y_pred):
    num = sum((y-y_pred)**2)
    den = sum((y-mean(y))**2)
    r_sqr = 1 - (num/den)
    return r_sqr[0]

def train_test_split(X,Y,test_split,shuffle=True):
    # given two numpy arrays X (two dimensional) and Y (one dimensional)
    
    try :
    
        dataset = np.hstack((X,Y.reshape(X.shape[0],1)))  #create the dataset by concating X,Y
        shuffled_dataset = np.random.permutation(dataset)  #shuffle the dataset
        
        testset_length = np.int(len(shuffled_dataset)*test_split)  # length of test set
        trainset_length = len(shuffled_dataset) - testset_length   # length of training set
        
        X_train = shuffled_dataset[:trainset_length,:-1]
        Y_train = suffled_dataset[:trainset_length,-1]
        X_test = shuffled_dataset[trainset_length:,:-1]
        Y_test = shuffled_dataset[trainset_length:,-1]
        
        return X_train,Y_train,X_test,Y_test    
        
    except:
        print('Error !') # need to specify the kind and methods to change input
        print('Shape of Input X and Y arrays dont match !')
        
        return None