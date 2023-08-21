import numpy as np
import random
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    label_sum = np.array(np.sum(y, axis=0))
    label_sum /= np.sum(label_sum)
    label_log = np.log(label_sum +  EPS)
    result = -label_sum.dot(label_log)
 #   print(f"result = {result}")
    return result
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """
    label_sum = np.array(np.sum(y, axis=0))
    label_sum /= np.sum(label_sum)
    result = 1 - label_sum.dot(label_sum)
    return result
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    n_objects = y.shape[0]
    y_mean = np.mean(y,axis=0)
    y-=y_mean
    y = y.T.dot(y)
    y/=n_objects
    return y

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """
    
    n_objects = y.shape[0]
    y_mean = np.median(y, axis=0)
    y_l = y - y_mean
    y_l = np.sum(np.abs(y_l))
    y_l /=n_objects
    return y_l
    

def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index = None, threshold = None, proba = None, left_child = None, right_child = None):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = left_child
        self.right_child = right_child
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):
        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """
        X_left = []
        y_left = []
        X_right = []
        y_right = []
        for i in range(len(X_subset)):
            if X_subset[i][feature_index] < threshold:
                X_left.append(X_subset[i])
                y_left.append(y_subset[i])
            else:
                X_right.append(X_subset[i])
                y_right.append(y_subset[i])
        return (np.array(X_left), np.array(y_left)), (np.array(X_right), np.array(y_right))
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """
        y_left = []
        y_right = []
        for i in range(len(X_subset)):
            if X_subset[i][feature_index] < threshold:
                y_left.append(y_subset[i])
            else:
                y_right.append(y_subset[i])
       
        return (np.array(y_left), np.array(y_right))

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        feature_index = None
        threshold = None
        val_h = self.all_criterions[self.criterion_name][0](y_subset)
        score = 0
        subset_sz = len(y_subset)
        for feature_ind in range(X_subset.shape[1]):
            vals = np.unique(X_subset[:,feature_ind])
            for val in vals:
                    left, right = self.make_split_only_y(feature_ind, val ,X_subset, y_subset)
                    left_sz, right_sz = len(left), len(right)
                    res = -1
                    if left_sz < self.min_samples_split or right_sz < self.min_samples_split:
                        res = -np.inf
                    else:
                        left_h = self.all_criterions[self.criterion_name][0](left)
                        right_h = self.all_criterions[self.criterion_name][0](right)
                        res = val_h - left_sz / subset_sz * left_h - right_sz / subset_sz * right_h
                    if res > score :
                        score = res
                        threshold = val
                        feature_index = feature_ind
        return (feature_index, threshold)
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """
       # print(f" X, delpth = {X_subset},{self.depth}")
        n_samples = X_subset.shape[0]
        n_features = 0
        if X_subset.shape[0] > 0:
            n_features = X_subset.shape[1]
        n_class = np.count_nonzero(np.sum(y_subset, axis = 0))
        if (self.classification and n_class ==1) or self.depth >= self.max_depth  or n_samples < self.min_samples_split:
             if self.classification:
                proba = np.array( np.sum(y_subset,axis=0) / y_subset.shape[0])
             else:
                proba = np.array(np.mean(y_subset, axis=0))
        #     print(f"proba is {proba}")
             return Node(proba=proba)

        feature_index, threshold =  self.choose_best_split(X_subset, y_subset)
    #   print(f"best is {feature_index},{threshold}")
        if feature_index == None:
            if self.classification:
                proba = np.array( np.sum(y_subset,axis=0) / y_subset.shape[0])
            else:
                proba = np.array(np.mean(y_subset, axis=0))
    #        print(f"proba is {proba}")
            return Node(proba=proba)

        left, right = self.make_split(feature_index, threshold, X_subset, y_subset)
        self.depth+=1
        left_child = self.make_tree(left[0], left[1])
        right_child = self.make_tree(right[0], right[1])
        self.depth-=1
        return Node(feature_index=feature_index, threshold=threshold, left_child=left_child, right_child=right_child)
        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)
        self.root = self.make_tree(X, y)

    def go_down(self, v, x):
        if v.proba is not None:
            return v.proba

        if x[v.feature_index] < v.value:
            return self.go_down(v.left_child, x)
        return self.go_down(v.right_child, x)


    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """        
        y_predicted = []
        for row in X:
            probs = self.go_down(self.root, row)
           # print(f"probs = {probs}")
            if self.classification:
                probs = np.array([probs.argmax()])
            y_predicted.append(probs[0])
        return np.array(y_predicted).T
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        y_predicted_probs = []
        for row in X:
            y_predicted_probs.append(self.go_down(self.root, row))
        
        return np.array(y_predicted_probs)
