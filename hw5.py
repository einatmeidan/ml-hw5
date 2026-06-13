import numpy as np
import math
import statistics

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance or an array of discrete instances
    rate: poisson rate parameter (lambda)

    return the log pmf value for instances k given the rate
    """
    log_p = None
    k = np.asarray(k)
    log_factorial = np.zeros(k.shape)
    for i in range(k.size):
        log_factorial.flat[i] = math.log(math.factorial(int(k.flat[i])))
    log_p = k * np.log(rate) - rate - log_factorial
    if log_p.ndim == 0:
        log_p = float(log_p)
    return log_p

def poission_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    mean = np.mean(samples)
    return mean

def poission_confidence_interval(lambda_mle, n, alpha=0.05):
    """


    lambda_mle: an MLE for the rate parameter (lambda) in a Poisson distribution
    n: the number of samples used to estimate lambda_mle
    alpha: the significance level for the confidence interval (typically small value like 0.05)
 
    return: a tuple (lower_bound, upper_bound) representing the confidence interval
    """
    # You may use statistics.NormalDist().inv_cdf to compute the inverse of the normal CDF
    lower_bound = None
    upper_bound = None
    z = statistics.NormalDist().inv_cdf(1 - alpha / 2)
    se = np.sqrt(lambda_mle / n)
    lower_bound = lambda_mle - z * se
    upper_bound = lambda_mle + z * se
    return lower_bound, upper_bound

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represents the log-likelihood value of rates[i]
    """
    likelihoods = None
    likelihoods = np.zeros(len(rates))
    for i in range(len(rates)):
        likelihoods[i] = np.sum(poisson_log_pmf(samples, rates[i]))
    return likelihoods


class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        ###########################################################################
        # TODO: Fill in the joint probability dictionaries below.                 #
        #       Make sure that X, Y, C, X_Y, X_C, and Y_C are all                 #
        #       consistent with X_Y_C                                             # 
        ###########################################################################
        # P(X=x, Y=y, C=c)
        self.X_Y_C = {
            (0, 0, 0): 0.18,
            (0, 0, 1): 0.00,
            (0, 1, 0): 0.12,
            (0, 1, 1): 0.00,
            (1, 0, 0): 0.12,
            (1, 0, 1): 0.00,
            (1, 1, 0): 0.08,
            (1, 1, 1): 0.50,
        } 
        # P(X=x, Y=y)
        self.X_Y = {
            (0, 0): 0.18,
            (0, 1): 0.12,
            (1, 0): 0.12,
            (1, 1): 0.58
        }

        # P(X=x, C=c)
        self.X_C = {
            (0, 0): 0.30,
            (0, 1): 0.00,
            (1, 0): 0.20,
            (1, 1): 0.50
        }

        # P(Y=y, C=c)
        self.Y_C = {
            (0, 0): 0.30,
            (0, 1): 0.00,
            (1, 0): 0.20,
            (1, 1): 0.50
        }

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are dependent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for x in X:
            for y in Y:
                if abs(X_Y[(x, y)] - X[x] * Y[y]) > 1e-10:
                    return True
        return False
    
    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are independent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        for c in C:
            for x in X:
                for y in Y:
                    p_xy_given_c = X_Y_C[(x, y, c)] / C[c]
                    p_x_given_c = X_C[(x, c)] / C[c]
                    p_y_given_c = Y_C[(y, c)] / C[c]
                    if abs(p_xy_given_c - p_x_given_c * p_y_given_c) > 1e-10:
                        return False    
        return True

def normal_pdf(x, mean, std):
    
    """
    Calculate normal density function for a given x, mean and standard deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    p = (1 / np.sqrt(2 * np.pi * std**2)) * np.exp(-((x - mean)**2) / (2 * std**2))
    return p

class NaiveNormalClassDistribution():
    def __init__(self, class_value):
        """
        A class which encapsulates information on the feature-specific
        class conditional distributions for a given class label.
        Each of these distributions is a univariate normal distribution with
        separate parameters (mean and std).
        The distribution parameters are estimated using the fit method.
        
        Input
        - class_value : The class label to calculate the class conditionals for.
        """
        self.class_value = class_value
        self.prior = None
        self.means = None
        self.stds = None

    def fit(self, dataset):
        """
        Fit the class prior and feature-specific normal parameters from data.

        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        """
        class_rows = dataset[dataset[:, -1] == self.class_value]
        features = class_rows[:, :-1]
        self.prior = len(class_rows) / len(dataset)
        self.means = np.mean(features, axis=0)
        self.stds = np.std(features, axis=0)
    
    def get_prior(self):
        """
        Returns the prior probability of the class, as computed from the training data.
        """
        prior = None
        prior = self.prior
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the feature-specific class conditionals fitted to the training data.
        """
        likelihood = None
        likelihood = 1.0
        for i in range(len(x)):
            likelihood *= normal_pdf(x[i], self.means[i], self.stds[i])
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        return joint_prob

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class holds a ClassDistribution object (either NaiveNormal or MultiNormal)
        for each of the two class labels (0 and 1). 
        Using these objects it predicts class labels for input instances using the MAP rule.
    
        Input
            - ccd0 : A ClassDistribution object for class label 0.
            - ccd1 : A ClassDistribution object for class label 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        if self.ccd0.get_instance_joint_prob(x) > self.ccd1.get_instance_joint_prob(x):
            pred = 0
        else:
            pred = 1
        return pred

    
def multi_normal_pdf(x, mean, cov):
    """
    Calculate multivariate normal density function under specified mean vector
    and covariance matrix for a given x.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    x = np.array(x)
    mean = np.array(mean)
    p = len(x)
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    diff = x - mean
    exponent = -0.5 * diff @ inv @ diff
    pdf = (2 * np.pi) ** (-p / 2) * det ** (-0.5) * np.exp(exponent)
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, class_value):
        """
        A class which encapsulates the multivariate normal distribution
        representing the class conditional distribution for a given class label.
        The distribution parameters are estimated using the fit method.
        
        Input
        - class_value : The class label to calculate the parameters for.
        """
        self.class_value = class_value
        self.prior = None
        self.mean = None
        self.cov = None

    def fit(self, dataset):
        """
        Fit the class prior, mean vector, and covariance matrix from data.

        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        The covariance matrix should be estimated using the sample covariance matrix.
        """
        class_rows = dataset[dataset[:, -1] == self.class_value]
        features = class_rows[:, :-1]
        self.prior = len(class_rows) / len(dataset)
        self.mean = np.mean(features, axis=0)
        self.cov = np.cov(features, rowvar=False, bias=True)
        
        
    def get_prior(self):
        """
        Returns the prior probability of the class, as computed from the training data.
        """
        prior = None
        prior = self.prior
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the multivariate class conditionals fitted to the training data.
        """
        likelihood = None
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        return joint_prob



def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given MAP classifier on a given test set.
    
    Input
        - test_set: The test data (Numpy array) on which to compute the accuracy. The class label is the last column
        - map_classifier : A MAPClassifier object that predicts the class label from a feature vector.
        
    Output
        - Accuracy = #Correctly Classified / number of test samples
    """
    acc = None
    correct = 0
    for i in range(len(test_set)):
        if map_classifier.predict(test_set[i, :-1]) == test_set[i, -1]:
            correct += 1
    acc = correct / len(test_set)
    return acc

class DiscreteNBClassDistribution():
    def __init__(self, class_value):
        """
        A class which encapsulates the probabilities for a discrete naive bayes
        class conditional distribution for a given class label.
        The feature-specific class conditional probabilities are estimated using
        the fit method with Laplace smoothing.
        
        Input
        - class_value : The class label to calculate the probabilities for.
        """
        self.class_value = class_value
        self.n_class = None
        self.prior = None
        self.feature_values = None
        self.probabilities = None

    def fit(self, dataset):
        """
        Fit the class prior and feature-specific discrete probabilities from data.

        Input
        - dataset: The training dataset as a 2d numpy array, assuming the class label is the last column
        """
        # find the feature-specific class conditional probabilities with Laplace smoothing
        class_rows = dataset[dataset[:, -1] == self.class_value]
        features = class_rows[:, :-1]
        self.n_class = len(class_rows)
        self.prior = len(class_rows) / len(dataset)
        self.feature_values = []
        self.probabilities = []
        all_features = dataset[:, :-1]
        for i in range(features.shape[1]):
            unique_vals = np.unique(all_features[:, i])
            self.feature_values.append(unique_vals)
            probs = {}
            for val in unique_vals:
                n_j_a_y = np.sum(features[:, i] == val)
                probs[val] = (n_j_a_y + 1) / (self.n_class + len(unique_vals))
            self.probabilities.append(probs)
    
    def get_prior(self):
        """
        Returns the prior probability of the class, as computed from the training data.
        """
        prior = None
        prior = self.prior
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance given the class label according to
        the product of feature-specific discrete class conditionals fitted to the training data.
        """
        likelihood = None
        likelihood = 1.0
        for i in range(len(x)):
            val = x[i]
            if val in self.probabilities[i]:
                likelihood *= self.probabilities[i][val]
            else:
                likelihood *= 1 / (self.n_class + len(self.feature_values[i]))
        return likelihood
    
    def get_instance_joint_prob(self, x):
        """
        Returns the joint probability of the input instance (x) and the class label.
        """
        joint_prob = None
        joint_prob = self.get_prior() * self.get_instance_likelihood(x)
        return joint_prob
