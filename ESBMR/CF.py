import numpy as np
import multiprocessing_functions
from sklearn.metrics import mean_absolute_error
from functools import partial
import multiprocessing
import time

# Custom modules
import multiprocessing_functions


class CF:
    """
    Memory-based, user-based collaborative filtering.
    Generates recommendations as the average rating of the most similar users.
    It computes the similarities between the users based on absolute deviation metric.
    """

    def __init__(self):
        '''
        Inizializing an object of the class "CF", which is a memory-based, user-based recommender system.
        The recommendations for the n-th user are the average of the recommendations of the N most similar users.
        '''
    
    def fit(self, train, neighborhood=10):
        """
        Fitting the memory-based collaborative filtering.

        Parameters
        ----------
        train : np.array
            Training set
        neighborhood : int
            Number of neighbors of each user.
        """
        self.train = train
        self.neighborhood = neighborhood
        self.deviations = np.zeros(shape=(self.train.shape[0], self.train.shape[0]))

        for i in range(self.train.shape[0]):
            for j in range(self.train.shape[0]):
                self.deviations[i, j] = mean_absolute_error(self.train[i, :], self.train[j, :])

        self.similar_users = {}
        for i in range(self.train.shape[0]):
            self.similar_users[i] = (-self.deviations[i, :]).argsort()[:self.neighborhood]

        self.pred = np.zeros(shape=self.train.shape)
        for i in range(self.train.shape[0]):
            self.pred[i, :] = self.train[self.similar_users[i]].mean(axis=0)

        return self.pred

    
    def fit2(self, train, neighborhood=10):
        """
        Fitting the memory-based collaborative filtering. Multiprocessing-compatible.

        Parameters
        ----------
        train : np.array
            Training set
        neighborhood : int
            Number of neighbors of each user.
        """
        self.train = train
        self.neighborhood = neighborhood
        self.deviations = np.zeros(shape=(self.train.shape[0], self.train.shape[0]))

        for i in range(self.train.shape[0]):
            for j in range(self.train.shape[0]):
                self.deviations[i, j] = mean_absolute_error(self.train[i, :], self.train[j, :])

        self.similar_users = {}
        for i in range(self.train.shape[0]):
            self.similar_users[i] = (-self.deviations[i, :]).argsort()[:self.neighborhood]

        self.pred = np.zeros(shape = train.shape)
        
        prediction = multiprocessing_functions.multi_func_list([i for i in range(self.train.shape[0])],
                                                             self.predict,
                                                             n_procs=multiprocessing.cpu_count())

        return prediction

    def predict(self, i):
        self.pred[i] = self.train[self.similar_users[i]].mean(axis=0)
        return self.pred[i]
