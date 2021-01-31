import numpy as np
from scipy.special import digamma
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve


class hpf_vi():
    def __init__(self, a=0.3, c=0.3, a1=0.3, b1=1, c1=0.3, d1=1, K=10):
        '''
        Initialization of the parameter matrices used in the CAVI algorithm.
        The user can modify the hyperparameters and the dimension of latent attributes and preferences K.

        Parameters:
        ----------
            - a : float
              shape parameter for the Gamma(a, activity_u) prior
              for user preferences.
            - c : float
              shape parameter for the Gamma(c, popularity_i) prior
              for item attributes.
            - a1, b1 : floats
              parameters of the Gamma(a1, a1/b1) prior for user activity.
            - c_1, d_1 : floats
              parameters of the Gamma(c1, c1/d1) prior for item popularity.
            - K : int
              dimensionality of latent attributes and preferences.
        '''
        self.a, self.c, self.a1, self.b1, self.c1, self.d1, self.K = a, c, a1, b1, c1, d1, K

    def fit(self, train, iterations, tol=0.5, valid=None):
        '''
        Fit the Hierarchical Poisson Factorization model via Coordinate Ascent Variational Algorithm (CAVI)
        and generates the corresponding observation matrix based on the variational parameters.
        The algorithm stops either when the log-likelihood difference is less than the tolerance or after the number of iterations specified.

        Parameters:
        ----------
            - train : numpy.array
              UxI array with U = users and I = items.
            - iterations: int
              number of desired training epochs.
            - tol: float
              tolerance stopping criterion.
        '''
        self.train = train
        self.valid = valid

        # Dataset dimensions
        self.U, self.I = train.shape

        self.initialize()
        self.mse_train = np.zeros(iterations)
        self.ll = np.zeros(iterations)

        self.its = 0

        # Building user preferences and item attributes
        self.theta = self.gamma_shp/self.gamma_rte
        self.beta = self.lambda_shp/self.lambda_rte
        old_ll = self.log_likelihood(self.train, self.beta, self.theta)
        stop = False

        self.mse_valid = np.zeros(iterations)

        def MSE(pred, values):
            prediction, values = pred[values.nonzero()].flatten(), values[values.nonzero()].flatten()
            return mean_squared_error(prediction, values)

        tic = time.time()

        while not stop and self.its < iterations:
            for u, i in zip(train.nonzero()[0], train.nonzero()[1]):
                self.phi[u, i] = [np.exp(digamma(self.gamma_shp[u, k]) - np.log(self.gamma_rte[u, k])
                                         + digamma(self.lambda_shp[i, k]) - np.log(self.lambda_rte[i, k])) for k in range(self.K)]
                self.phi[u, i] = self.phi[u, i] / np.sum(self.phi[u, i])

            for u in range(self.U):
                self.gamma_shp[u] = [self.a + np.sum(train[u]*self.phi[u, :, k]) for k in range(self.K)]
                self.gamma_rte[u] = [self.k_shp/self.k_rte[u] +
                                     np.sum(self.lambda_shp[:, k]/self.lambda_rte[:, k]) for k in range(self.K)]
                self.k_rte[u] = self.a1/self.b1 + np.sum(self.gamma_shp[u]/self.gamma_rte[u])

            for i in range(self.I):
                self.lambda_shp[i] = [self.c + np.sum(train[:, i]*self.phi[:, i, k]) for k in range(self.K)]
                self.lambda_rte[i] = [self.tau_shp/self.tau_rte[i] +
                                      np.sum(self.gamma_shp[:, k]/self.gamma_rte[:, k]) for k in range(self.K)]
                self.tau_rte[i] = self.c1/self.d1 + np.sum(self.lambda_shp[i]/self.lambda_rte[i])

            # Building user preferences and item attributes
            self.theta = self.gamma_shp/self.gamma_rte
            self.beta = self.lambda_shp/self.lambda_rte

            # Evaluating the Loglikelihood
            self.ll[self.its] = self.log_likelihood(self.train, self.beta, self.theta)

            # Generating observations y
            self.predicted = np.dot(self.theta, self.beta.T)

            # In-sample MSE
            self.mse_train[self.its] = MSE(self.predicted, self.train)

            # Out-of-sample MSE
            if valid is not None:
                self.mse_valid[self.its] = MSE(self.predicted, self.valid)

            if abs(self.ll[self.its] - old_ll) > tol:
                old_ll = self.ll[self.its]
                self.its += 1
                print(f"Iteration {self.its} completed. Log-likelihood: {self.ll[self.its-1]}")
            else:
                stop = True

        else:
            toc1 = time.time()
            time1 = np.round(toc1 - tic, 3)
            if self.its < iterations:
                print(
                    f"Converged in {time1} seconds after {self.its} iterations. Log-likelihood: {self.ll[self.its-1]}.")
            else:
                print(f"Stopped after {time1} seconds, {self.its} iterations. Log-likelihood: {self.ll[self.its-1]}.")

    def recommend(self, test, t=0.3):
        '''
        Using the fitted algorithm to make recommendations to users of our dataset.

        Parameters:
        ----------
            - t : float
              Delta-threshold for activating recommendations
        '''
        for u in range(self.U):
            recomm = []
            for i in range(self.I):
                if self.predicted[u, i] > t:
                    recomm.append(i)
            if [i > 0 for i in recomm]:
                print(f"User {u} may also like these items: {recomm}")

    def roc_curve(self, test):
        valid_dim = 0.2
        prediction, values = self.predicted[test.nonzero()].flatten(), test[test.nonzero()].flatten()
        addit_ind_u = np.random.choice(np.where(self.train == 0)[0], round(
            self.train.shape[0]*valid_dim/5), replace=False)
        addit_ind_i = np.random.choice(np.where(self.train == 0)[1], round(
            self.train.shape[0]*valid_dim/5), replace=False)
        for ind_u in addit_ind_u:
            for ind_i in addit_ind_i:
                prediction = np.append(prediction, self.predicted[ind_u, ind_i])
                values = np.append(values, 0)
        return roc_curve(values, prediction)

    def initialize(self):
        self.gamma_shp = np.random.uniform(0, 1, size=(self.U, self.K)) + self.a
        self.gamma_rte = np.repeat(self.a/self.b1, self.K) + np.random.uniform(0, 1, size=(self.U, self.K))
        self.k_rte = self.a1/self.b1 + np.random.uniform(0, 1, self.U)
        self.k_shp = self.a1 + self.K*self.a

        self.lambda_shp = np.random.uniform(0, 1, size=(self.I, self.K)) + self.c
        self.lambda_rte = np.repeat(self.c/self.d1, self.K) + np.random.uniform(0, 1, size=(self.I, self.K))
        self.tau_rte = self.c1/self.d1 + np.random.uniform(0, 1, self.I)
        self.tau_shp = self.c1 + self.K*self.c
        # Note that the parameters tau_shp and k_shp are not updated in the algorithm, so they are declared here.

        self.phi = np.zeros(shape=[self.U, self.I, self.K])

    def log_likelihood(self, train, beta, theta):
        '''
        Evaluating the log-likelihood
        '''
        self.train = train
        self.beta = beta
        self.theta = theta

        self.sumlog = 0
        self.prod = 1
        count_array = 0
        for u, i in zip(self.train.nonzero()[0], self.train.nonzero()[1]):
            self.dot = float(np.dot(theta[u], beta[i].T))
            self.dot_y = float(self.dot**train[u, i])
            self.dot_y_fact = float(self.dot/np.math.factorial(train[u, i]))
            self.logdot_y_fact = np.log(self.dot_y_fact)
            self.sumlog += self.logdot_y_fact
            count_array += 1

        for u, i in zip(range(train.shape[0]), range(train.shape[1])):
            self.exp = float(np.exp(-np.dot(theta[u], beta[i].T)))
            self.prod = float(self.prod * self.exp)
        self.logprod = np.log(self.prod)

        return self.sumlog+self.logprod
