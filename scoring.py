import numpy as np
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score

# Custom modules
import data_processing


def precision_at_z(test, y_pred, threshold=0.1):
    """
    Computes precision at z, as illustrated in the thesis.

    Parameters
    ----------
        test : np.array
            Test dataset
        y_pred : np.array
            Predicted dataset
        threshold : float, optional
            Threshold value for the classification. Defaults to 0.1.

    Returns
    -------
        precision : np.array
            Precision at z
    """
    self.block_recommend(range(self.Y.shape[0]), 1)  # call to function
    precision = 0
    recommended_and_relevant = np.zeros(shape=self.Y.shape[0])
    for i in range(self.Y.shape[0]):
        for item in range(self.recommendations[i][0].shape[0]):
            index_item = self.recommendations[i][0][item][0]
            if test[i, index_item] == 1 and y_pred[i, index_item] > threshold:
                recommended_and_relevant[i] += 1
        precision += recommended_and_relevant[i] / self.recommendations[i][0].shape[0]
    precision = precision / self.Y.shape[0]
    return precision


def recall_at_z(test, y_pred, threshold=0.1):
    """
    Computes recall at z, as illustrated in the thesis.

    Parameters
    ----------
        test : np.array
            Test dataset
        y_pred : np.array
            Predicted dataset
        threshold : float, optional
            Threshold value for the classification. Defaults to 0.1.

    Returns
    -------
        recall : np.array
            Recall at z
    """
    self.block_recommend(range(self.Y.shape[0]), 1)  # call to function
    recall = 0
    recommended_and_relevant = np.zeros(shape=self.Y.shape[0])
    no_users_with_rel_items = 0
    for i in range(self.Y.shape[0]):
        for item in range(self.recommendations[i][0].shape[0]):
            index_item = self.recommendations[i][0][item][0]
            relevant_items = 0
            if test[i, index_item] == 1 and y_pred[i, index_item] > threshold:
                recommended_and_relevant[i] += 1
        relevant_items = test.sum(axis=1)[i]
        if relevant_items != 0:
            recall += recommended_and_relevant[i] / relevant_items
            no_users_with_rel_items += 1
    recall = recall / no_users_with_rel_items
    return recall


def accuracy(train, test, y_predicted, threshold=0.5):
    '''
    Computes the out-of-sample accuracy of the model given a threshold for binary classification.
    For implicit feedback datasets only.

    Parameters:
    ----------
        - test : array-like
            held-out test set
        - y_predicted : array-like
            predicted values
        - threshold : int

    Return:
    ------
        - acc : int
            Accuracy of the fitted model (no. correct predictions / total predictions, i.e. test set cardinality)
    '''
    y_predicted_threshold = copy.copy(y_predicted)
    prediction, values = y_predicted_threshold[test.nonzero()].flatten(), test[test.nonzero()].flatten()

    indeces_u = np.intersect1d(np.where(test == 0)[0], np.where(train == 0)[0])
    indeces_i = np.intersect1d(np.where(test == 0)[1], np.where(train == 0)[1])

    addit_ind_u = np.random.choice(indeces_u, round(test.shape[0]*0.2), replace=False)
    addit_ind_i = np.random.choice(indeces_i, round(test.shape[0]*0.2), replace=False)

    for ind_u in addit_ind_u:
        for ind_i in addit_ind_i:
            prediction = np.append(prediction, y_predicted[ind_u, ind_i])
            values = np.append(values, 0)

    for row in range(y_predicted_threshold.shape[0]):
        for col in range(y_predicted_threshold.shape[1]):
            if y_predicted_threshold[row, col] > threshold:
                y_predicted_threshold[row, col] = 1
            else:
                y_predicted_threshold[row, col] = 0

    prediction, values = y_predicted_threshold[test.nonzero()].flatten(), test[test.nonzero()].flatten()
    return accuracy_score(values, prediction), prediction, values


def multiclass_accuracy(train, test, y_predicted):
    '''
    Computes the out-of-sample accuracy of the model given the threshold for multiclass classification.
    The thresholds correspond to the predefined quantiles.
    The training set must be passsed in order to pick the right zeros
    (i.e. those in train and test simultaneously) for the final scoring.
    For explicit feedback datasets only.

    Parameters:
    ----------
        - train : array-like
            training set
        - test : array-like
            held-out test set
        - y_predicted : array-like
            predicted values

    Return:
    ------
        - acc : int
            Accuracy of the fitted model (no. correct predictions / total predictions, i.e. test set cardinality)
    '''

    y_predicted_threshold = copy.copy(y_predicted)
    prediction, values = y_predicted_threshold[test.nonzero()].flatten(), test[test.nonzero()].flatten()

    indeces_u = np.intersect1d(np.where(test == 0)[0], np.where(train == 0)[0])
    indeces_i = np.intersect1d(np.where(test == 0)[1], np.where(train == 0)[1])

    addit_ind_u = np.random.choice(indeces_u, round(test.shape[0]*0.2), replace=False)
    addit_ind_i = np.random.choice(indeces_i, round(test.shape[0]*0.2), replace=False)

    for ind_u in addit_ind_u:
        for ind_i in addit_ind_i:
            prediction = np.append(prediction, y_predicted[ind_u, ind_i])
            values = np.append(values, 0)

    return accuracy_score(values, prediction), prediction, values


def mse(train, test, pred):
    values = test[test != 0].flatten()
    prediction = pred[test != 0].flatten()

    indeces_u = np.intersect1d(np.where(test == 0)[0], np.where(train == 0)[0])
    indeces_i = np.intersect1d(np.where(test == 0)[1], np.where(train == 0)[1])

    addit_ind_u = np.random.choice(indeces_u, round(test.shape[0]*0.2), replace=False)
    addit_ind_i = np.random.choice(indeces_i, round(test.shape[0]*0.2), replace=False)

    for ind_u in addit_ind_u:
        for ind_i in addit_ind_i:
            prediction = np.append(prediction, pred[ind_u, ind_i])
            values = np.append(values, 0)

    return mean_squared_error(values, prediction)


def roc(test, y_pred, addit_ind):
    # y_pred[y_pred > 1] = 1
    valid_dim = 0.2
    prediction, values = y_pred[test.nonzero()].flatten(), test[test.nonzero()].flatten()
    addit_ind_u = np.random.choice(np.where(test == 0)[0], round(test.shape[0]*valid_dim), replace=False)
    addit_ind_i = np.random.choice(np.where(test == 0)[1], round(test.shape[0]*valid_dim), replace=False)
    for ind_u in addit_ind_u:
        for ind_i in addit_ind_i:
            prediction = np.append(prediction, y_pred[ind_u, ind_i])
            values = np.append(values, 0)
    return roc_curve(values, prediction)


def roc_auc(train, test, y_pred):
    prediction, values = y_pred[test.nonzero()].flatten(), test[test.nonzero()].flatten()
    indeces_u = np.intersect1d(np.where(test == 0)[0], np.where(train == 0)[0])
    indeces_i = np.intersect1d(np.where(test == 0)[1], np.where(train == 0)[1])

    addit_ind_u = np.random.choice(indeces_u, round(test.shape[0]*0.2), replace=False)
    addit_ind_i = np.random.choice(indeces_i, round(test.shape[0]*0.2), replace=False)
    for ind_u in addit_ind_u:
        for ind_i in addit_ind_i:
            prediction = np.append(prediction, y_pred[ind_u, ind_i])
            values = np.append(values, 0)
    return roc_auc_score(values, prediction)

def cohen_kappa(train, test, y_pred):
    prediction, values = y_pred[test.nonzero()].flatten(), test[test.nonzero()].flatten()
    indeces_u = np.intersect1d(np.where(test == 0)[0], np.where(train == 0)[0])
    indeces_i = np.intersect1d(np.where(test == 0)[1], np.where(train == 0)[1])

    addit_ind_u = np.random.choice(indeces_u, round(test.shape[0]*0.2), replace=False)
    addit_ind_i = np.random.choice(indeces_i, round(test.shape[0]*0.2), replace=False)
    for ind_u in addit_ind_u:
        for ind_i in addit_ind_i:
            prediction = np.append(prediction, y_pred[ind_u, ind_i])
            values = np.append(values, 0)
    return cohen_kappa_score(values, prediction)