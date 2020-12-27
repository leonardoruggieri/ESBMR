import numpy as np


def train_val_split(data: np.array, valid_dim=0.2):
    '''
    Creating two additional objects, i.e. training and validation set, which can be used in the fitting process

    Parameters:
    data = np.array
    valid_dim = float
    '''
    if valid_dim >= 1:
        raise ValueError("valid_dim must be lower than 1")

    train = data.copy()
    valid = np.zeros(data.shape)

    for u in np.unique(data.nonzero()[0]):
        ind = data[u].nonzero()[0]

        if len(ind) > 0:
            valid_ind = np.random.choice(ind, round(len(ind)*valid_dim), replace=False)
            for i in valid_ind:
                valid[u, i], train[u, i] = data[u, i], 0
    return train, valid


def train_val_split_zero(data: np.array, valid_dim=0.2):
    '''
    Creating two additional objects, i.e. training and validation set, which can be used in the fitting process.
    It also includes zeros in the test set from the original input data matrix.
    In particular, half of the held-out test set is made of zeros and the other half by non-zeros.

    Parameters:
    data = np.array
    valid_dim = float
    '''
    if valid_dim >= 1:
        raise ValueError("valid_dim must be lower than 1")

    train = data.copy()
    valid = np.zeros(data.shape)

    for u in np.array(np.unique(data), dtype=int):
        ind = np.array(data[u], dtype=int)
        valid_ind_1 = np.random.choice(ind, round(len(ind)*valid_dim/2), replace=False)
        for i in valid_ind_1:
            valid[u, i], train[u, i] = data[u, i], 0

    for u in np.unique(data.nonzero()[0]):
        ind = data[u].nonzero()[0]
        if len(ind) > 0:
            valid_ind_2 = np.random.choice(ind, round(len(ind)*valid_dim/2), replace=False)
            for i in valid_ind_2:
                valid[u, i], train[u, i] = data[u, i], 0
    return train, valid
