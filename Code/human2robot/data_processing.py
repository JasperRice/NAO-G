import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def normalize(data):
    """Normalize the data and save the scaler so that the inverse tranform can be done
    
    :param data: [description]
    :type data: numpy.ndarray
    :return: [description]
    :rtype: numpy.ndarray, sklearn.preprocessing.data.StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data), scaler


def decompose(data):
    """Use PCA to decompose the dimensionality of data
    
    :param data: [description]
    :type data: numpy.ndarray
    :return: [description]
    :rtype: [type]
    """
    n, d = np.shape(data)
    pca = PCA(0.95)
    pca.fit(data)
    data_decomposed = pca.transform(data)
    return data_decomposed, pca


def split(X, Y):
    """Seperate the dataset into train, test, and validation
    X and Y should have same amount of data
    
    :param X: [description]
    :type X: numpy.ndarray
    :param Y: [description]
    :type Y: numpy.ndarray
    :return: [description]
    :rtype: [type]
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1000
    )
    X_test, X_val, Y_test, Y_val = train_test_split(
        X_test, Y_test, test_size=0.5, random_state=2000
    )
    return X_train, X_test, X_val, Y_train, Y_test, Y_train


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    TEST = False
    if TEST:
        from sklearn.datasets import load_breast_cancer
        breast_cancer = load_breast_cancer()
        breast_cancer_data = breast_cancer.data
        breast_cancer_data_normalized, breast_cancer_scaler = normalize(breast_cancer_data)
        breast_cancer_data_denormalized = breast_cancer_scaler.inverse_transform(breast_cancer_data_normalized)

    else:
        from io_routines import readCSV
        human_data = readCSV("dataset/HUMAN.csv")
        human_data_normalized, human_scaler = normalize(human_data)
        human_data_normalized_decomposed, human_pca = decompose(human_data_normalized)

        human_data_normalized_inverse = human_pca.inverse_transform(human_data_normalized_decomposed)
        human_data_inverse_inverse = human_scaler.inverse_transform(human_data_normalized_inverse)

        