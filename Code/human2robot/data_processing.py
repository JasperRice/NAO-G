import numpy as np

from sklearn.decomposition import PCA
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


if __name__ == "__main__":
    
    TEST = False
    if TEST:
        from sklearn.datasets import load_breast_cancer
        breast_cancer = load_breast_cancer()
        breast_cancer_data = breast_cancer.data
        breast_cancer_data_normalized, breast_cancer_scaler = normalize(breast_cancer_data)
        breast_cancer_data_denormalized = breast_cancer_scaler.inverse_transform(breast_cancer_data_normalized)

        print breast_cancer_data[0]
        # print breast_cancer_data_normalized[0]
        print breast_cancer_data_denormalized[0]
    else:
        from io_routines import readCSV
        human_data = readCSV("HUMAN.csv")
        human_data_normalized, human_scaler = normalize(human_data)
        human_data_denormalized = human_scaler.inverse_transform(human_data_normalized)

        print human_data[0]
        # print human_data_normalized[0]
        print human_data_denormalized

        new_human_data_normalized, human_pca = decompose(human_data_normalized)
        print new_human_data_normalized[0]