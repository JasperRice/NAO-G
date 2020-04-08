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
    return X_train, X_test, X_val, Y_train, Y_test, Y_val


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from sklearn.metrics import mean_squared_error
    
    from execute import execGesture
    from io_routines import readCSV

    TEST = "NAO"
    if TEST is "TALK":
        talk_01 = readCSV("human2robot/dataset/TALK_01.csv")
        talk_02 = readCSV("human2robot/dataset/TALK_02.csv")
        talk = np.vstack((talk_01, talk_02))

        talk_normalized, talk_scaler = normalize(talk)
        talk_normalized_decomposed, talk_pca = decompose(talk_normalized)

        talk_normalized_composed = talk_pca.inverse_transform(talk_normalized_decomposed)
        talk_denormalized_composed = talk_scaler.inverse_transform(talk_normalized_composed)

        print("Components after decomposition: "+str(talk_pca.n_components_))
        print("MSE between normalized data: "+str(mean_squared_error(talk_normalized, talk_normalized_composed)))
        print("MSE between original data: "+str(mean_squared_error(talk, talk_denormalized_composed)))
    elif TEST is "NAO":
        nao = readCSV("human2robot/dataset/NAO.csv")
        
        nao_normalized, nao_scaler = normalize(nao)
        nao_normalized_decomposed, nao_pca = decompose(nao_normalized)

        nao_normalized_composed = nao_pca.inverse_transform(nao_normalized_decomposed)
        nao_denormalized_composed = nao_scaler.inverse_transform(nao_normalized_composed)
        # execGesture("127.0.0.1", 45817, nao[50][2:].tolist())
        execGesture("127.0.0.1", 45817, nao_denormalized_composed[50][2:].tolist())

        print("Components after decomposition: "+str(nao_pca.n_components_))