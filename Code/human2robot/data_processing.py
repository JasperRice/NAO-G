
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    breast_cancer = load_breast_cancer()
    breast_cancer_data = breast_cancer.data

    print breast_cancer_data.shape