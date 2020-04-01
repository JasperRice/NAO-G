from sklearn.preprocessing import StandardScaler
input_scaler = StandardScaler()
output_scaler = StandardScaler()




if __name__ == "__main__":
    TEST = True
    if TEST:
        from sklearn.datasets import load_breast_cancer
        breast_cancer = load_breast_cancer()
        breast_cancer_data = breast_cancer.data

        input_scaler.fit(breast_cancer_data)
        breast_cancer_data_normalized = input_scaler.transform(breast_cancer_data)

        breast_cancer_data_denormalized = input_scaler.inverse_transform(breast_cancer_data_normalized)

        print breast_cancer_data[0]
        print breast_cancer_data_normalized[0]
        print breast_cancer_data_denormalized[0]
    else:
        pass