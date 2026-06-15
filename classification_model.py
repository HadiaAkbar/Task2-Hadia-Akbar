import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris


def run_classification():

    print("========================================")
    print(" DecodeLabs Machine Learning Classifier ")
    print("========================================")

    try:
        iris = load_iris()

        data = pd.DataFrame(
            iris.data,
            columns=iris.feature_names
        )

        data["target"] = iris.target

        print("\nDataset Information:")
        print("Total Records:", len(data))
        print("Features:", iris.feature_names)
        print("Classes:", iris.target_names)


        X = data.drop("target", axis=1)
        y = data["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        print("\nData Split Completed")
        print("Training Samples:", len(X_train))
        print("Testing Samples:", len(X_test))


        model = KNeighborsClassifier(n_neighbors=3)

        print("\nTraining KNN Model...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)


        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print("\nModel Evaluation:")
        print("Accuracy:", round(accuracy * 100, 2), "%")

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=iris.target_names
            )
        )


        sample = [[5.1, 3.5, 1.4, 0.2]]

        result = model.predict(sample)

        print("Sample Prediction:")
        print("Input Values:", sample[0])
        print(
            "Predicted Flower:",
            iris.target_names[result[0]]
        )


    except Exception as e:
        print("Something went wrong:", e)



if __name__ == "__main__":
    run_classification()
