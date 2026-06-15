import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

def main():
    print("--- Welcome to DecodeLabs Data Classification Engine ---")
    
   
    print("\n[1] Loading the Iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    print("Feature Names:", iris.feature_names)
    print("Target Classes:", iris.target_names)
    
   
    print("\n[2] Splitting data into training (80%) and testing (20%) sets...")
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    
    print("\n[3] Applying K-Nearest Neighbors (KNN) algorithm...")
    knn = KNeighborsClassifier(n_neighbors=3)
    
   
    print("Training the model...")
    knn.fit(X_train, y_train)
    
    
    print("Making predictions on the test set...")
    y_pred = knn.predict(X_test)
    
  
    print("\n[4] Validating the model performance...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    
    print("\n--- Sample Prediction ---")
    sample_data = [[5.1, 3.5, 1.4, 0.2]]\
    prediction = knn.predict(sample_data)
    print(f"Input Features: {sample_data[0]}")
    print(f"Predicted Class: {iris.target_names[prediction[0]]}")

if __name__ == "__main__":
    main()
