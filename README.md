# **AI and Machine Learning Fundamentals Lab Guide**

## **Table of Contents**

1. [Introduction](#Introduction)
2. [Prerequisites](#Prerequisites)
3. [Installation Steps](#Installation_Steps)
4. [Lab_Activities](#Lab_Activities)
   - 4.1. Set up a Python Environment
   - 4.2. Linear Regression with Scikit-learn
   - 4.3. Decision Tree Classifier
   - 4.4. Data Preprocessing (Normalization and Encoding)
   - 4.5. Neural Network using TensorFlow or PyTorch
   - 4.6. K-Means Clustering
   - 4.7. Cross-Validation and Hyperparameter Tuning
   - 4.8. Data Visualization (Matplotlib & Seaborn)
   - 4.9. Support Vector Machine (SVM) Model
   - 4.10. Deploying a Model as a REST API
5. [Conclusion](#Conclusion)
6. [References](#References)



# **Introduction**

This guide aims to provide hands-on experience with various machine learning (ML) and artificial intelligence (AI) concepts. You will learn how to implement fundamental machine learning models, preprocess data, and deploy models through a REST API. All activities will be done using **VSCode** to enhance productivity and keep code well-organized.

# **2. Prerequisites**

Before starting the labs, ensure the following prerequisites are met:

## **1. Operating System:**
   - **Linux (Recommended)**: Ubuntu 18.04 LTS or later for better compatibility and performance.
   - **Windows/MacOS**: Supported via **Windows Subsystem for Linux (WSL)** or native installations.
   - **Python Version**: Ensure **Python 3.8** or later is installed.

## **2. Python and VSCode Setup:**
   - **Python**: Install Python 3.x from [python.org](https://www.python.org/).
   - **VSCode**: Download and install [Visual Studio Code](https://code.visualstudio.com/).
   - **VSCode Python Extension**: Ensure the **Python extension** for VSCode is installed. This can be done through the Extensions panel in VSCode.

## **3. Required Libraries:**
   Install the following Python libraries using **pip**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tensorflow torch torchvision torchaudio fastapi uvicorn
   ```

## **4. Resources:**
   - **CPU**: Minimum 1 core (2+ cores recommended).
   - **Memory**: At least 4 GB of RAM.
   - **Storage**: Minimum 20 GB of free disk space.


# **3. Installation Steps**

## **Step 1: Install Python and VSCode**
   - Download and install **Python 3.x** from [python.org](https://www.python.org/).
   - Download and install **VSCode** from [Visual Studio Code](https://code.visualstudio.com/).


## **Step 2: Set Up Virtual Environment**
   Open VSCode terminal and create a virtual environment to isolate the project:
   ```bash
   python -m venv ml_env
   source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
   ```

## **Step 3: Install Required Packages**
   Once the environment is activated, install all the required libraries:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn tensorflow torch fastapi uvicorn
   ```

## **Step 4: Configure VSCode**
   - Open **VSCode**.
   - Install the **Python extension** (if not installed).
   - Open your project folder in VSCode and ensure the correct interpreter is selected (the one from the virtual environment).



# **4. Lab_Activities**

## **4.1. Set up a Python Environment**
Test that your environment and libraries are installed correctly:
```python
import numpy as np
import pandas as pd

# Create random data
data = pd.DataFrame({'x': np.random.randn(100), 'y': np.random.randn(100)})
print(data.head())
```

Save the file as `setup.py`, open it in **VSCode**, and run the script using the terminal (`python setup.py`).

---

## **4.2. Linear Regression with Scikit-learn**

1. Load sample data.
2. Split it into training and test sets.
3. Train a linear regression model and evaluate it.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
X = data[['x']]
y = data['y']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
```

Save the file as `linear_regression.py` and run it in **VSCode**.

---

#### **4.3. Decision Tree Classifier**

1. Load the Iris dataset.
2. Train a decision tree classifier and predict the target values.

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train decision tree
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Make predictions
predictions = clf.predict(X)
print(f'Predictions: {predictions}')
```

Save the file as `decision_tree.py` and run it in VSCode.

---

#### **4.4. Data Preprocessing (Normalization and Encoding)**

1. Normalize numerical data using `StandardScaler`.
2. Perform one-hot encoding on categorical data.

```python
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Normalize numerical data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

# Example of encoding categorical data (if present)
data_with_categorical = pd.get_dummies(data)
print(data_with_categorical.head())
```

Save this as `data_preprocessing.py` and run it.

---

#### **4.5. Neural Network using TensorFlow or PyTorch**

Create a simple feedforward neural network for regression:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

Save this as `neural_network.py` and run it.

---

#### **4.6. K-Means Clustering**

Segment the data using K-means clustering.

```python
from sklearn.cluster import KMeans

# K-Means Clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
print(f'Cluster labels: {labels}')
```

Save it as `k_means.py` and run it.

---

#### **4.7. Cross-Validation and Hyperparameter Tuning**

Apply K-fold cross-validation and hyperparameter tuning:

```python
from sklearn.model_selection import GridSearchCV

# Define parameters to tune
params = {'max_depth': [3, 5, 7]}

# Grid search for hyperparameter tuning
grid_search = GridSearchCV(DecisionTreeClassifier(), params, cv=5)
grid_search.fit(X, y)

print(f'Best parameters: {grid_search.best_params_}')
```

Save as `cross_validation.py` and run.

---

#### **4.8. Data Visualization (Matplotlib & Seaborn)**

Visualize data distributions and relationships:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize pairplot
sns.pairplot(data)
plt.show()
```

Save as `visualization.py` and run.

---

#### **4.9. Support Vector Machine (SVM) Model**

Train and evaluate an SVM model:

```python
from sklearn.svm import SVC

# Train SVM
svm = SVC()
svm.fit(X_train, y_train)

# Evaluate
accuracy = svm.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
```

Save as `svm.py` and run.

---

#### **4.10. Deploying a Model as a REST API**

Deploy a trained model using **FastAPI**:

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/predict")
def predict(input_data: dict):
    # Model prediction logic goes here
    return {"prediction": "model result"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

Save it as `api.py` and run it using:
```bash
uvicorn api:app --reload
```

---

### **5. Content in Excel**

You can track the results of each lab activity (such as accuracy, MSE, or cluster labels) in an Excel sheet for documentation and comparison.

### **6. Conclusion**

This lab guide covered a broad range of machine learning fundamentals, from data preprocessing to model

 deployment. Working through these activities will give you hands-on experience in key AI/ML concepts.

### **7. References**

- [Python Official Website](https://www.python.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

--- 

This guide is designed to ensure clarity and ease of understanding for beginners, all while using VSCode as the primary IDE.
