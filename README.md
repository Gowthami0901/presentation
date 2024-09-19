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
<br>

# **Prerequisites**

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


# **Installation_Steps**

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



# **Lab_Activities**

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
