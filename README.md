**AI and Machine Learning Fundamentals Lab**:

---

### **1. Set up a Python environment with libraries like NumPy, Pandas, and Scikit-learn**
   - [NumPy Documentation](https://numpy.org/doc/stable/)
   - [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
   - [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

---

### **2. Implement linear regression using Scikit-learn on a sample dataset**
   - [Scikit-learn: Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
   - [Linear Regression Example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html)

---

### **3. Build and evaluate a decision tree classifier on a real-world dataset**
   - [Scikit-learn: Decision Tree Classifier](https://scikit-learn.org/stable/modules/tree.html#classification)
   - [Decision Tree Example on Iris Dataset](https://scikit-learn.org/stable/auto_examples/tree/plot_iris.html)

---

### **4. Perform data preprocessing tasks like normalization and encoding using Pandas**
   - [Pandas: DataFrame Normalization](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
   - [Scikit-learn: Preprocessing Data](https://scikit-learn.org/stable/modules/preprocessing.html)

---

### **5. Create and train a simple neural network using TensorFlow or PyTorch**
   - [TensorFlow Official Guide](https://www.tensorflow.org/guide)
   - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
   - [Building Neural Networks with TensorFlow](https://www.tensorflow.org/tutorials/quickstart/beginner)
   - [Building Neural Networks with PyTorch](https://pytorch.org/tutorials/beginner/nn_tutorial.html)

---

### **6. Use K-means clustering to segment a dataset into different groups**
   - [Scikit-learn: K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
   - [K-Means Example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html)

---

### **7. Implement cross-validation and hyperparameter tuning for a machine learning model**
   - [Scikit-learn: Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
   - [Scikit-learn: Grid Search for Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)

---

### **8. Visualize data distributions and model results using Matplotlib and Seaborn**
   - [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
   - [Seaborn Documentation](https://seaborn.pydata.org/)

---

### **9. Train a support vector machine (SVM) model and evaluate its performance**
   - [Scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
   - [SVM Classifier Example](https://scikit-learn.org/stable/auto_examples/classification/plot_iris_svc.html)

---

### **10. Deploy a trained machine learning model as a REST API using Flask or FastAPI**
   - [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
   - [Flask Official Documentation](https://flask.palletsprojects.com/en/2.0.x/)
   - [Deploying ML Models with FastAPI](https://towardsdatascience.com/deploying-machine-learning-models-as-api-using-fastapi-1730c30239fa)
   - [Deploying ML Models with Flask](https://towardsdatascience.com/deploy-machine-learning-models-using-flask-8bd7911a6b23)

---



# Lab Guide: Setting Up a Python Environment with NumPy, Pandas, and Scikit-learn

# Table of Contents

1. [Description](#Description)
2. **Problem Statement**
3. **Prerequisites**
   - Software Required
   - Hardware Requirements
4. **Setup Instructions**
   - Step 1: Install Python
   - Step 2: Verify Python Installation
   - Step 3: Install Libraries
   - Step 4: Verify Library Installation
5. **Reference**
6. **Downloadable File**

---

## 1. Description
This lab guide provides step-by-step instructions to set up a Python environment equipped with essential libraries such as NumPy, Pandas, and Scikit-learn, which are vital for data manipulation, analysis, and machine learning tasks.

## 2. Problem Statement
Setting up a Python environment with the right libraries is crucial for conducting data science and machine learning projects. This guide aims to simplify the process for beginners and provide a solid foundation for further exploration in AIML.

## 3. Prerequisites

### Software Required
- **Python Installation:** You can choose either:
  - **Anaconda**: A popular distribution that comes with many data science packages pre-installed.
  - **Standalone Python**: The basic Python installation, which requires manual installation of libraries.

### Hardware Requirements
- **Minimum System Requirements**:
  - CPU: Intel Core i3 or equivalent
  - RAM: 4 GB (8 GB recommended for better performance)
  - Disk Space: 1 GB free for Python and libraries installation

## 4. Setup Instructions

## **Step-by-Step Python Installation Guide**


## **Installing Python on Windows**

**Step 1: Download the Python Installer:**

   ![Python Installation](https://www.python.org/static/img/python-logo.png)

   - Visit the [official Python website](https://www.python.org/downloads/).
   - Click on the yellow **"Download Python 3.x.x"** button (the latest stable version will be shown).
   ![alt text](Images/image.png)

---

**Step 2: Run the Installer:**
   1. Run the downloaded Python Installer.
   2. The installation window shows two checkboxes:
      - **Admin privileges:** The parameter controls whether to install Python for the current or all system users. This option allows you to change the installation folder for Python.

      - **Add Python to PATH:** The second option places the executable in the PATH variable after installation. You can also add Python to the PATH environment variable manually later.

      ![alt text](Images/img3.JPG)

   3. Select the **Install Now** option for the recommended installation (in that case, skip the next two steps).

   4. To adjust the default installation options, choose Customize installation instead and proceed to the following step.
   
    - **Installation Directory:** `C:\Users\[user]\AppData\Local\Programs\Python\Python[version]`
    - **Included Components:**
    - **IDLE** (the default Python Integrated Development and Learning Environment).
    - **PIP** (Python's package installer).
    - **Additional Documentation.**
    - The installer also creates:
    - Required shortcuts.
    - File associations for `.py` files.

    If you choose the **"Customize Installation"** option during setup, you can modify the default configurations, such as the installation location, optional features, and advanced settings. This flexibility allows you to tailor the setup to your specific project requirements or environment.


**Step 3: Choose the optional installation features**
   - Python works without these features, but adding them improves the program's usability.

   ![alt text](Images/img3(1).JPG)

   - Click Next to proceed to the Advanced Options screen.

**Step 4: Choosing advanced options**
   - Choose whether to install Python for all users. The option changes the install location to C:\Program Files\Python[version]. 

   - If selecting the location manually, a common choice is C:\Python[version] because it avoids spaces in the path, and all users can access it.Due to administrative rights, both paths may cause issues during package installation.
  
   ![alt text](Images/img3(2).JPG)

   After picking the appropriate options, click Install to start the installation.

   ![alt text](Images/img3(3).JPG)

**Step 5: Final Setup**
   - Select whether to disable the path length limit. Choosing this option will allow Python to bypass the 260-character MAX_PATH limit.

   ![alt text](Images/img5.JPG)

   - The option will not affect any other system settings, and disabling it resolves potential name-length issues. We recommend selecting the option and closing the setup.

**Step 6: Add Python to Path (Optional)**

If the Python installer does not include the Add Python to PATH checkbox or you have not selected that option, continue in this step. Otherwise, skip to the next step.

To add Python to PATH, do the following:

1. In the **Start menu**, search for **Environment Variables** and press **Enter**.

![alt text](Images/image-2.png)

2. Click **Environment Variables** to open the overview screen.

![alt text](Images/img8.JPG)

3. Double-click **Path** on the list to edit it.

![alt text](Images/img9.JPG)

Alternatively, select the variable and click the Edit button.

4. Double-click the first empty field and paste the Python installation folder path.

![alt text](Images/img10.JPG)

Alternatively, click the **New button** instead and paste the path. Click **OK** to save the changes.


**Step 7: Verify Python Was Installed on Windows**

The first way to verify that Python was installed successfully is through the command line. Open the command prompt and run the following command:

```bash
python --version
```

![alt text](Images/img11.JPG)
The output shows the installed Python version.

**Verify PIP Was Installed**

To verify whether PIP was installed, enter the following command in the command prompt:

```bash
pip --version
```
If it was installed successfully, you should see the PIP version number, the executable path, and the Python version:

![alt text](Images/img12.JPG)

PIP has not been installed yet if you get the following output:

```
'pip' is not recognized as an internal or external command,
Operable program or batch file.
```

**Step 8: Connect VScode with Python**
To set up Visual Studio Code (VS Code) with Python, follow these steps:

1. **Install Visual Studio Code:**
   - Download and install the latest version of [Visual Studio Code](https://code.visualstudio.com/Download) for your operating system.

2. **Install the Python Extension for VS Code:**
   - Open VS Code.
   - Go to the **Extensions** view by clicking on the square icon in the left sidebar or pressing `Ctrl+Shift+X`.

   ![alt text](Images/img14.JPG)

   - Search for the **Python** extension by Microsoft.

   ![alt text](Images/img15.JPG)

   - Click **Install** to add the extension to your VS Code.

3. **Create a Python File:**
   - Open a new file in your workspace.
   - Save it with the `.py` extension (e.g., `modal.py`).

4. **Write and Run Python Code:**
   - Type a simple Python script, such as:

     ```python
     print("Hello, World!")
     ```

   - Save the file.

   ![alt text](Images/img16.png)

   - To run the code, right-click in the file and select **Run Python File in Terminal**, or click the **Run** button in the top-right corner.

   ![alt text](Images/img17.JPG)
   This is the terminal output (Hello World)


5. **(Optional) Set Up Virtual Environments:**
   - It's recommended to use virtual environments for isolated project dependencies.
   - Create a virtual environment using:

     ```bash
     python -m venv .venv
     ```

   - Activate the virtual environment in the terminal using:

     ```bash
     .venv\Scripts\activate  # For Windows
     ```

   - VS Code will automatically detect and prompt you to use the virtual environment as the Python interpreter.

**Step 8: Install Libraries**

To install the essential libraries, open your command prompt and run the following command:

```bash
pip install numpy pandas scikit-learn
```

This command will install NumPy, Pandas, and Scikit-learn, which are essential for data manipulation, analysis, and machine learning tasks.

**Step 9: Verify Library Installation**

To verify the installation, you can run a simple script that imports the libraries and prints their versions. Follow these steps to create and run the script in Visual Studio Code (VS Code):

**1. Open VS Code**
   - Launch Visual Studio Code from your Start menu or desktop shortcut.

 2. Create a New Python File
   - Click on `File > New File` or press `Ctrl+N` to create a new file.
   - Save the file with a `.py` extension, e.g., `verify_libraries.py`, by clicking on `File > Save As` or pressing `Ctrl+S`.

 3. Write the Verification Script
   - In the new file, type the following Python script:

     ```python
     import numpy as np
     import pandas as pd
     import sklearn

     print("NumPy version:", np.__version__)
     print("Pandas version:", pd.__version__)
     print("Scikit-learn version:", sklearn.__version__)
     ```

 4. Run the Script in VS Code
   - Right-click inside the editor window and select `Run Python File in Terminal`, or click the `Run` button in the top-right corner of the editor.
   - The script will execute, and the terminal at the bottom of the VS Code window will display the versions of the installed libraries.

Example Output
   - After running the script, you should see an output similar to the following in the terminal:

     ```
     NumPy version: 1.21.0
     Pandas version: 1.3.0
     Scikit-learn version: 0.24.2
     ```


