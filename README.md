# PySpark Breast Cancer Classifier 🧠🔥  
This mini-project demonstrates how to use **PySpark**, the Python API for Apache Spark, to build a logistic regression model that classifies breast cancer cases using real-world medical data.  
The goal is to showcase a simple end-to-end **machine learning pipeline** using PySpark's DataFrame API and MLlib for modeling and evaluation.  
It includes both local and Databricks Community Edition instructions so it can run in any environment without complex setup.

## 🚀 What’s Inside  
- Load and preprocess structured data using **PySpark**
- Feature engineering using `VectorAssembler`
- Split the data for training and testing
- Build and evaluate a **logistic regression** model
- Calculate **AUC** (Area Under ROC Curve) as a performance metric
- Fully compatible with Databricks Free Edition


## 🗂️ Dataset  
We use the **Breast Cancer Wisconsin dataset** available from `sklearn.datasets`. It includes diagnostic features extracted from breast mass imagery.

## 🛠️ Tech Stack
- Python
- PySpark
- Scikit-learn (for dataset loading)
- Pandas (for data conversion)
- Spark MLlib (for modeling)
- Databricks Free Edition (optional)


## ▶️ How to Run  
Install the required packages:
```bash
pip install -r requirements.txt
```
Run the script:
```bash
python breast_cancer_spark.py
```

▶️ How to Run on Databricks (No Java Setup Needed)
1. Sign up for Databricks Community Edition
2. Create a New Notebook
3. Set Language to Python
4. Create a Cluster (runtime Spark 3.x)
5. Paste the Databricks-friendly code (same as in breast_cancer_spark.py, slightly adapted)
6. Click Run All
   
## 📊 Output  
The script prints the **AUC score** for model performance:
```
AUC on test data: 0.98
```
