from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Step 1: Create Spark session
spark = SparkSession.builder.appName("BreastCancerClassification").getOrCreate()

# Step 2: Load dataset using sklearn and convert to Spark DataFrame
data = load_breast_cancer()
df_pd = pd.DataFrame(data.data, columns=data.feature_names)
df_pd["target"] = data.target
df_spark = spark.createDataFrame(df_pd)

# Step 3: Assemble features
features = data.feature_names[:5]  # Use first 5 features
vec_assembler = VectorAssembler(inputCols=list(features), outputCol="features")
df_features = vec_assembler.transform(df_spark)

# Step 4: Split data
train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)

# Step 5: Train logistic regression model
lr = LogisticRegression(labelCol="target", featuresCol="features")
model = lr.fit(train_df)

# Step 6: Make predictions and evaluate
predictions = model.transform(test_df)
evaluator = BinaryClassificationEvaluator(labelCol="target")
auc = evaluator.evaluate(predictions)

print(f"AUC on test data: {auc:.4f}")

spark.stop()