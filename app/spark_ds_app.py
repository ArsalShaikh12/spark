from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Data Science Pipeline") \
    .config("spark.ui.port", "4040") \
    .getOrCreate()

# Load Dataset
print("Loading dataset...")
data_path = "/data/Housing.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Display Schema
print("Dataset Schema:")
df.printSchema()

# EDA: Summary Statistics
print("Summary Statistics:")
df.describe().show()

# Data Munging: Handle Missing Values
print("Handling missing values...")
df = df.dropna()

# Visualization: Histogram of 'price' column
print("Creating a histogram of prices...")
pandas_df = df.select("price").toPandas()
plt.hist(pandas_df["price"], bins=30, color='blue')
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.savefig("/data/price_histogram.png")

# ML: Simple Linear Regression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Feature Engineering
features = ["area", "bedrooms", "bathrooms"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
df = assembler.transform(df).select("features", col("price").alias("label"))

# Train-Test Split
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Train Linear Regression Model
lr = LinearRegression(featuresCol="features", labelCol="label")
model = lr.fit(train)

# Evaluate Model
print("Model Coefficients:", model.coefficients)
print("Model Intercept:", model.intercept)

# Predictions
predictions = model.transform(test)
predictions.show(5)

# Save predictions
predictions.select("features", "label", "prediction").write.csv("/data/predictions.csv")

print("Pipeline Execution Completed.")

# Stop Spark Session
spark.stop()
