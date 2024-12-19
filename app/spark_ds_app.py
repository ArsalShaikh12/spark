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
data_path = "./data/housing.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)

# Rename Columns to remove spaces or special characters
df = df.select([col(f"`{c}`").alias(c.replace(' ', '_').replace('.', '')) for c in df.columns])

# Display Schema
print("Dataset Schema:")
df.printSchema()

# EDA: Summary Statistics
print("Summary Statistics:")
df.describe().show()

# Data Munging: Handle Missing Values
print("Handling missing values...")
df = df.dropna()

# Visualization: Histogram of 'Price' column
print("Creating a histogram of prices...")
pandas_df = df.select("Price").toPandas()
plt.hist(pandas_df["Price"], bins=30, color='blue')
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.savefig("./price_histogram.png")

# ML: Simple Linear Regression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression


# Convert string columns to numeric type
df = df.withColumn("Avg_Area_Income", col("Avg_Area_Income").cast("double"))
df = df.withColumn("Avg_Area_House_Age", col("Avg_Area_House_Age").cast("double"))

# Feature Engineering
features = ["Avg_Area_Income", "Avg_Area_House_Age", "Area_Population"]
assembler = VectorAssembler(inputCols=features, outputCol="features")

# Transform the data
df = assembler.transform(df).select("features", col("Price").alias("label"))

# Continue with your pipeline...


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
from pyspark.ml.linalg import Vector

# Extract individual components from the 'features' vector (optional, if needed for the final dataset)
predictions = predictions.withColumn("feature_1", col("features").getItem(0)) \
                         .withColumn("feature_2", col("features").getItem(1)) \
                         .withColumn("feature_3", col("features").getItem(2))

# Select relevant columns to write out, including the extracted features and predictions
predictions.select("feature_1", "feature_2", "feature_3", "label", "prediction").write.csv("./predictions.csv")

print("Predictions saved successfully.")



print("Pipeline Execution Completed.")

# Stop Spark Session
spark.stop()
