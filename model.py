# Import Packages
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, IsotonicRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from make_dataframes import company_data
import time
from pyspark.sql import SparkSession


# Function for train and test data for different models
def evaluate_model(regressor, model_name, trainingdata, testdata):
    # Pipeline
    pipeline = Pipeline(stages=[regressor])

    # Measure training time
    start_train_time = time.time()

    model = pipeline.fit(trainingdata)
    training_time = time.time() - start_train_time

    # Measure prediction time
    start_pred_time = time.time()
    predictions = model.transform(testdata)
    prediction_time = time.time() - start_pred_time

    # Evaluate RMSE
    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    # Display results
    print(f"Results for {model_name}:")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  - Training Time: {training_time:.2f} seconds")
    print(f"  - Prediction Time: {prediction_time:.2f} seconds\n")

    # Show example predictions
    predictions.select("prediction", "label", "features").show(5, truncate=False)

    return rmse, training_time, prediction_time


def core_compare(partition_flag: bool):
    # Get the processed DataFrame
    df = company_data()  # This calls your function and gets the combined DataFrame

    # Run models using different number of cores from 1 to 5
    for core in range(1, 6):
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("RegressionModelsComparison") \
            .master("local[*]") \
            .config("spark.executor.cores", "1") \
            .config("spark.executor.instances", str(core)) \
            .config("spark.driver.cores", "1") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "4g") \
            .getOrCreate()

        # Suppress warnings by setting log level
        spark.sparkContext.setLogLevel("ERROR")

        # print number of cores used
        print("Number of Cores:", spark.sparkContext.getConf().get(key="spark.executor.instances"))

        # Run models w/ and w/o data partition
        if partition_flag:
            # Partition the data base on the Company and number of cores
            print("Run models with data partition.")
            partitioned_df = df.repartition(core, df["Company"])
            partitioned_df.cache()
        else:
            print("Run models without data partition.")
            partitioned_df = df

        # Rename the target column for regression
        partitioned_df = partitioned_df.withColumnRenamed("Close", "label")

        # Assemble features into a single vector column
        feature_columns = [
            "open_1", "open_2", "open_3", "open_4", "open_5",
            "close_1", "close_2", "close_3", "close_4", "close_5",
            "high_1", "high_2", "high_3", "high_4", "high_5",
            "low_1", "low_2", "low_3", "low_4", "low_5",
            "volume_1", "volume_2", "volume_3", "volume_4", "volume_5"
        ]

        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        partitioned_df = assembler.transform(partitioned_df)

        # Split the data into training and test sets
        (trainingData, testData) = partitioned_df.randomSplit([0.7, 0.3], seed=42)

        # Random Forest Regression
        rf = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=50, maxDepth=10)
        evaluate_model(rf, "Random Forest Regression", trainingData, testData)

        # Gradient Boosted Tree Regression
        gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=20, maxDepth=10)
        evaluate_model(gbt, "Gradient Boosted Tree Regression", trainingData, testData)

        # Isotonic Regression (Special Case)
        # Prepare data for isotonic regression by using a single feature and label
        isotonic_data = partitioned_df.select("high_1", "label").withColumnRenamed("high_1", "features")
        isotonic_data = isotonic_data.withColumn("features", isotonic_data["features"].cast("double"))

        # Measure training time for Isotonic Regression
        start_train_time = time.time()
        iso_model = IsotonicRegression(featuresCol="features", labelCol="label").fit(isotonic_data)
        training_time_iso = time.time() - start_train_time

        # Measure prediction time for Isotonic Regression
        start_pred_time = time.time()
        iso_predictions = iso_model.transform(isotonic_data)
        prediction_time_iso = time.time() - start_pred_time

        # Evaluate RMSE for Isotonic Regression
        evaluator_iso = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
        rmse_iso = evaluator_iso.evaluate(iso_predictions)

        # Display results for Isotonic Regression
        print(f"Results for Isotonic Regression:")
        print(f"  - Root Mean Squared Error (RMSE): {rmse_iso:.4f}")
        print(f"  - Training Time: {training_time_iso:.2f} seconds")
        print(f"  - Prediction Time: {prediction_time_iso:.2f} seconds\n")

        # Show example predictions for Isotonic Regression
        iso_predictions.select("prediction", "label", "features").show(5, truncate=False)

        # cloase the Spark Session
        spark.stop()


if __name__ == "__main__":
    core_compare(partition_flag=False)
    core_compare(partition_flag=True)

# Results
# core number: 1
# w/ partition
# Results for Random Forest Regression:
#   - Root Mean Squared Error (RMSE): 20.4076
#   - Training Time: 23.97 seconds
#   - Prediction Time: 0.77 seconds
# Results for Gradient Boosted Tree Regression:
#   - Root Mean Squared Error (RMSE): 22.9709
#   - Training Time: 28.60 seconds
#   - Prediction Time: 0.36 seconds
# Results for Isotonic Regression:
#   - Root Mean Squared Error (RMSE): 6.9436
#   - Training Time: 1.00 seconds
#   - Prediction Time: 0.09 seconds
#
# w/o partition
# Results for Random Forest Regression:
#   - Root Mean Squared Error (RMSE): 19.1792
#   - Training Time: 25.57 seconds
#   - Prediction Time: 0.73 seconds
# Results for Gradient Boosted Tree Regression:
#   - Root Mean Squared Error (RMSE): 24.7308
#   - Training Time: 33.51 seconds
#   - Prediction Time: 0.32 seconds
# Results for Isotonic Regression:
#   - Root Mean Squared Error (RMSE): 6.9436
#   - Training Time: 2.65 seconds
#   - Prediction Time: 0.14 seconds
#
# core number: 2
# w/ partition
# Results for Random Forest Regression:
#   - Root Mean Squared Error (RMSE): 19.9592
#   - Training Time: 14.86 seconds
#   - Prediction Time: 0.53 seconds
# Results for Gradient Boosted Tree Regression:
#   - Root Mean Squared Error (RMSE): 20.2046
#   - Training Time: 23.74 seconds
#   - Prediction Time: 0.47 seconds
# Results for Isotonic Regression:
#   - Root Mean Squared Error (RMSE): 6.9436
#   - Training Time: 0.87 seconds
#   - Prediction Time: 0.13 seconds
#
# w/o partition
# Results for Random Forest Regression:
#   - Root Mean Squared Error (RMSE): 19.1792
#   - Training Time: 14.65 seconds
#   - Prediction Time: 0.57 seconds
# Results for Gradient Boosted Tree Regression:
#   - Root Mean Squared Error (RMSE): 24.7308
#   - Training Time: 28.81 seconds
#   - Prediction Time: 0.34 seconds
# Results for Isotonic Regression:
#   - Root Mean Squared Error (RMSE): 6.9436
#   - Training Time: 1.73 seconds
#   - Prediction Time: 0.22 seconds
#
# core number: 3
# w/ partition
# Results for Random Forest Regression:
#   - Root Mean Squared Error (RMSE): 22.0976
#   - Training Time: 13.94 seconds
#   - Prediction Time: 0.60 seconds
# Results for Gradient Boosted Tree Regression:
#   - Root Mean Squared Error (RMSE): 22.6558
#   - Training Time: 22.36 seconds
#   - Prediction Time: 0.26 seconds
# Results for Isotonic Regression:
#   - Root Mean Squared Error (RMSE): 6.9436
#   - Training Time: 0.50 seconds
#   - Prediction Time: 0.05 seconds
#
# w/o partition
# Results for Random Forest Regression:
#   - Root Mean Squared Error (RMSE): 19.1792
#   - Training Time: 12.23 seconds
#   - Prediction Time: 0.62 seconds
# Results for Gradient Boosted Tree Regression:
#   - Root Mean Squared Error (RMSE): 24.7308
#   - Training Time: 25.41 seconds
#   - Prediction Time: 1.01 seconds
# Results for Isotonic Regression:
#   - Root Mean Squared Error (RMSE): 6.9436
#   - Training Time: 2.04 seconds
#   - Prediction Time: 0.09 seconds
#
# core number: 4
# w/ partition
# Results for Random Forest Regression:
#   - Root Mean Squared Error (RMSE): 21.8258
#   - Training Time: 13.90 seconds
#   - Prediction Time: 0.59 seconds
# Results for Gradient Boosted Tree Regression:
#   - Root Mean Squared Error (RMSE): 20.3722
#   - Training Time: 25.45 seconds
#   - Prediction Time: 0.51 seconds
# Results for Isotonic Regression:
#   - Root Mean Squared Error (RMSE): 6.9436
#   - Training Time: 0.59 seconds
#   - Prediction Time: 0.04 seconds
#
# w/o partition
# Results for Random Forest Regression:
#   - Root Mean Squared Error (RMSE): 19.1792
#   - Training Time: 14.38 seconds
#   - Prediction Time: 0.44 seconds
# Results for Gradient Boosted Tree Regression:
#   - Root Mean Squared Error (RMSE): 24.7308
#   - Training Time: 25.87 seconds
#   - Prediction Time: 0.32 seconds
# Results for Isotonic Regression:
#   - Root Mean Squared Error (RMSE): 6.9436
#   - Training Time: 1.09 seconds
#   - Prediction Time: 0.09 seconds
#
# core number: 5
# w/ partition
# Results for Random Forest Regression:
#   - Root Mean Squared Error (RMSE): 19.3510
#   - Training Time: 25.19 seconds
#   - Prediction Time: 0.77 seconds
# Results for Gradient Boosted Tree Regression:
#   - Root Mean Squared Error (RMSE): 21.0787
#   - Training Time: 32.94 seconds
#   - Prediction Time: 0.28 seconds
# Results for Isotonic Regression:
#   - Root Mean Squared Error (RMSE): 6.9436
#   - Training Time: 1.33 seconds
#   - Prediction Time: 0.11 seconds
#
# w/o partition
# Results for Random Forest Regression:
#   - Root Mean Squared Error (RMSE): 19.1041
#   - Training Time: 25.92 seconds
#   - Prediction Time: 0.63 seconds
# Results for Gradient Boosted Tree Regression:
#   - Root Mean Squared Error (RMSE): 24.0250
#   - Training Time: 29.48 seconds
#   - Prediction Time: 0.27 seconds
# Results for Isotonic Regression:
#   - Root Mean Squared Error (RMSE): 6.9436
#   - Training Time: 2.72 seconds
#   - Prediction Time: 0.17 seconds
