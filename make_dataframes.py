from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf,  collect_list, lit, row_number
from pyspark.sql.types import IntegerType, StringType, DateType, DoubleType
from pyspark.sql.window import Window
import kagglehub


def company_data():
    spark = SparkSession.builder.appName("Main").config("spark.ui.showConsoleProgress", "false").getOrCreate()

    # # Download latest version
    path = kagglehub.dataset_download("borismarjanovic/price-volume-data-for-all-us-stocks-etfs")

    print("Path to dataset files:", path)

    all_dataframes = []

    microsoft_path = f"{path}/Stocks/msft.us.txt"
    google_path = f"{path}/Stocks/googl.us.txt"
    netflix_path = f"{path}/Stocks/nflx.us.txt"
    apple_path = f"{path}/Stocks/aapl.us.txt"
    amazon_path = f"{path}/Stocks/amzn.us.txt"

    microsoft_df = spark.read.option("delimiter", ",").csv(microsoft_path, header=True, inferSchema=True).withColumn("Company", lit("Microsoft")).drop("OpenInt")
    google_df = spark.read.option("delimiter", ",").csv(google_path, header=True, inferSchema=True).withColumn("Company", lit("Google"))
    netflix_df = spark.read.option("delimiter", ",").csv(netflix_path, header=True, inferSchema=True).withColumn("Company", lit("Netflix"))
    apple_df = spark.read.option("delimiter", ",").csv(apple_path, header=True, inferSchema=True).withColumn("Company", lit("Apple"))
    amazon_df = spark.read.option("delimiter", ",").csv(amazon_path, header=True, inferSchema=True).withColumn("Company", lit("Amazon"))

    all_dataframes = [microsoft_df, google_df, netflix_df, apple_df, amazon_df]
    joined_dataframe = None
    for company_dataframe in all_dataframes:

        window_spec = Window.orderBy("date").rowsBetween(-5, -1)

        company_dataframe = company_dataframe.withColumn("previous_5_open", collect_list("Open").over(window_spec))
        company_dataframe = company_dataframe.withColumn("previous_5_high", collect_list("High").over(window_spec))
        company_dataframe = company_dataframe.withColumn("previous_5_low", collect_list("Low").over(window_spec))
        company_dataframe = company_dataframe.withColumn("previous_5_close", collect_list("Close").over(window_spec))
        company_dataframe = company_dataframe.withColumn("previous_5_volume", collect_list("Volume").over(window_spec))
        #now we only need close for the current day, not using any current day data in predicting close of the current day
        company_dataframe = company_dataframe.drop("Open").drop("High").drop("Low").drop("Volume")

        #joined_df.filter(joined_df["Date"] == "2017-11-10").show(n=10)
        window_spec = Window.orderBy("date")

        # Add a row number column
        company_dataframe = company_dataframe.withColumn("row_number", row_number().over(window_spec))

        # Filter out rows with row_number <= 5
        company_dataframe = company_dataframe.filter("row_number > 5").drop("row_number")

        company_dataframe = company_dataframe.select(
            col("previous_5_open")[0].alias("open_1").cast(DoubleType()),
            col("previous_5_open")[1].alias("open_2").cast(DoubleType()),
            col("previous_5_open")[2].alias("open_3").cast(DoubleType()),
            col("previous_5_open")[3].alias("open_4").cast(DoubleType()),
            col("previous_5_open")[4].alias("open_5").cast(DoubleType()),

            col("previous_5_close")[0].alias("close_1").cast(DoubleType()),
            col("previous_5_close")[1].alias("close_2").cast(DoubleType()),
            col("previous_5_close")[2].alias("close_3").cast(DoubleType()),
            col("previous_5_close")[3].alias("close_4").cast(DoubleType()),
            col("previous_5_close")[4].alias("close_5").cast(DoubleType()),

            col("previous_5_high")[0].alias("high_1").cast(DoubleType()),
            col("previous_5_high")[1].alias("high_2").cast(DoubleType()),
            col("previous_5_high")[2].alias("high_3").cast(DoubleType()),
            col("previous_5_high")[3].alias("high_4").cast(DoubleType()),
            col("previous_5_high")[4].alias("high_5").cast(DoubleType()),

            col("previous_5_low")[0].alias("low_1").cast(DoubleType()),
            col("previous_5_low")[1].alias("low_2").cast(DoubleType()),
            col("previous_5_low")[2].alias("low_3").cast(DoubleType()),
            col("previous_5_low")[3].alias("low_4").cast(DoubleType()),
            col("previous_5_low")[4].alias("low_5").cast(DoubleType()),

            col("previous_5_volume")[0].alias("volume_1").cast(DoubleType()),
            col("previous_5_volume")[1].alias("volume_2").cast(DoubleType()),
            col("previous_5_volume")[2].alias("volume_3").cast(DoubleType()),
            col("previous_5_volume")[3].alias("volume_4").cast(DoubleType()),
            col("previous_5_volume")[4].alias("volume_5").cast(DoubleType()),

            col("Close").cast(DoubleType()),
            col("Date"),
            col("Company")
        )
        if not joined_dataframe:
            joined_dataframe = company_dataframe
        else:
            joined_dataframe = joined_dataframe.union(company_dataframe)
    #joined_dataframe.filter(joined_dataframe["Date"] == "2017-11-10").show(n=10)
    #joined_dataframe.show(n=10)
    #joined_dataframe.show(n=10, truncate=False)
    #partitioned_df = joined_dataframe.repartition(5, joined_dataframe["Company"])
    #partitioned_df.show()
    return joined_dataframe


def change_date(date):
    date_entries = date.split("-")
    return f"{date_entries[2]}-{date_entries[1]}-{date_entries[0]}"


#used because not worrying about junk after the decimal place
def remove_decimal(num):
    return num.split(".")[0]


if __name__ == "__main__":
    company_data()
