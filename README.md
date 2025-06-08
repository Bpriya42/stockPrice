# stockPrice
Using PySpark to parallelise the ingestion of stock data for faster runtime and analysis

For each file, simply use python3 file_name to run.

make_dataframes.py contains the code for data preparation, so converting the text files from the kaggle dataset and transforming into the necessary spark dataframes for later portions of the project.

analysis.py contains code used to make the plots for the explatory data analysis portion of the project.

model.py contains code for our machine learning models using various cores (from 1 to 5) with and without data partition(train, prediction, and evaluation).
