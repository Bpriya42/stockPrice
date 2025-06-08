# Performs analysis on the dataset to gain insights and observe trends


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from make_dataframes import company_data


# This function analyses trends for opening, closing, high, and low prices over time for each company. 
# It creates a line plots for individual companies to observe patterns or anomalies in stock prices.
def plot_trends(df,company_name):
    # Filter data for a single company for plotting
    company_data = df[df['Company'] == company_name]

    # Plotting trends
    plt.figure(figsize=(20, 6))
    plt.plot(company_data['Date'], company_data['open_1'], label='Open Price')
    plt.plot(company_data['Date'], company_data['close_1'], label='Close Price')
    plt.plot(company_data['Date'], company_data['high_1'], label='High Price')
    plt.plot(company_data['Date'], company_data['low_1'], label='Low Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'Trend Analysis for {company_name}')
    plt.legend()
    plt.show()


#  This function calculates the daily range (high - low) for each stock and analyze its volatility. 
# It can be used to see which company has the most stable or volatile stock prices over the given period.
def plot_volitality(df,company_name):
    # Calculate daily volatility
    df['Volatility'] = df['high_1'] - df['low_1']

    # Plot volatility
    plt.figure(figsize=(20, 6))
    plt.plot(df['Date'], df['Volatility'], label='Volatility', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Volatility (High - Low)')
    plt.title(f'Volatility Analysis for {company_name}')
    plt.legend()
    plt.show()


# This function investigate the relationship between the traded volume 
# and stock price movements.
def plot_relation(df,company_name):
    # Scatter plot between volume and closing price
    plt.figure(figsize=(20, 6))
    plt.scatter(df['volume_1'], df['close_1'], alpha=0.5, c='blue')
    plt.xlabel('Volume')
    plt.ylabel('Closing Price')
    plt.title(f'Volume vs Closing Price for {company_name}')
    plt.show()

# This function computes percentage changes in opening and closing prices 
# day-to-day for each company.
def day2day(df,company_name):
    # Calculate percentage changes
    df['Open_Percent_Change'] = df['open_1'].pct_change() * 100
    df['Close_Percent_Change'] = df['close_1'].pct_change() * 100

    # Plot percentage changes
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Open_Percent_Change'], label='Open Price % Change', color='green')
    plt.plot(df['Date'], df['Close_Percent_Change'], label='Close Price % Change', color='red')
    plt.xlabel('Date')
    plt.ylabel('% Change')
    plt.title(f'Day-to-Day Percentage Change for {company_name}')
    plt.legend()
    plt.show()


# This function examine correlations between stock prices of different companies.
def corelation(df):
    # Pivot the dataframe so that each company's closing price is a column
    pivoted_data = df.pivot(index='Date', columns='Company', values='close_1')

    # Compute the correlation matrix
    correlation_matrix = pivoted_data.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Between Companies Based on Closing Prices')
    plt.show()



if __name__ == "__main__":
    partitioned_df = company_data()
    df  = partitioned_df.toPandas()
    # Company list
    companies=['Amazon','Apple','Google','Microsoft','Netflix']

    corelation(df)
    