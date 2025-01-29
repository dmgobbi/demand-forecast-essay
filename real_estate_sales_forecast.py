import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
from matplotlib.ticker import FuncFormatter

# Load the dataset
file_path = "Real_Estate_Sales_2001-2022_GL.csv"
data = pd.read_csv(file_path, low_memory=False)

# Convert 'Date Recorded' to datetime
data['Date Recorded'] = pd.to_datetime(data['Date Recorded'])

# Extract year and month
data['Year'] = data['Date Recorded'].dt.year
data['Month'] = data['Date Recorded'].dt.month

# Exclude outlier years
data = data[~data['Year'].isin([2008, 2020, 2021])]

# Exclude specific months
exclude_months = (data['Year'] == 2016) & (data['Month'] >= 10) | (data['Year'] == 2017) & (data['Month'] <= 1)
data = data[~exclude_months]

# Group data by year and month to calculate total sales
monthly_sales = data.groupby(['Year', 'Month'])['Sale Amount'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))

# Function to train and forecast for different time frames
def forecast_sales(train_end_year, forecast_years, image_name):
    train_data = monthly_sales[monthly_sales['Year'] <= train_end_year]
    test_data = monthly_sales[monthly_sales['Year'].between(train_end_year + 1, train_end_year + forecast_years)]
    
    train_sales = train_data['Sale Amount']
    model = ExponentialSmoothing(train_sales, trend='add', seasonal='add', seasonal_periods=12).fit()
    forecast = model.forecast(len(test_data))
    
    actual_sales = test_data['Sale Amount']
    mape = mean_absolute_percentage_error(actual_sales, forecast)
    print(f"MAPE for {image_name}: {mape:.2f}%")
    
    plt.figure(figsize=(15, 7))
    plt.plot(train_data['Date'], train_sales, label="Historical Sales (Training)", color='blue')
    plt.plot(test_data['Date'], actual_sales, label="Actual Sales", color='green')
    plt.plot(test_data['Date'], forecast, label="Forecasted Sales", color='red', linestyle="--")
    
    plt.xlabel('Year')
    plt.ylabel('Sale Amount (Billions of USD)')
    plt.title(f'Real Estate Sales Forecast ({image_name})')
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(YearLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x/1e9:.2f}B'))
    
    plt.text(0.02, 0.95, 
             f'Forecast Method: Holt-Winters\n'
             f'Trend: Additive\n'
             f'Seasonal: Monthly (12 periods)\n'
             f'MAPE: {mape:.2f}%', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(image_name)
    plt.show()

# Generate forecasts for short, medium, and long term
forecast_sales(2018, 1, 'short_term_forecast.png')
forecast_sales(2011, 3, 'medium_term_forecast.png')
forecast_sales(2015, 7, 'long_term_forecast.png')
