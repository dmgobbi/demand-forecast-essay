######################################################
### Step 1: Data Preparation with Outlier Handling ###
######################################################

import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter

# Load the dataset
file_path = "Real_Estate_Sales_2001-2022_GL.csv"
data = pd.read_csv(file_path, low_memory=False)

# Convert 'Date Recorded' to datetime
data['Date Recorded'] = pd.to_datetime(data['Date Recorded'])

# Extract year and month
data['Year'] = data['Date Recorded'].dt.year
data['Month'] = data['Date Recorded'].dt.month

# Exclude outlier years
data = data[~data['Year'].isin([2020, 2021])]

# Group data by year and month to calculate total sales
monthly_sales = data.groupby(['Year', 'Month'])['Sale Amount'].sum().reset_index()

# Ensure at least 60 data points
print("Total monthly data points:", len(monthly_sales))

#########################################
### Step 2: Split Data for Validation ###
#########################################

# Split the data into training and validation sets
train_data = monthly_sales[monthly_sales['Year'] < 2022]
test_data = monthly_sales[monthly_sales['Year'] == 2022]

###################################################
### Step 3: Apply and Validate Forecast Methods ###
###################################################

# Prepare training data
train_sales = train_data['Sale Amount']

# Apply Holt's or Winter's model
model = ExponentialSmoothing(train_sales, trend='add', seasonal='add', seasonal_periods=12).fit()

# Forecast the next 12 months
forecast = model.forecast(12)

# Validate against actual data
actual_sales = test_data['Sale Amount']
mape = mean_absolute_percentage_error(actual_sales, forecast)
print(f"MAPE: {mape:.2f}%")

# Formatter for y-axis
def billions_formatter(x, pos):
    return f'${x/1e9:.2f}B'

# Plot the results
# Create datetime index for visualization
monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(DAY=1))
train_data.loc[:, 'Date'] = pd.to_datetime(train_data[['Year', 'Month']].assign(DAY=1))
test_data.loc[:, 'Date'] = pd.to_datetime(test_data[['Year', 'Month']].assign(DAY=1))

# Plot with dates
plt.figure(figsize=(15, 7))
plt.plot(train_data['Date'], train_sales, label="Historical Sales (Training)", color='blue')
plt.plot(test_data['Date'], actual_sales, label="Actual Sales 2022", color='green')
plt.plot(test_data['Date'], forecast, label="Forecasted Sales 2022", color='red', linestyle="--")

plt.xlabel('Year')
from matplotlib.ticker import FuncFormatter
plt.gca().yaxis.set_major_formatter(FuncFormatter(billions_formatter))
plt.ylabel('Sale Amount (Billions of USD)')
plt.title('Real Estate Sales Forecast using Holt-Winters Method')
plt.legend()
plt.grid(True)

# Format x-axis to show year-month
plt.gca().xaxis.set_major_locator(YearLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

# Add text box with model information
plt.text(0.02, 0.95, 
         f'Forecast Method: Holt-Winters\n'
         f'Trend: Additive\n'
         f'Seasonal: Monthly (12 periods)\n'
         f'MAPE: {mape:.2f}%', 
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()
