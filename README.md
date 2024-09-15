# Restaurant Sales Forecasting

This project aims to forecast daily sales for a restaurant using time series forecasting models. The solution uses **FB Prophet** and is deployed as a web application using **Flask**, and can be deployed on **Google Cloud Platform (GCP)**.

## Project Structure

### Files in the Repository

1. **main.py**
   - The main Flask application file that handles the routing, data upload, model training, and sales forecasting.
   
2. **app.yaml**
   - Configuration file for deploying the app on **Google App Engine**.

3. **requirements.txt**
   - Contains the list of Python libraries required to run the application.

4. **restaurant_data.csv**
   - Sample dataset that contains restaurant sales data.

### Dataset

The dataset (`restaurant_data.csv`) contains the following columns:

- **Date**: The date of the sales transaction.
- **Item**: The item code representing different menu items.
- **Price**: The price at which the item is sold.
- **Cost**: The cost to make the item.
- **Day_Sale**: The number of units sold on a given day.

You can upload your own dataset in CSV format when using the application, following the same structure.

## Features

1. **Data Upload**: Upload a CSV file with your restaurant sales data through the web interface.
   
2. **Model Training**: Train a time series model using **FB Prophet** to predict future sales trends.

3. **Sales Forecasting**: Generate future sales forecasts for each item in the dataset.
   
4. **Visualization**: Displays sales forecast plots directly in the application.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/restaurant-sales-forecasting.git
cd restaurant-sales-forecasting

## 2. Install Dependencies

Install the necessary Python packages using `requirements.txt`.

```bash
pip install -r requirements.txt
