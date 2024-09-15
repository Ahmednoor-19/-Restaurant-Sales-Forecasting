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
```

### 2. Install Dependencies
Install the necessary Python packages using requirements.txt.

```bash
pip install -r requirements.txt
```

### 3. Run the Application Locally
Run the Flask application locally with the following command:

```bash
python main.py
```
The application will be accessible at http://127.0.0.1:5000.

### 4. Upload Data
Access the /upload route in the browser to upload your sales data in CSV format.

### 5. Train the Model
Navigate to the /train route to train the model on the uploaded data.

### 6. Forecast Sales
Go to the /forecast route to generate and view sales forecasts.

GCP Deployment
Steps to Deploy on Google App Engine
Create a GCP Project:

Visit the GCP Console and create a new project.
Enable App Engine:

Enable the App Engine service for your project.
Deploy the Application:

Use the following commands to deploy the app to App Engine:
``` bash
gcloud app deploy app.yaml
```
After deployment, the app will be available via the GCP App Engine URL.

## Libraries Used

- **Flask**: A web framework used to build the application.
- **FB Prophet**: A time series forecasting model utilized for generating predictions.
- **pandas**: Used for data processing and manipulation.
- **matplotlib**: Employed for generating sales forecast plots.
- **gunicorn**: An HTTP server used to run the application in a production environment.

## Usage

1. **Upload CSV Data**: Use the provided interface to upload your restaurant’s sales data in CSV format.
2. **Train the Model**: Train the forecasting model using the uploaded data by navigating to the `/train` route.
3. **View Forecast**: Generate and view sales forecasts, and visualize the results by visiting the `/forecast` route.

## Future Enhancements

- **More Models**: Integrate additional time series models like ARIMA or LSTM for performance comparison.
- **Enhanced UI**: Upgrade the user interface with more interactive data visualizations.
- **Better Forecast Accuracy**: Fine-tune model parameters to enhance prediction accuracy.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.


Contact
For any questions or suggestions, please feel free to reach out to:

Sheikh Ahmed Noor: sheikhahmednoor00@gmail.com
