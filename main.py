import os
from flask import Flask, request, jsonify, send_file
from prophet import Prophet
from google.cloud import storage
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import io  

# Setup Google Cloud Storage client and bucket details
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_key.json'
storage_client = storage.Client()
bucket_name = 'restaurant-csv'
dataset_blob_name = 'restaurant_data(1)(1).csv'

def upload_to_bucket(blob_name, file_path, bucket_name, buffer=None):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if buffer:
            blob.upload_from_string(buffer.getvalue(), content_type='image/png')
        else:
            blob.upload_from_filename(file_path)
        print(f"File {file_path} uploaded to {blob_name}.")
        return True
    except Exception as e:
        print(e)
        return False

def download_from_bucket(blob_name, bucket_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        buffer = io.BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(e)
        return None

def upload_pickle_to_bucket(blob_name, data, bucket_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_string(pickle.dumps(data))
        print(f"Pickle object uploaded to {blob_name}.")
        return True
    except Exception as e:
        print(e)
        return False

def download_pickle_from_bucket(blob_name, bucket_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        pickle_data = blob.download_as_string()
        data = pickle.loads(pickle_data)
        print(f"Pickle object downloaded from {blob_name}.")
        return data
    except Exception as e:
        print(e)
        return None

app = Flask(__name__)

# Predefined column names
date_column = 'Date'
target_column = 'Day_Sale'
item_column = 'Item'

models = {}
dataframes = {}
accuracies = {}

@app.route('/train', methods=['POST'])
def train():
    try:
        restaurant_name = request.form['restaurant_name']

        if not restaurant_name:
            return jsonify({"error": "Missing required parameter: restaurant_name"}), 400

        # Download dataset from GCS
        buffer = download_from_bucket(dataset_blob_name, bucket_name)
        
        if buffer is None:
            return jsonify({"error": "Failed to download dataset from GCS"}), 500
        
        # Load dataset
        try:
            df = pd.read_csv(buffer, parse_dates=[date_column])
        except Exception as e:
            return jsonify({"error": f"Error reading the CSV file: {str(e)}"}), 500
        
        if df.empty:
            return jsonify({"error": "CSV file is empty"}), 500

        df.set_index(date_column, inplace=True)
        df.dropna(inplace=True)

        # Prepare the data for FB Prophet
        df_prophet = df.reset_index().rename(columns={date_column: 'ds', target_column: 'y'})

        # Split the data into train and test sets
        split_date = '2022-12-15'
        train = df_prophet[df_prophet['ds'] < split_date]
        test = df_prophet[df_prophet['ds'] >= split_date]

        # Initialize and fit the Prophet model
        model = Prophet()
        model.fit(train)

        # Save the model and dataframe
        models[restaurant_name] = {
            'model': model,
            'item_column': item_column,
            'date_column': date_column,
        }
        dataframes[restaurant_name] = df

        # Save the model object to GCS
        model_path = f'{restaurant_name}_model.pkl'
        upload_pickle_to_bucket(model_path, model, bucket_name)

        # Create a dataframe to hold future dates for predictions
        future = model.make_future_dataframe(periods=15)
        forecast = model.predict(future)

        # Generate and save plots
        # Plot the forecast
        fig, ax = plt.subplots(figsize=(10, 6))
        model.plot(forecast, ax=ax)
        plt.title(f'{restaurant_name} Sales Forecast')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        
        # Save to an in-memory buffer
        plot_buffer = io.BytesIO()
        fig.savefig(plot_buffer, format='png')
        plot_buffer.seek(0)
        plot_path = f'{restaurant_name}_forecast.png'
        upload_to_bucket(plot_path, None, bucket_name, buffer=plot_buffer)
        plt.close()

        # Plot the components
        fig_components = model.plot_components(forecast)
        
        # Save to an in-memory buffer
        components_buffer = io.BytesIO()
        fig_components.savefig(components_buffer, format='png')
        components_buffer.seek(0)
        components_image_path = f'{restaurant_name}_components_plot.png'
        upload_to_bucket(components_image_path, None, bucket_name, buffer=components_buffer)
        plt.close(fig_components)

        # Evaluate forecast accuracy
        forecast_sub = forecast[['ds', 'yhat']]
        forecast_sub['ds'] = forecast_sub['ds'].astype(str)
        test['ds'] = test['ds'].astype(str)
        eval_df = test.merge(forecast_sub, on=['ds'], how='left')
        eval_df['abs_error'] = abs(eval_df['y'] - eval_df['yhat'])
        eval_df['daily_FA'] = 1 - (eval_df['abs_error'] / eval_df['y'])

        total_y = eval_df['y'].sum()
        total_error = eval_df['abs_error'].sum()
        forecast_acc = 1 - (total_error / total_y)

        accuracies[restaurant_name] = forecast_acc

        return jsonify({
            "message": "Model trained successfully",
            "forecast_accuracy": forecast_acc,
            "forecast_plot": request.host_url + 'train/forecast_plot/' + restaurant_name,
            "components_plot": request.host_url + 'train/components_plot/' + restaurant_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/train/forecast_plot/<restaurant_name>', methods=['GET'])
def get_forecast_plot(restaurant_name):
    plot_path = f'{restaurant_name}_forecast.png'
    buffer = download_from_bucket(plot_path, bucket_name)
    if buffer is None:
        return jsonify({"error": "Forecast plot not found"}), 404
    buffer.seek(0)  # Ensure buffer is at the beginning
    return send_file(buffer, mimetype='image/png')

@app.route('/train/components_plot/<restaurant_name>', methods=['GET'])
def get_components_plot(restaurant_name):
    components_image_path = f'{restaurant_name}_components_plot.png'
    buffer = download_from_bucket(components_image_path, bucket_name)
    if buffer is None:
        return jsonify({"error": "Components plot not found"}), 404
    buffer.seek(0)  # Ensure buffer is at the beginning
    return send_file(buffer, mimetype='image/png')

@app.route('/train/forecast_accuracy/<restaurant_name>', methods=['GET'])
def get_forecast_accuracy(restaurant_name):
    if restaurant_name not in accuracies:
        return jsonify({"error": "Forecast accuracy not found"}), 404
    return jsonify({"forecast_accuracy": accuracies[restaurant_name]})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse request data
        restaurant_name = request.form['restaurant_name']
        item_code = request.form['item_code']
        future_periods = int(request.form['future_periods'])

        if restaurant_name not in models:
            # Reload model from GCS
            model_path = f'{restaurant_name}_model.pkl'
            model = download_pickle_from_bucket(model_path, bucket_name)
            if model is None:
                return jsonify({"error": "Restaurant not found"}), 404
            models[restaurant_name] = {
                'model': model,
                'item_column': item_column,
                'date_column': date_column,
            }

        model_info = models[restaurant_name]
        model = model_info['model']

        # Download dataset from GCS
        buffer = download_from_bucket(dataset_blob_name, bucket_name)
        if buffer is None:
            return jsonify({"error": "Failed to download dataset from GCS"}), 500

        # Reload dataset
        df = pd.read_csv(buffer, parse_dates=[date_column])

        # Convert item_code to the correct type (if needed)
        try:
            item_code = int(item_code)
        except ValueError:
            return jsonify({"error": "Invalid item code format"}), 400

        # Check if item_code exists in the dataset
        if item_code not in df[item_column].unique():
            return jsonify({"error": "Item code not found"}), 404

        # Filter the dataset for the specific item
        item_df = df[df[item_column] == item_code]

        if item_df.empty:
            return jsonify({"error": "Item code not found after filtering"}), 404

        # Prepare the data for FB Prophet
        df_prophet = item_df.rename(columns={date_column: 'ds', target_column: 'y'})

        # Create a dataframe to hold future dates for predictions
        future = model.make_future_dataframe(periods=future_periods)
        forecast = model.predict(future)

        # Separate actual data and predicted data
        actual_data = df_prophet.set_index('ds')['y']
        predicted_data = forecast.set_index('ds').yhat[-future_periods:]

        # Ensure the index is datetime
        actual_data.index = pd.to_datetime(actual_data.index)
        predicted_data.index = pd.to_datetime(predicted_data.index)

        # Get the last 30 days of actual data for plotting
        actual_data_last_30 = actual_data[-130:-30]

        # Plot the actual data for the last 30 days and the predicted data
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual_data_last_30.index, actual_data_last_30, label='Actual Day Sales (Last 30 Days)')
        ax.plot(predicted_data.index, predicted_data, label='Predicted Day Sales (Next 15 Days)', linestyle='dashed', color='green')
        ax.set_title(f'Day Sale Prediction using FB Prophet for Item {item_code}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Day Sale')
        ax.legend()

        # Save the plot to a buffer
        plot_buffer = io.BytesIO()
        fig.savefig(plot_buffer, format='png')
        plot_buffer.seek(0)  # Reset buffer position to start
        plt.close(fig)

        # Upload plot to GCS
        plot_path = f'{restaurant_name}_predict_item_{item_code}.png'
        upload_to_bucket(plot_path, None, bucket_name, buffer=plot_buffer)

        # Construct URL for the plot
        plot_url = request.host_url + 'predict_plot/' + restaurant_name + '/' + str(item_code)

        return jsonify({
            "message": "Prediction successful",
            "predict_plot_url": plot_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict/<restaurant_name>/<item_code>', methods=['GET'])
def get_predict_plot(restaurant_name, item_code):
    predict_plot_path = f'{restaurant_name}_predict_item_{item_code}.png'
    buffer = download_from_bucket(predict_plot_path, bucket_name)
    if buffer is None:
        return jsonify({"error": "Predict plot not found"}), 404
    buffer.seek(0)  # Ensure buffer is at the beginning
    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)