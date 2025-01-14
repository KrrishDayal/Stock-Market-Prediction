# Stock Market Prediction

The Stock Market Prediction project leverages machine learning to forecast stock prices based on historical data. By utilizing advanced predictive models, the project aims to assist traders and investors in making informed decisions. This tool is designed to handle large datasets efficiently and provide meaningful insights into stock price movements.

## Features
* Data Preprocessing: Handles missing values, normalizes data, and splits it into training and testing sets.
* Machine Learning Models: Utilizes algorithms like Linear Regression, LSTM, and Random Forest for accurate predictions.
* Visualization: Graphs and charts for analyzing trends and model performance.

## Installation

Clone the repository:
```bing 
git clone https://github.com/username/Stock_Market_Prediction.git
```

Navigate to the project directory:
```bing
cd Stock_Market_Prediction
```

## Models Used
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Prediction Model</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
        }
        h1 {
            color: #333;
        }
        p {
            margin: 10px 0;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 4px;
        }
    </style>
</head>
<body>

    <h1>Stock Prediction Model</h1>

    <h2>Overview</h2>
    <p>This project involves building a stock price prediction model using machine learning techniques. The model is trained on historical stock data and predicts future stock prices based on various features.</p>

    <h2>Model Details</h2>
    <p>The model used in this project is a <strong>Random Forest Regressor</strong>, a popular ensemble learning method for regression tasks. This method leverages decision trees to make predictions, combining their results to improve accuracy.</p>

    <h3>Key Features</h3>
    <ul>
        <li>Features such as Relative Strength Index (RSI), MACD, Bollinger Width, etc.</li>
        <li>Pre-processing of data, including handling missing values and feature engineering.</li>
        <li>Evaluation using performance metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score.</li>
    </ul>

    <h2>Code Example</h2>
    <pre>
        <code>
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Example of training the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Example of evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
        </code>
    </pre>

    <h2>Conclusion</h2>
    <p>This stock prediction model has demonstrated good performance with a focus on handling financial data. Future improvements could include refining features, using hyperparameter tuning, and experimenting with other machine learning models.</p>

</body>
</html>

## Contributors
- [Sainy-Mishra](https://github.com/Sainy-Mishra)
- [Krrish-Dayal](https://github.com/KrrishDayal)
  
## License
This project is licensed under the MIT License.
