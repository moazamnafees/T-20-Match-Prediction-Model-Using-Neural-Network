# T20-Match-Prediction-Model-Using-Neural-Network
This project is an advanced T20 Cricket Match Score Predictor designed using Neural Networks and Machine Learning techniques. It predicts the total score of a batting team based on several match-related features, providing real-time insights and aiding strategy development.

 ## Key Features:

Real-Time Score Prediction with interactive widgets

Deep Learning Model using TensorFlow/Keras

User-friendly Web Interface for quick predictions

Analysis of model performance with key evaluation metrics

## Tech Stack:

Languages: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, Keras, ipywidgets

Tools: Google Colab

## Project Structure:

Data Preprocessing:

Dropped irrelevant features (e.g., date, striker, non-striker)

Label Encoding for categorical data

Min-Max Scaling for numerical stability

## Model Development:

Input Layer: Features include venue, batting team, bowling team, batsman, and bowler

Hidden Layers: Multiple dense layers (512 → 256 → 128 → 64 → 32 neurons) with ReLU activation

Output Layer: Single neuron with linear activation for continuous score prediction

Loss Function: Huber Loss for handling outliers

Optimizer: Adam Optimizer

## Model Evaluation:

Metrics: Mean Absolute Error (MAE), Validation Loss

Visualization: Residuals Distribution, Predicted vs Actual Scores, Loss vs Epochs

Interactive Interface:

Dropdown widgets for selecting venue, batting team, bowling team, batsman, and bowler

Real-time score prediction with a simple button click

Run the Notebook:

GoogleColab

## How It Works:

Select the match conditions (venue, teams, players) using the dropdowns.

Click the "Predict Score" button.

The predicted score will be displayed based on the trained model.

## Model Performance:

MAE: Low error margins indicating high accuracy

Residual Distribution: Centered around zero, showing unbiased predictions

Loss vs Epochs: Smooth convergence without overfitting

## Challenges & Limitations:

Limited handling of unseen players or venues

Potential bias due to historical data dependency

Difficulty in extreme score predictions

## License:

This project is open-source and available for educational and non-commercial use.
