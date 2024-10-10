# **Sentiment-Analysis-on-IMDB-Movie-Reviews-with-LSTM**

This project is a Deep Learning model designed to perform sentiment analysis on IMDB movie reviews using Long Short-Term Memory (LSTM) networks. Given a review, the model can classify the sentiment as positive or negative. The project uses the Keras library to train and build the LSTM model on the IMDB dataset and evaluates the model's performance on unseen data. This repository includes code to train the model, make predictions on new reviews, and visualize training history.

Project Overview
Sentiment analysis is a common task in natural language processing, often used to determine the overall emotional tone behind a body of text. In this project, we use LSTM networks, a type of Recurrent Neural Network (RNN), to classify movie reviews from the IMDB dataset into positive and negative sentiments.

Key Features:
Data Preprocessing: Loads and processes the IMDB dataset, converting text to sequences of integers.
LSTM Model Architecture: Embedding layer, LSTM layer, and Dense output layer.
Model Training and Evaluation: Model is trained on the dataset, with evaluation metrics like accuracy and loss.
Prediction Function: Custom function to predict sentiment for new, user-provided reviews.
Model Persistence: Save and load the trained model using the latest Keras format.
Dataset
The dataset used for this project is the IMDB dataset, which contains 25,000 highly polarized movie reviews for training and 25,000 reviews for testing. The dataset is preloaded in Keras, with reviews already tokenized and encoded as integers.

Installation and Setup
To get started, clone this repository and install the dependencies:

bash
Copy code
git clone https://github.com/your-username/sentiment-analysis-lstm.git
cd sentiment-analysis-lstm
Install the required packages:

bash
Copy code
pip install tensorflow matplotlib
You can run this code in a Jupyter Notebook. If you don’t have Jupyter installed, you can install it with the following command:

bash
Copy code
pip install jupyter
How to Run the Project
Open Jupyter Notebook:

bash
Copy code
jupyter notebook
Load the Notebook: In Jupyter Notebook, navigate to the project folder and open the notebook containing the code (e.g., Sentiment_Analysis_IMDB_LSTM.ipynb).

Run the Cells: Run the code cells sequentially:

Load and preprocess the data.
Build, compile, and train the model.
Evaluate the model.
Save and load the trained model.
Use the predict_sentiment function to make predictions on custom input reviews.
Example Usage
You can use the predict_sentiment() function to test the model with custom reviews. For example:

python
Copy code
# Sample review
predict_sentiment("This movie was absolutely amazing! I loved every second of it.")
Example Output
sql
Copy code
Review: This movie was absolutely amazing! I loved every second of it.
Predicted Sentiment: Positive with confidence: 0.87
Project Structure
bash
Copy code
sentiment-analysis-lstm/
│
├── Sentiment_Analysis_IMDB_LSTM.ipynb   # Jupyter Notebook with code
├── README.md                            # Project readme
└── requirements.txt                     # List of dependencies
Dependencies
TensorFlow
Keras
Matplotlib
Model Details
The model architecture consists of:

Embedding Layer: Converts integer sequences into dense vectors of fixed size.
LSTM Layer: A Long Short-Term Memory layer with 128 units.
Dense Layer: A single neuron with sigmoid activation for binary classification.
Hyperparameters
Vocabulary Size: 10,000 most common words
Maximum Sequence Length: 100
Batch Size: 64
Epochs: 5
Training History Visualization
After training, the notebook includes code to plot the model's accuracy and loss over each epoch, showing how well the model performs on both training and validation sets.

Saving and Loading the Model
The trained model can be saved and loaded in the latest Keras .keras format:

python
Copy code
# Save the model
model.save("sentiment_analysis_lstm.keras")

# Load the model
model = load_model("sentiment_analysis_lstm.keras")
License
This project is licensed under the MIT License - see the LICENSE file for details.
