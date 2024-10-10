# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Set parameters
vocab_size = 10000  # Only consider the top 10,000 words in the dataset
max_len = 100  # Pad sequences to a length of 100

# Load and preprocess the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Build the LSTM model
model = Sequential([
    Embedding(vocab_size, 100, input_length=max_len),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model on the test data
score = model.evaluate(x_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# Plot training & validation accuracy values
plt.figure(figsize=(14,5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.show()

# Save the model in the new Keras format
model.save("sentiment_analysis_lstm.keras")

# Load the model
model = load_model("sentiment_analysis_lstm.keras")

# Define a function to predict sentiment from new reviews
def predict_sentiment(review):
    # Convert text to sequence of integers
    tokenizer = imdb.get_word_index()
    words = review.lower().split()
    words = [tokenizer[word] if word in tokenizer else 0 for word in words]
    words_padded = pad_sequences([words], maxlen=max_len)
    
    # Predict sentiment
    prediction = model.predict(words_padded)
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    confidence = prediction[0][0] if prediction >= 0.5 else 1 - prediction[0][0]
    
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment} with confidence: {confidence:.2f}")
    print()

# Test the model with example reviews
reviews = [
    "I absolutely loved this movie! The acting was fantastic, and the story was heartwarming.",
    "This film was terrible. I hated every minute of it.",
    "The movie had some good parts, but overall it was just okay.",
    "A masterpiece! One of the best movies I've seen in a long time.",
    "I wouldn't recommend this movie to anyone. It was really boring and predictable."
]

for review in reviews:
    predict_sentiment(review)

####################
# Additional example reviews
additional_reviews = [
    "The plot was very confusing, and the actors seemed uninterested. I wouldn’t watch it again.",
    "Absolutely brilliant! The visuals were stunning, and the storyline was deeply moving.",
    "It was a complete waste of time. I regret watching this movie.",
    "The characters were well-developed, and I loved the unexpected twists in the story.",
    "Not my cup of tea. The pacing was slow, and I found it hard to stay engaged.",
    "A fantastic movie with top-notch acting and a great script!",
    "The storyline was predictable, and it felt like I've seen this type of movie a hundred times before.",
    "It was an enjoyable watch. The soundtrack was amazing, and I left the theater feeling happy.",
    "One of the worst movies I have ever seen. Don't waste your money on this one.",
    "The cinematography was beautiful, but the story didn’t make much sense.",
]

# Predict the sentiment for each additional review
for review in additional_reviews:
    predict_sentiment(review)

