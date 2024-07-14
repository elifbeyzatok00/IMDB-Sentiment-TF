## IMDB Sentiment Analysis with TensorFlow

This project demonstrates a series of experiments with different neural network architectures to perform sentiment analysis on the IMDB movie reviews dataset using TensorFlow. The aim is to classify movie reviews as positive or negative based on their text content.

### Requirements

- Python 3.8+
- TensorFlow 2.x
- TensorFlow Datasets
- NumPy

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/imdb-sentiment-analysis.git
   cd imdb-sentiment-analysis
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install tensorflow tensorflow-datasets numpy
   ```

### Data Preparation

The IMDB reviews dataset is loaded and split into training and test sets. The reviews are tokenized and padded to ensure consistent input lengths for the neural networks.

### Model Architectures

#### 1. SimpleRNN Model

```python
model = keras.models.Sequential([
    keras.layers.Embedding(10000, 64),
    keras.layers.SimpleRNN(64),
    keras.layers.Dense(1, activation='sigmoid')
])
```

#### 2. Improved RNN Model with Dropout and Regularization

```python
model = keras.models.Sequential([
    keras.layers.Embedding(10000, 32),
    keras.layers.Dropout(0.5),
    keras.layers.SimpleRNN(32, kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(1, activation='sigmoid')
])
```

#### 3. Bidirectional LSTM Model

```python
model = keras.models.Sequential([
    keras.layers.Embedding(10000, 32),
    Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=keras.regularizers.l2(0.001))),
    keras.layers.Dense(1, activation='sigmoid')
])
```

#### 4. CNN + Bi-LSTM Model

```python
model = keras.models.Sequential([
    keras.layers.Embedding(10000, 32),
    Conv1D(32, 7, activation='relu'),
    MaxPooling1D(5),
    Bidirectional(LSTM(32, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=keras.regularizers.l2(0.001))),
    Bidirectional(LSTM(32, dropout=0.5, recurrent_dropout=0.5, kernel_regularizer=keras.regularizers.l2(0.001))),
    keras.layers.Dense(1, activation='sigmoid')
])
```

### Training

Each model is compiled with binary cross-entropy loss and the Adam optimizer. Early stopping is used to prevent overfitting.

```python
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(train_padded, train_labels, epochs=10, validation_data=(test_padded, test_labels), callbacks=[early_stop])
```

### Results

The training and validation accuracy and loss are tracked for each model. The final results are saved to help identify the best-performing architecture.

### Conclusion

This project showcases different approaches to text classification using RNN and LSTM architectures, with enhancements like dropout, bidirectionality, and convolutional layers. The models demonstrate various performance levels, helping in understanding the strengths and weaknesses of each approach in the context of sentiment analysis.

### References

- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets)

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgements

Special thanks to the TensorFlow and Keras teams for their excellent libraries and documentation.
