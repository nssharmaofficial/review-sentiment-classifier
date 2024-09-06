# Review sentiment classifier

The sentiment classifier uses a Long Short-Term Memory (LSTM) network to process sequences of word indices and determine the sentiment of a review. It includes modules for data processing, model training, and evaluation.

## Data

Reviews from [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/)

## Features

- **Sentiment classification**: Classifies reviews into positive or negative sentiment
- **Data handling**: Processes and loads movie review data
- **Vocabulary building**: Constructs and saves a vocabulary for mapping words to indices
- **Model training**: Trains an LSTM model for sentiment analysis
- **Evaluation**: Evaluates the model and plots training and validation losses and accuracies

## File structure

- `config.py`: Contains configuration settings for the model.
- `classifier.py`: Defines the `SentimentClassifier` model.
- `dataset.py`: Contains the `MyDataset` class for handling data.
- `utils.py`: Utility functions for data processing, vocabulary handling, and accuracy calculation.
- `train.py`: Script to train the model.
- `predict.py`: Script to make predictions on new reviews.
- `saved/`: Directory for saving models and vocabulary.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sentiment-classifier.git
    ```
2. Navigate into the project directory:
    ```bash
    cd sentiment-classifier
    ```
3. Install the required dependencies:
    ```bash
    pip install torch torchvision torchtext matplotlib
    ```

## Usage

### Training the Model

To train the model, run the following command:
```bash
python train.py
```

This will:
1. Load training and testing data.
2. Build or load the vocabulary.
3. Train the model and save the best-performing model.
4. Plot and save training and validation losses and accuracies.

### Making Predictions

To classify a review, use the `predict.py` script. Provide the review text as a command-line argument:

```bash
python predict.py "This movie was absolutely fantastic!"
```

This will output the classification of the review as either `Positive` or `Negative`.

## Configuration

Configuration settings are located in `config.py`. You can adjust parameters such as vocabulary size, embedding dimension, hidden size, and learning rate.

## Requirements

- Python 3.7+
- PyTorch
- TorchText
- Matplotlib


