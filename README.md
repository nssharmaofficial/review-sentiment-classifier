# Review_Classification_IMDB

Assignment for sentiment classification of the reviews from IMDB dataset on the Kaggle: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
 
## Classifier

The classifier ```SentimentClassifier() ``` consists of:
```python
self.emblayer = nn.Embedding(self.config.VOC_SIZE, self.config.EMBED_SIZE)
self.lstmlayer = nn.LSTM(self.config.EMBED_SIZE, self.config.HIDDEN_SIZE ,batch_first=True)
self.linear1 = nn.Linear(self.config.HIDDEN_SIZE, 32)
self.relu = nn.ReLU()
self.linear2 = nn.Linear(32, 2)
```
### Embedding layer

The ```nn.Embedding``` layer in PyTorch is a layer that maps discrete categorical variables to continuous vector representations, known as "embeddings". It is commonly used in natural language processing and other text-based tasks to represent words as dense, continuous vectors, which are easier to work with than the one-hot encoded representations that are commonly used as inputs to traditional neural networks.

This model takes in a batch of sequences of word indices of shape ```(batch_size, seq_length)``` and passes them through the embedding layer to produce a batch of sequences of word embeddings of shape ```(batch_size, seq_length, embedding_dim)```. 


### LSTM layer

Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) that is able to capture long-term dependencies in sequential data. It is called "long short-term memory" because it can remember information for long periods of time, but it is also able to forget irrelevant information and keep the most important information.

An LSTM layer ```nn.LSTM``` in PyTorch consists of four different "gates", which control the flow of information into and out of the memory cell:

- The "input gate" determines which parts of the input to store in the memory cell.
- The "forget gate" determines which parts of the previous state to forget.
- The "output gate" determines which parts of the previous state to output.
- The "memory cell" stores the relevant parts of the input and the previous state.

At each time step, the LSTM layer takes in an input x and the previous hidden state h and updates the hidden state based on these gates and the memory cell. The updated hidden state is then output and used as the input to the next time step.


<br>

## How to run this code

You'll need [Git](https://git-scm.com) to be installed on your computer.
```
# Clone this repository
$ git clone https://github.com/natasabrisudova/Review_Classification
```


To try the model run in command prompt:
```
$ predict.py ReviewToBeClassified
```

For example:

```
$ predict.py I love this movie

# Review: I love this movie 
# Classification: Positive
```





