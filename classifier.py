import torch
import torch.nn as nn

from config import *

class SentimentClassifier(nn.Module):
    """
    This model takes in a sequence of word indices and passes them through an embedding layer to 
    produce a sequence of word embeddings (of dimension EMBED_SIZE). These word embeddings are then
    passed through the LSTM layer and the fully connected layer.
    The VOC_SIZE specifies the size of the vocabulary (i.e the number of most frequent unique words in the
    dataset)
    """

    def __init__(self) -> None:
        super(SentimentClassifier,self).__init__()
        self.config = Config()
        
        self.emblayer = nn.Embedding(self.config.VOC_SIZE,self.config.EMBED_SIZE)
        self.lstmlayer = nn.LSTM(self.config.EMBED_SIZE, self.config.HIDDEN_SIZE ,batch_first=True)
        self.linear1 = nn.Linear(self.config.HIDDEN_SIZE, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 2)

    def forward(self, x):
        
        # Embed the input data
        x = self.emblayer(x)
        
        # Set initial hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.config.HIDDEN_SIZE).cuda()
        c0 = torch.zeros(1, x.size(0), self.config.HIDDEN_SIZE).cuda()
        
        # Forward propagate LSTM
        x, _ = self.lstmlayer(x, (h0, c0)) 
        
        # x: tensor of shape (batch_size, seq_length, hidden_dim)
        # h: tensor of shape (batch_size, hidden_dim)
        # c: tensor of shape (batch_size, hidden_dim)s
        
        # Decode the hidden state of the last time step 
        x = x[:,-1,:]
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
    

