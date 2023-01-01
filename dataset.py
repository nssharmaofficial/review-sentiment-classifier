import os
import torch
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, pack_sequence
import io
import string
import pickle

class MyDataset(Dataset):

  def __init__(self, files, vocab) -> None:
    self.files = files
    self.vocab = vocab


  def __len__(self):
    return len(self.files)


  def __getitem__(self, index):
    path = self.files[index]
    label = 1 if "pos" in path else 0
    with io.open(path, encoding = 'utf-8') as f:
      data =  f.read().strip().lower().replace("<br />", " ").translate(str.maketrans('', '', string.punctuation)).split(" ")
    return torch.LongTensor(self.vocab.vocab.lookup_indices(data)), label
  
 
  @staticmethod
  def collate_fun(batch):
    X = [x for x,_ in batch]
    y = [y for _,y in batch]
    X = pad_sequence(X,batch_first=True)
    return X,torch.LongTensor(y)

