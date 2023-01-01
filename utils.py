import os
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
import io
import string
from torch.nn.utils.rnn import pad_sequence, pack_sequence
import pickle

def get_file(dir):
  paths_list = [] 
  for par, dirs, files in os.walk("Imdb/" + dir):
    if "pos" in par or "neg" in par:
      paths_list.extend([par + "/" + f for f in files])
  return paths_list

def yield_tokens(file_paths):
  for file_path in file_paths:
    with io.open(file_path, encoding = 'utf-8') as f:
      yield f.read().strip().lower().replace("<br />", " ").translate(str.maketrans('', '', string.punctuation)).split(" ")
      
def save_vocab(vocab):
    with open("saved/vocabulary.pkl","wb") as file:
        pickle.dump(vocab, file)
        
def load_vocab():
    with open("saved/vocabulary.pkl", "rb") as file:
        vocab = pickle.load(file)
    return vocab
        
def build_vocab(files, load_from_saved = True, min_freq = 10, max_tokens = 20000):
    if load_from_saved:
        try:
            vocab = load_vocab()
        except:
            print('You dont have any saved vocabulary')
    else:
        vocab = build_vocab_from_iterator(yield_tokens(files), min_freq = min_freq, max_tokens = max_tokens)
        vocab.set_default_index(0)
        save_vocab(vocab)
    return vocab

def getAccuracy(logits,labels):
    predictions = torch.argmax(logits,dim=1)
    acc = torch.sum(predictions == labels)/predictions.shape[0]
    return acc.item()
    
