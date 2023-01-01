import torch
import torch.nn as nn
import argparse

from utils import *
from config import *
from classifier import SentimentClassifier


def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('review', type=str, nargs='+',help = 'A review to classify')
    parsed_arguments = parser.parse_args()
    return parsed_arguments


def process(review, vocab):
    review_processed =  review.strip().lower().replace("<br />", " ").translate(str.maketrans('', '', string.punctuation)).split(" ")
    #print(review)
    #print(review_processed)
    #print(torch.LongTensor(vocab.vocab.lookup_indices(review_processed)))
    return torch.LongTensor(vocab.vocab.lookup_indices(review_processed))

def predict(model, review, vocab):
    review_processed = process(review, vocab)
    review_processed = review_processed.unsqueeze(0)
    model.eval()
    review_processed = review_processed.cuda()
    logits = model(review_processed)
    predictions = torch.argmax(logits,dim=1)
    prediction_output = predictions.item()
    if prediction_output == 0:
        result = 'Negative'
    else:
        result = 'Positive'
    return result


if __name__ == '__main__':
    
    args = parse_command_line_arguments()
    review = ' '.join([str(elem) for elem in args.review])
    
    config = Config()
    
    vocab = load_vocab()

    model = SentimentClassifier().cuda()
    model.load_state_dict(torch.load(f"saved/IMDBmodel_best.pth"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr= config.L_RATE)
    
    result = predict(model, review, vocab)
    print('Review: {} \nClassification: {}'.format(review, result))
    


