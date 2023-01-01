import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence, pack_sequence
import matplotlib.pyplot as plt

from utils import *
from dataset import MyDataset
from config import *
from classifier import SentimentClassifier

if __name__ == '__main__':
    
    config = Config()
    
    print('Loading files ...')
    train_files = get_file("train")
    test_files = get_file("test")
    
    print("Building vocabulary ...")
    vocab = build_vocab(train_files, load_from_saved=True, min_freq = 10, max_tokens=config.VOC_SIZE)
    
    print("Setting up dataloaders ...")
    train_dataset = MyDataset(train_files, vocab)
    test_dataset = MyDataset(test_files, vocab)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE,shuffle=True,collate_fn=MyDataset.collate_fun,num_workers=0)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE,shuffle=True,collate_fn=MyDataset.collate_fun, num_workers=0)
    
    print("Setting up the model ...")
    torch.manual_seed(42)
    model = SentimentClassifier().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr= config.L_RATE)
    
    epoch = config.EPOCH
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("Starting epochs ...")
    for i in range(epoch):
        
        print("Current epoch [{}/{}]".format(i+1, epoch))
        
        model.train()
        
        train_batch_acc = []
        test_batch_acc = []
        train_batch_loss = []
        test_batch_loss  = []
        current_best_testAcc = -1
        

        for j, (features,labels) in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            
            optimizer.zero_grad()   
                
            features = features.cuda()     
            labels = labels.cuda()
            
            logits = model(features)
            loss = criterion(logits,labels)
            loss.backward()
            optimizer.step()
            
            acc = getAccuracy(logits,labels)
            
            train_batch_acc.append(acc)
            train_batch_loss.append(loss.item())
            
        train_avg_batch_acc = sum(train_batch_acc)/len(train_batch_acc)
        train_accuracies.append(train_avg_batch_acc)
        train_avg_batch_loss = sum(train_batch_loss)/len(train_batch_loss)
        train_losses.append(train_avg_batch_loss)
        print("epoch:{}/{},Train loss:{}, Training Accuracy:{}".format(i+1,epoch,train_avg_batch_loss,train_avg_batch_acc))
        
        
        model.eval()
        for j, (features,labels) in enumerate(test_dataloader):
            with torch.no_grad(): 
                features = features.cuda()
                labels = labels.cuda()
                
                logits = model(features)
                loss = criterion(logits,labels)
                
                test_batch_loss.append(loss.item())
                acc = getAccuracy(logits,labels)
                test_batch_acc.append(acc)
                
                if test_batch_acc[j] > current_best_testAcc:
                    torch.save(model.state_dict(), "saved/IMDBmodel_best.pth")
                    current_best_testAcc = test_batch_acc[j]
                
        test_avg_batch_acc= sum(test_batch_acc)/len(test_batch_acc)
        test_accuracies.append(test_avg_batch_acc)
        test_avg_batch_loss = sum(test_batch_loss)/len(test_batch_loss)
        test_losses.append(test_avg_batch_loss) 
        print("epoch:{}/{},Test loss:{}, Validation Accuracy:{}".format(i+1,epoch,test_avg_batch_loss,test_avg_batch_acc))
        
    
    plt.subplot(1,2,1)
    plt.plot(train_accuracies)
    plt.plot(test_accuracies)
    plt.title('Accuracies vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])

    plt.subplot(1,2,2)
    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.title('Losses vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    
    plt.savefig('saved/Plots.jpg')