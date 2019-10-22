import argparse
import os, sys
import time
import string
import pprint
import json

import torch
import torch.nn as nn
from model import *
from utils import *

import nltk
import spacy
# python3 -m spacy download en
spacy_en = spacy.load('en')

import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt

def make_parser():
  parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
  parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
  parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings [Uses pretrained on 50, 100, 200, 300]')
  parser.add_argument('--hidden', type=int, default=300,
                        help='number of hidden units for the RNN encoder')
  parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers of the RNN encoder')
  parser.add_argument('--lr', type=float, default=5e-4,
                        help='initial learning rate')
  parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
  parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
  parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
  parser.add_argument('--drop', type=float, default=0,
                        help='dropout')
  parser.add_argument('--bi', action='store_true',
                        help='[USE] bidirectional encoder')
  parser.add_argument('--cuda', action='store_true',
                    help='[DONT] use CUDA')
  parser.add_argument('--fine', action='store_true', 
                    help='use fine grained labels in SST')
  return parser

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def showAttention(input_sentence, prediction, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(np.array(attentions).reshape(1,len(input_sentence)), vmin=0)        
    fig.colorbar(cax)
   
    # Set up axes
    ax.set_xticklabels([''] + input_sentence + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + ["cluster " + str(prediction)])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def update_stats(accuracy, confusion_matrix, logits, y):
    _, max_ind = torch.max(logits, 1)
    equal = torch.eq(max_ind, y)
    correct = int(torch.sum(equal))

    for j, i in zip(max_ind, y):
        confusion_matrix[int(i),int(j)]+=1

    return accuracy + correct, confusion_matrix

def train(model, data, optimizer, criterion, args):
    model.train()
    accuracy, confusion_matrix = 0, np.zeros((args.nlabels, args.nlabels), dtype=int)
    t = time.time()
    total_loss = 0
    for batch_num, batch in enumerate(data):
        model.zero_grad()
        x = batch.Text
        y = batch.Label

        logits, _ = model(x)
        loss = criterion(logits.view(-1, args.nlabels), y)
        total_loss += float(loss)
        accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), True)
        optimizer.step()

        print("[Batch]: {}/{} in {:.5f} seconds".format(
              batch_num, len(data), time.time() - t), end='\r', flush=True)
        t = time.time()

    print()
    print("[Loss]: {:.5f}".format(total_loss / len(data)))
    print("[Accuracy]: {}/{} : {:.3f}%".format(
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
#   print(confusion_matrix)
    return total_loss / len(data)

def evaluate(model, data, optimizer, criterion, mode='Valid', resp_list=None, attentions_list=None, prediction_list=None, ids_list=None):
  model.eval()
  accuracy, confusion_matrix = 0, np.zeros((10, 10), dtype=int)
  t = time.time()
  total_loss = 0

  with torch.no_grad():
    for batch_num, batch in enumerate(data):
      x = batch.Text
      reversed_x = TEXT.reverse(x)
      y = batch.Label
      ids = POST_ID.reverse(batch.Post_Id, mode='id')

      logits, attentions = model(x)
      reversed_examples = [reversed_x[i:i+x.shape[0]] for i in range(0, len(reversed_x), x.shape[0])]
    
      if mode == 'Test':
          for i,ex in enumerate(reversed_examples):
            if '<pad>' in ex:
                pad_start = ex.index('<pad>')
            else:
                pad_start = 0
            resp_list.append(ex[:pad_start])
            attentions_list.append(attentions.cpu().numpy()[i][0][:pad_start].reshape(1,pad_start))
            prediction_list.append(np.argmax(logits[i]))
            ids_list.append(ids[i])
    
      total_loss += float(criterion(logits.view(-1, 10), y))
      accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
      print("[Batch]: {}/{} in {:.5f} seconds".format(
            batch_num, len(data), time.time() - t), end='\r', flush=True)
      t = time.time()

  print("")
  print("[{} loss]: {:.5f}".format(type, total_loss / len(data)))
  print("[{} accuracy]: {}/{} : {:.3f}%".format(type,
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))

  return total_loss / len(data)
    
def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]
    
def main():
    args = make_parser().parse_args()
    print("[Model hyperparams]: {}".format(str(args)))
    
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")

    TEXT = SplitReversibleField(sequential=True,  tokenize=tokenizer)
    LABEL = data.Field(sequential=False, use_vocab=False)
    QUESTION = SplitReversibleField(sequential=True,  tokenize=tokenizer)
    POST_ID = SplitReversibleField(sequential=False)

    train_data, val_data = data.TabularDataset.splits(
      path='./data/', train='train_real.tsv',
      validation='val_real.tsv', format='csv', skip_header=True,
      fields=[('Text', TEXT), ('Label', LABEL), ('Post_Id', POST_ID)])

    all_data = data.TabularDataset('./data/all.tsv',format='csv',skip_header=True,
                                     fields=[('Text', TEXT), ('Label', LABEL), ('Post_Id', POST_ID)])

    TEXT.build_vocab(train_data, vectors="glove.6B.200d")
    POST_ID.build_vocab(all_data)

    #  sort_key=lambda x: len(x.Text)
    val_iter = BucketIterator(val_data, 64, device=device, sort_within_batch=False,repeat=False)
    train_iter = BucketIterator(train_data, 64, device=device, sort_within_batch=False, repeat=False)
    
    ntokens, nlabels = len(TEXT.vocab), 10
    args.nlabels = nlabels # hack to not clutter function arguments

    embedding = nn.Embedding(ntokens, args.emsize, padding_idx=1, max_norm=1)
    # if vectors: embedding.weight.data.copy_(TEXT.vocab.vectors)
    encoder = Encoder(args.emsize, args.hidden, nlayers=args.nlayers)

    attention_dim = args.hidden if not args.bi else 2*args.hidden*args.hidden
    attention = Attention(attention_dim, attention_dim, attention_dim)

    model = Classifier(embedding, encoder, attention, attention_dim, nlabels)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, amsgrad=True)

    try:
        best_valid_loss = None

        for epoch in range(1, args.epochs + 1):
          train(model, train_iter, optimizer, criterion, args)
          loss = evaluate(model, val_iter, optimizer, criterion, args)

          if not best_valid_loss or loss < best_valid_loss:
            best_valid_loss = loss
    except KeyboardInterrupt:
        print("[Ctrl+C] Training stopped!")       

    loss = evaluate(model, val_iter, optimizer, criterion, mode='Test', resp_list=resp_list, attentions_list=attentions_list, prediction_list=prediction_list, ids_list=ids_list)
    
    torch.save(model, 'model')
    
if __name__ == '__main__':
  main()    
    