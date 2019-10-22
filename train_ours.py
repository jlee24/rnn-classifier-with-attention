import argparse
import os, sys
import time
import numpy as np
import torch
import torch.nn as nn

import spacy
spacy_en = spacy.load('en')

from datasets import dataset_map
from model import *
from torchtext.vocab import GloVe
from torchtext import data
from torchtext.data import Field
from torchtext.data import Iterator, BucketIterator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_parser():
  parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
  parser.add_argument('--data', type=str, default='SST',
                        help='Data corpus: [SST, TREC, IMDB]')
  parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net [LSTM, GRU]')
  parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings [Uses pretrained on 50, 100, 200, 300]')
  parser.add_argument('--hidden', type=int, default=500,
                        help='number of hidden units for the RNN encoder')
  parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers of the RNN encoder')
  parser.add_argument('--lr', type=float, default=1e-3,
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
  parser.add_argument('--cuda', action='store_false',
                    help='[DONT] use CUDA')
  parser.add_argument('--fine', action='store_true', 
                    help='use fine grained labels in SST')
  return parser


def seed_everything(seed, cuda=False):
  # Set the random seed manually for reproducibility.
  np.random.seed(seed)
  torch.manual_seed(seed)
  if cuda:
    torch.cuda.manual_seed_all(seed)


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
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
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


def evaluate(model, data, optimizer, criterion, args, type='Valid'):
  model.eval()
  accuracy, confusion_matrix = 0, np.zeros((args.nlabels, args.nlabels), dtype=int)
  t = time.time()
  total_loss = 0
#   print(data.data())
  with torch.no_grad():
    for batch_num, batch in enumerate(data):
      x = batch.Text
      y = batch.Label

      logits, _ = model(x)
      total_loss += float(criterion(logits.view(-1, args.nlabels), y))
      accuracy, confusion_matrix = update_stats(accuracy, confusion_matrix, logits, y)
      print("[Batch]: {}/{} in {:.5f} seconds".format(
            batch_num, len(data), time.time() - t), end='\r', flush=True)
      t = time.time()

  print()
  print("[{} loss]: {:.5f}".format(type, total_loss / len(data)))
  print("[{} accuracy]: {}/{} : {:.3f}%".format(type,
        accuracy, len(data.dataset), accuracy / len(data.dataset) * 100))
#   print(confusion_matrix)
  return total_loss / len(data)

pretrained_GloVe_sizes = [50, 100, 200, 300]

def load_pretrained_vectors(dim):
  if dim in pretrained_GloVe_sizes:
    # Check torchtext.datasets.vocab line #383
    # for other pretrained vectors. 6B used here
    # for simplicity
    name = 'glove.{}.{}d'.format('6B', str(dim))
    return name
  return None

# def tokenizer(text): # create a tokenizer function
#     return [tok.text for tok in spacy_en.tokenizer(text)]

def main():
  args = make_parser().parse_args()
  print("[Model hyperparams]: {}".format(str(args)))

  cuda = torch.cuda.is_available() and args.cuda
  device = torch.device("cpu") if not cuda else torch.device("cuda:0")
  seed_everything(seed=1337, cuda=cuda)
  vectors = load_pretrained_vectors(args.emsize)

#   tokenize = lambda x: x.split()
  TEXT = data.Field(sequential=True)
  LABEL = data.Field(sequential=False, use_vocab=False)
    
  train_data, val_data = data.TabularDataset.splits(
      path='./data/', train='train.tsv',
      validation='val.tsv', format='csv', skip_header=True,
      fields=[('Text', TEXT), ('Label', LABEL)])

  TEXT.build_vocab(train_data, vectors="glove.6B.200d")
    
#   train_iter, val_iter = data.Iterator.splits(
#         (train, val), sort_key=lambda x: len(x.Text),
#         batch_sizes=(32, 256, 256), device=device)

#   train_iter, val_iter = BucketIterator.splits(
#         (train_data, val_data), # we pass in the datasets we want the iterator to draw data from
#         batch_sizes=(64, 64),
#         device=device, # if you want to use the GPU, specify the GPU number here
#         sort_key=lambda x: len(x.Text), # the BucketIterator needs to be told what function it should use to group the data.
#         sort_within_batch=False,
#         repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
#   )

  train_iter = BucketIterator(train_data, 64, sort_key=lambda x: len(x.Text), device=device, sort_within_batch=False, repeat=False)
  val_iter = BucketIterator(val_data, 64, sort_key=lambda x: len(x.Text), device=device, sort_within_batch=False,repeat=False)

  print("[Corpus]: train: {}, test: {}, vocab: {}, labels: {}".format(
            len(train_iter.dataset), len(val_iter.dataset), len(TEXT.vocab), 10))

  ntokens, nlabels = len(TEXT.vocab), 10
  args.nlabels = nlabels # hack to not clutter function arguments

  embedding = nn.Embedding(ntokens, args.emsize, padding_idx=1, max_norm=1)
  if vectors: embedding.weight.data.copy_(TEXT.vocab.vectors)
  encoder = Encoder(args.emsize, args.hidden, nlayers=args.nlayers)

  attention_dim = args.hidden if not args.bi else 2*args.hidden
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
  loss = evaluate(model, val_iter, optimizer, criterion, args, type='Test')
  return model

if __name__ == '__main__':
  model = main()
  torch.save(model, 'model')

  