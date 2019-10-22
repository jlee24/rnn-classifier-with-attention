## Recurrent neural network classifier with self-attention

A minimal RNN-based classification model (many-to-one) with self-attention.
Tested on `master` branches of both `torch` (commit 5edf6b2) and `torchtext` (commit c839a79). The `volatile` warnings that might be printed are due to using pytorch version 4 with torchtext.

Inspired by @Keon's [barebone seq2seq implementation](https://github.com/keon/seq2seq), this repository aims to provide a minimal implementation of an RNN classifier with self-attention.

#### Model description:
- LSTM or GRU encoder for the embedded input sequence
- [Scaled dot-product](https://arxiv.org/pdf/1706.03762.pdf) self-attention with the encoder outputs as keys and values and the hidden state as the query
- Logistic regression classifier on top of attention outputs

#### Arguments:

```
  --data DATA        Corpus: [SST, TREC, IMDB]
  --model MODEL      type of recurrent net [LSTM, GRU]
  --emsize EMSIZE    size of word embeddings [Uses pretrained on 50, 100, 200, 300]
  --hidden HIDDEN    number of hidden units for the RNN encoder
  --nlayers NLAYERS  number of layers of the RNN encoder
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch_size N     batch size
  --drop DROP        dropout
  --bi               bidirectional encoder
  --cuda             [DONT] use CUDA
  --fine             use fine grained labels in SST # currently unused
```

A sample set of arguments can be viewed in `run.sh`.

#### Results 

Accuracy on test set after 5 epochs of the model with sample params:

|               |    SST    |    TREC   |    IMDB   |
| ------------- |:---------:|:---------:|:---------:|
| `run.sh`      |  80.340%  |  87.000%  |  86.240%  |
=======
# rnn-classifier-with-attention
An RNN classifier with attention that takes responses from people as input and outputs what kind of question would have prompted a given response.

<b>Problem:</b> In an ideal world, we would want to train a Seq2Seq model from noisy human responses to clean answers. However, we do not have sufficient training data. We came up with a semi-supervised approach to this task, instead training an RNN for a classification task. The model input is the noisy response, and the output is the question type that elicited the response, where types are curated by clustering the questions.

<b>Goal:</b> Our objective is to gather visual question answering data, so we also add an attention layer with the hopes of identifying what words in the response are salient in the prediction. Using only the most salient words as part of a final, clean answer, we evaluate performance by comparing these predicted answers with those provided by HITs.

<b>Functionality:</b>
- Train your own classifier and run inferennce 
- Visualize attentions over responses
- Update training data as new HITs roll in

# How to train/run the classifier
1. Aggregate output from HITs
We have been launching HITs in which people are coming up with clean answers based on an image, question, and the noisy human response. Since we will be using their answers in evaluation, we will stick to training on the question/noisy response pairs in the HIT results. Specify source directory where HIT results are located and destination location for output, which is a dataset in .tsv form.

`python build_dataset.py --hits_src ./results/ --dataset_dest ./data/all.tsv`

2. Train 
Make sure to have `revtok`, `spacy`, and most importantly, `TorchText` installed based on this great tutorial by Allen Nie <a src="http://anie.me/On-Torchtext/">here</a>. This will train and save the model to 

`python train.py --dataset_src ./data/all.tsv --model_dest ./model`

3. Evaluate 
We're still adding the ability to choose a particular evaluation metric, but for now it performs BLEU. It also saves all the responses and their attentions to be visualized later.

`python evaluate.py --model_src ./model --attn_dest ./results_with_attentions.json`

# Visualize the attentions
Fire up the `Visualize attentions.ipynb` notebook, which will plot the attentions over the response words from `results_with_attentions.json`.

# Ongoing next steps
- Additional evaluation metrics (CIDER, METEOr)
- Training model bidirectionally
