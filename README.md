# rnn-classifier-with-attention
An RNN classifier with attention that takes responses from people as input and outputs what kind of question would have prompted that response. This project shows a semi-supervised approach to training an RNN classifier. There were no ground truth responses available for the responses, so we clustered questions and decided to use them as soft labels, loosely interpreted as question type. Our overall goal was to gather visual question answering data, so we also added an attention layer to our model with the hopes of identifying what words in the response are salient in the prediction. Using only the most salient to be used as part of a final, clean answer, we evaluate performance by comparing these harvested answers with those provided through HITs.

# How to train/run the classifier

# Ongoing next steps
- Additional evaluation metrics
- Training model bidirectionally
