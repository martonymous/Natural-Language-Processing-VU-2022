"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an recurrent network that predicts the tags for each token in the sentence. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net, self).__init__()
        self.params = params

        # the embedding takes as input the vocab_size and the embedding_dim
        self.embedding1 = nn.Embedding(params.vocab_size, params.embedding_dim1)
        self.embedding2 = nn.Embedding(params.vocab_size, params.embedding_dim2)
        self.embedding3 = nn.Embedding(params.vocab_size, params.embedding_dim3)
        self.embedding4 = nn.Embedding(params.vocab_size, params.embedding_dim4)

        # "skip" layers. Actually, this is just a linear layer which changes dimension of the embedded inputs so that it
        #  can skip to fc layer (where dimensions need to add up)
        self.skip1 = nn.Linear(params.embedding_dim1, params.lstm_hidden_dim)
        self.skip2 = nn.Linear(params.embedding_dim2, params.lstm_hidden_dim)
        self.skip3 = nn.Linear(params.embedding_dim3, params.lstm_hidden_dim)
        self.skip4 = nn.Linear(params.embedding_dim4, params.lstm_hidden_dim)

        # the LSTM takes as input the size of its input (embedding_dim), its hidden size
        # for more details on how to use it, check out the documentation
        self.lstm1 = nn.LSTM(params.embedding_dim1,
                            params.lstm_hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(params.embedding_dim2,
                            params.lstm_hidden_dim, batch_first=True)
        self.lstm3 = nn.LSTM(params.embedding_dim3,
                            params.lstm_hidden_dim, batch_first=True)
        self.lstm4 = nn.LSTM(params.embedding_dim4,
                            params.lstm_hidden_dim, batch_first=True)

        # the fully connected layer transforms the output to give the final output layer
        self.fc = nn.Sequential(
            nn.Linear(params.lstm_hidden_dim, 25),

            # dropour layer to reset connections
            nn.Dropout(0.15),

            # non-linearity, alternative to ReLU
            nn.SiLU(),
            nn.Linear(25, self.params.number_of_tags)
        )


    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        #                                -> batch_size x seq_len
        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x seq_len x embedding_dim
        a = self.embedding1(s)
        b = self.embedding2(s)
        c = self.embedding3(s)
        d = self.embedding4(s)

        # create layers for residual connections
        a1 = a.view(-1, a.shape[2])
        b1 = b.view(-1, b.shape[2])
        c1 = c.view(-1, c.shape[2])
        d1 = d.view(-1, d.shape[2])

        a1 = self.skip1(a1)
        b1 = self.skip2(b1)
        c1 = self.skip3(c1)
        d1 = self.skip4(d1)

        # run the LSTM along the sentences of length seq_len
        # dim: batch_size x seq_len x lstm_hidden_dim
        a, _ = self.lstm1(a)
        b, _ = self.lstm2(b)
        c, _ = self.lstm3(c)
        d, _ = self.lstm4(d)

        # make the Variable contiguous in memory (a PyTorch artefact)
        a = a.contiguous()
        b = b.contiguous()
        c = c.contiguous()
        d = d.contiguous()

        # reshape the Variable so that each row contains one token
        # dim: batch_size*seq_len x lstm_hidden_dim
        a = a.view(-1, a.shape[2])
        b = b.view(-1, b.shape[2])
        c = c.view(-1, c.shape[2])
        d = d.view(-1, d.shape[2])

        s = (self.fc(a) + self.fc(b) + self.fc(c) + self.fc(d) + self.fc(a1) + self.fc(b1) + self.fc(c1) + self.fc(d1)) / 8  # dim: batch_size*seq_len x num_tags

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0).float()

    # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
    # number. This does not affect training, since we ignore the PADded tokens with the mask.
    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask))

    # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
    return -torch.sum(outputs[range(outputs.shape[0]), labels]*mask)/num_tokens


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels)/float(np.sum(mask))


def precision(outputs, labels):
    labels = labels.ravel()
    mask = (labels >= 0)        # removes padding
    outputs = np.argmax(outputs, axis=1)

    # calculate true positives and false positives
    tp = np.sum(np.logical_and(outputs == 1, labels == 1))
    fp = np.sum(np.logical_and(outputs == 1, labels == 0))

    if (tp+fp) != 0:
        return tp/(tp+fp)
    else:
        return np.nan


def recall(outputs, labels):
    labels = labels.ravel()
    mask = (labels >= 0)        # removes padding
    outputs = np.argmax(outputs, axis=1)

    # calculate true positives and false negatives
    tp = np.sum(np.logical_and(outputs == 1, labels == 1))
    fn = np.sum(np.logical_and(outputs == 0, labels == 1))

    if (tp+fn) != 0:
        return tp/(tp+fn)
    else:
        return np.nan


def f1(outputs, labels):
    rec = recall(outputs, labels)
    prec = precision(outputs, labels)

    if prec == np.nan or rec == np.nan or (prec+rec) == 0:
        return np.nan
    else:
        return 2*((prec*rec)/(prec+rec))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy':     accuracy,
    # 'precision':    precision,
    # 'recall':       recall,
    # 'F1-score':     f1,
# could add more metrics such as accuracy for each token type
}
