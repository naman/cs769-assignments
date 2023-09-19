import torch
import torch.nn as nn
import zipfile
import numpy as np
import io

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])

def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """

    # source: https://fasttext.cc/docs/en/english-vectors.html
    
    # Load embedding file
    # if emb_file.endswith('.zip'):
    #     archive = zipfile.ZipFile(emb_file, 'r')
    #     f = archive.open('cc.en.300.vec')
    # else:

    f = io.open(emb_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, f.readline().split())

    emb = np.zeros((len(vocab), emb_size))
    # emb = np.random.uniform(-0.08, 0.08, (len(vocab), emb_size))
    emb = {}

    # Read embedding file
    for line in f:
        tokens = line.rstrip().split(' ')
        # check if token in vocabulary
        if tokens[0] in vocab:
            # emb[vocab[tokens[0]]] = np.asarray(tokens[1:], dtype='float32')
            emb[vocab[tokens[0]]] = np.asarray(tokens[1:])
    return emb

class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.layers = nn.ModuleList()
        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size)




    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        for layer in self.layers:
            nn.init.uniform_(layer.weight, -0.08, 0.08)
            nn.init.zeros_(layer.bias)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        # load embedding from numpy.array
        x = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        # copy embedding weights from numpy to nn.embedding
        self.embedding.weight.data.copy_(torch.from_numpy(x))

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        for layer in self.layers:
            x = layer(x)
        return x
        
