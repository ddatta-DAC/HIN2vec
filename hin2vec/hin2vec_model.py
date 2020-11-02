import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pickle
from torch import LongTensor as LT
from torch import FloatTensor as FT

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPSILON = 1e-16

class Hin2Vec_model(nn.Module):
    def __init__(self,
                 input_data_size,
                 relation_size,
                 embedding_dim
                 ):

        super(Hin2Vec_model, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding_size = input_data_size
        self.relation_size = relation_size
        self.embeddings = nn.Embedding(self.embedding_size, self.embedding_dim)
        self.relation_embedding = nn.Embedding(self.relation_size, self.embedding_dim)
        self.init_emb()

    def init_emb(self):
        self.embeddings.weight.data.normal_(0, 1.0 / math.sqrt(self.embedding_size))
        self.relation_embedding.weight.data.normal_(0, 1.0 / math.sqrt(self.embedding_size))
        return

    def forward(self, attr1, attr2, rel, ground_truth):
        global EPSILON

        emb_attr1 = self.embeddings(attr1)
        emb_attr2 = self.embeddings(attr2)
        emb_rel = self.relation_embedding(rel)
        sig_encoded_rel = torch.sigmoid(emb_rel)
        pred = torch.sigmoid(torch.sum(emb_attr1 * emb_attr2 * sig_encoded_rel, dim=1))
        loss = -1 * (ground_truth * torch.log(pred + eps) + (1 - ground_truth) * torch.log(1 - pred + eps))
        loss1 = torch.sum(loss)

        return loss1

    def save_embedding(self, file_name, use_cuda):
        if use_cuda:
            embedding = self.embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.embeddings.weight.data.numpy()
        # Save as output as numpy array
        arr = [embedding]

        if use_cuda:
            embedding = self.relation_embedding.weight.cpu().data.numpy()
        else:
            embedding = self.relation_embedding.weight.data.numpy()
        arr.append(embedding)
        arr = np.array(arr)
        np.save(file_name, arr)
        return



