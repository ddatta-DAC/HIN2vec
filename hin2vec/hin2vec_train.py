import os
import numpy as np
import sys

sys.path.append('./..')
import argparse
from torch import LongTensor as LT

try:
    from .data import InputData
    from .hin2vec_model import Hin2Vec_model
except:
    from data import InputData
    from hin2vec_model import Hin2Vec_model

from torch.autograd import Variable
import torch
from tqdm import tqdm


class Hin2Vec():

    def __init__(self,
                 input_file_name,
                 output_file_name,
                 emb_dimension=32,
                 batch_size=12,
                 iteration=100000,
                 initial_lr=5e-4,
                 neg_sample_size=1):
        self.output_file_name = output_file_name
        self.neg_sample_size = neg_sample_size
        self.data = InputData(batch_size, input_file_name)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iteration = iteration
        self.initial_lr = initial_lr

        print(emb_dimension)
        self.Hin2Vec_model = Hin2Vec_model(
            self.data.attr_size,
            self.data.rel_size,
            self.emb_dimension
        )
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.Hin2Vec_model.cuda()
        self.optimizer = torch.optim.Adam(self.Hin2Vec_model.parameters(), lr=self.initial_lr)

        # self.optimizer = optim.SGD(self.Hin2Vec_model.parameters(), lr=self.initial_lr)

    def train(self):
        """Multiple training.
        Returns:
            None.
        """
        batch_count = self.iteration
        process_bar = tqdm(range(int(self.iteration)))

        for i in process_bar:
            e1, e2, rel_ids, ground_truth = self.data.generate_batch_sampling(self.neg_sample_size)

            e1 = Variable(LT(e1))
            e2 = Variable(LT(e2))
            rel_ids = Variable(LT(rel_ids))
            ground_truth = Variable(LT(ground_truth))

            if self.use_cuda:
                e1 = e1.cuda()
                e2 = e2.cuda()
                ground_truth = ground_truth.cuda()
                rel_ids = rel_ids.cuda()
            self.optimizer.zero_grad()
            loss = self.Hin2Vec_model.forward(e1, e2, rel_ids, ground_truth)
            loss.backward()
            self.optimizer.step()

            process_bar.set_description(
                "Loss: {:.4f}".format(loss.data.cpu().numpy())
            )

        # --- Save embedding ---- #
        self.Hin2Vec_model.save_embedding(
            self.output_file_name,
            use_cuda=self.use_cuda
        )


# ========================================= #
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', choices=['dblp'],
    default='dblp'
)
parser.add_argument(
    '--emd_dim',
    default=128
)
args = parser.parse_args()
# ========================================= #
# Input format:
# node_id_1, node_id_2, relation_type, 1 [ground truth]
# ========================================= #

input_file = None
DATA = args.data
emb_dimension = args.emd_dim
model_use_data_DIR = 'model_use_data'
batch_size = 64
num_epochs = 100000  # keeping
if not os.path.exists(model_use_data_DIR):
    os.mkdir(model_use_data_DIR)
model_use_data_DIR = os.path.join(model_use_data_DIR, DATA)
if not os.path.exists(model_use_data_DIR):
    os.mkdir(model_use_data_DIR)

if DATA == 'dblp':
    source_data_DIR = './../../dblp/processed_data/DBLP'
    input_file = os.path.join(source_data_DIR, 'hin2vec_dblp_input.txt')

output_file = os.path.join(model_use_data_DIR, 'output_emb.txt')
# ---------------------- #
h2v = Hin2Vec(
    input_file_name=input_file,
    output_file_name=output_file,
    emb_dimension=emb_dimension,
    batch_size=batch_size,
    iteration=num_epochs
)
h2v.train()
