# fisrt install requierments using these:
# pip3 install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
# pip3 install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
# pip3 install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
# pip3 install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.6.0.html
# pip3 install torch-geometric
# wget -c https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh
# chmod +x Miniconda3-py37_4.8.3-Linux-x86_64.sh
# time bash ./Miniconda3-py37_4.8.3-Linux-x86_64.sh -b -f -p /usr/local
# time conda install -q -y -c conda-forge rdkit
#
# sudo pip3 install networkx

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')
import os
import numpy as np
from math import sqrt
from scipy import stats

from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from gcn import GCNNet
from torch.utils.data import WeightedRandomSampler

import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, Draw
import networkx as nx
from rdkit import Chem
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

DATASET = 'davis'

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis',
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark DATASET, default = 'davis'
        self.dataset = dataset
        print(self.processed_paths[0])
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            # print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])



def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


# from DeepDTA data
all_prots = []

print('convert data from DeepDTA ')
fpath = sys.argv[1]  ##################### entery

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

compound_iso_smiles = []

df = pd.read_csv(fpath,sep = '\t', header=None ) 
df.head()
compound_iso_smiles += list(df[1])
print(compound_iso_smiles)

compound_iso_smiles = set(compound_iso_smiles)
smile_graph = {}

for smile in compound_iso_smiles:
    g = smile_to_graph(smile)
    smile_graph[smile] = g

# convert to PyTorch data format
processed_data_file_test = DATASET+'test_test.pt'
if ((not os.path.isfile(processed_data_file_test))):
    df = pd.read_csv(fpath,sep = '\t', header=None  )
    test_drugs, test_prots= list(df[1]), list(df[0])
    XT = [seq_cat(t) for t in test_prots]
    test_drugs, test_prots= np.asarray(test_drugs), np.asarray(XT)
    print(test_drugs,test_prots)
    test_Y = np.asarray([0,1])
    # make data PyTorch DATASET ready
    print('preparing ', DATASET + 'test_test.pt in pytorch format!')
    test_data = TestbedDataset(root='data', dataset=DATASET + 'test_test', xd=test_drugs, xt=test_prots, y=test_Y,
                                smile_graph=smile_graph)
    print( processed_data_file_test, ' have been created')
else:
    print(processed_data_file_test, ' are already created')



def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


modeling = GCNNet
model_st = modeling.__name__

cuda_name = "cuda:0"
print('cuda_name:', cuda_name)


TEST_BATCH_SIZE = 1
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
print('\nrunning on ', model_st + '_' + DATASET)
processed_data_file_test = 'data/processed/' + DATASET + 'test_test.pt'
if ((not os.path.isfile(processed_data_file_test))):
    print('please run create_data.py to prepare data in pytorch format!')
else:
    test_data = TestbedDataset(root='data', dataset=DATASET + 'test_test')

    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    model_file_name = '/content/drive/My Drive/Phase2_ML_for_Bio_Project/model_GCNNet_davis.model'
    result_file_name = sys.argv[2]  ######################output
    model.load_state_dict(torch.load(model_file_name))
    model.eval()
    print('predicting for test data')
    G, P = predicting(model, device, test_loader)
    with open(result_file_name,'w') as f:
        f.write('\t'.join(map(str,P)))

