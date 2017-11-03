DATA_FOLDER = './'

import timeit
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from scipy.sparse import csr_matrix, vstack

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

with open(DATA_FOLDER + 'featureset.pickle', 'rb') as f:
    featureset_sparse = pickle.load(f)
    while 1:
        try:
            featureset_sparse = vstack([featureset_sparse, pickle.load(f)])
        except EOFError:
            break

X = featureset_sparse
Y = pickle.load(open(DATA_FOLDER + 'labels_int_with_name.pickle', 'rb'))

X_train_np, X_dev_test_np, y_train_np, y_dev_test_np = train_test_split(X, Y, test_size=0.2, random_state=42)
X_dev_np, X_test_np, y_dev_np, y_test_np = train_test_split(X_dev_test_np, y_dev_test_np, test_size=0.5, random_state=42)

number_of_train_example, feature_length = X_train_np.shape
class_length =  len(np.unique(Y))
class_length = max(y_train_np) + 1

print(number_of_train_example, feature_length, class_length)


class SenderClassifierDataset(Dataset):
    """Sender Classifier Dataset"""
    
    def __init__(self, featureset_sparse, Y, transform=None):
        """
        Args:
            featureset_sparse (csr_matrix): csr_matrix of features
            Y (list of int): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.featureset_sparse = featureset_sparse
        self.Y = Y
        self.transform = transform
    
    def __len__(self):
        return self.featureset_sparse.shape[0]
    
    def __getitem__(self, idx):
        sample = {'feature': self.featureset_sparse[idx].toarray(), 
                  'label': self.Y[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        return {'feature': torch.from_numpy(np.squeeze(feature)).float(),
                'label': torch.LongTensor([label])}
                
                
dataset = SenderClassifierDataset(X_train_np, y_train_np,
                                 transform=transforms.Compose([
                                               ToTensor()
                                           ]))
dataloader = DataLoader(dataset=dataset,
                       batch_size=1024,
                       shuffle=True,
                       num_workers=4)

dataset_dev = SenderClassifierDataset(X_dev_np, y_dev_np,
                                 transform=transforms.Compose([
                                               ToTensor()
                                           ]))
dataloader_dev = DataLoader(dataset=dataset_dev,
                       batch_size=1024,
                       shuffle=False,
                       num_workers=4)
                       
#
# Define network
N_layer_1 = 2000

net = torch.nn.Sequential(
    torch.nn.Linear(feature_length, N_layer_1),
    torch.nn.ReLU(),
    torch.nn.Linear(N_layer_1, class_length),
    torch.nn.LogSoftmax()
)
print(net)

optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
loss_func = torch.nn.NLLLoss()

start_time = timeit.default_timer()
for epoch in range(50):
    for i_batch, sample_batched in enumerate(dataloader):
        feature = sample_batched['feature']
        label = sample_batched['label'].view(-1)
        
        # print('Epoch:', epoch, i_batch, feature.size(), label.size())
        
        x = Variable(feature)
        y = Variable(label)
        
        out = net(x)                 # input x and predict based on x
        loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted
        
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
        
        
        # log
        if i_batch % 10 == 0:
            
            # evaluate
            _, prediction = torch.max(out, 1)
            pred_y = prediction.data.numpy().squeeze()
            target_y = y.data.numpy()
            acc_train = sum(pred_y == target_y)/len(target_y)
            
            # evaluate on all dev data
            if i_batch > 99 and i_batch % 100 == 0:
                pred_y_dev_all = None
                for i_batch_dev, sample_batched_dev in enumerate(dataloader_dev):
                    feature_dev = sample_batched_dev['feature']

                    x_dev = Variable(feature_dev)
                    out_dev = net(x_dev)
                    _, prediction_dev = torch.max(out_dev, 1)
                    pred_y_dev = prediction_dev.data.numpy().squeeze()

                    if pred_y_dev_all is None:
                        pred_y_dev_all = pred_y_dev
                    else:
                        pred_y_dev_all = np.append(pred_y_dev_all, pred_y_dev)

                assert(len(pred_y_dev_all) == len(y_dev_np))
                acc_dev = sum(pred_y_dev_all == y_dev_np)/len(y_dev_np)


                print('Full Dev Eval -> Epoch{0:3d} Batch{1:4d}  -> Loss={2:.6f}, Train Acc={3:.3f}, Dev Acc={4:.3f}'.format(epoch+1, i_batch, loss.data[0], acc_train, acc_dev), flush=True)
            # print but not evaluate on all dev data
            else: 
                print('Epoch{0:3d} Batch{1:4d}  -> Loss={2:.6f}, Train Acc={3:.3f}, Dev Acc=None, Time passed: {4:.2f}Hours'
                      .format(epoch+1, i_batch, loss.data[0], acc_train, (timeit.default_timer() - start_time)/60/60), flush=True)
        