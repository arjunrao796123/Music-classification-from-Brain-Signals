import os
import random

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from tqdm import *
import h5py

from model import *
DATA_PATH = "data/OpenMIIR-Perception-512Hz.hdf5"
lr = 0.0017

random.seed(69)
torch.manual_seed(69)
np.random.seed(69)

class Miir(data.Dataset):
    def __init__(self, data_path=DATA_PATH, train=True):
        h5 = h5py.File(data_path, 'r')
        self.train = train
        x = h5['features']
        y = h5['targets']
        subjects = h5['subjects']
        train_indx=[]
        self.train_x=[]
        self.test_x=[]
        self.train_y = []
        self.test_y = []
        self.train_subjects = []
        self.test_subjects=[]
        self.test_subject_id = 1
        for i,e in enumerate(subjects):
            if e != self.test_subject_id:
                train_indx.append(i)
        for i, e in enumerate(x):
            if i in train_indx:
                self.train_x.append(e)
            if i not in train_indx:
                self.test_x.append(e)
        for i, e in enumerate(y):
            if i in train_indx:
                self.train_y.append(e)
            if i not in train_indx:
                self.test_y.append(e)
        for i, e in enumerate(subjects):
            if i in train_indx:
                self.train_subjects.append(e)
            if i not in train_indx:
                self.test_subjects.append(e)
        self.train_size = len(self.train_x)
        self.test_size = len(self.test_x)
    
    def __len__(self):
        if self.train:
            return self.train_size
        else:
            return self.test_size

    def __getitem__(self, index):
        if self.train:
            return self.train_x[index], self.train_y[index], self.train_subjects[index]
        else:
            return self.test_x[index], self.test_y[index], self.test_subjects[index]


def loss_fn(y, yhat, f_mean, f_logvar, z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar):
   
    batch_size = y.size(0)
    cross_entropy = F.cross_entropy(yhat, y)
    kld_f = - 1 * torch.sum(1+f_logvar - torch.pow(f_mean,2) - torch.exp(f_logvar))
    z_post_var = torch.exp(z_post_logvar)
    z_prior_var = torch.exp(z_prior_logvar)
    kld_z = 1 * torch.mean(z_prior_logvar - z_post_logvar + ((z_post_var + torch.pow(z_post_mean - z_prior_mean, 2))
                                                               / z_prior_var) - 1)

    return (cross_entropy + (kld_f + kld_z)) / batch_size, kld_f / batch_size, kld_z / batch_size, cross_entropy/batch_size


def save_model(model, optim, epoch, path):
    torch.save({
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'opimizer': optim.state_dict()}, path)


def check_accuracy(model,test):
    model.eval()
    total=0
    correct_y=0
    with torch.no_grad():
        for item in test:
            x,y,sub=item
            y=torch.argmax(y, dim=1)
            *_, yhat = model(x)            
            yhat = torch.argmax(yhat.data, 1)           
            total += y.size(0)            
            correct_y+=(yhat==y).sum().item()
    model.train()
    return (correct_y/total)

def train_classifier(model, optim, dataset, epochs, path, test, start = 0):
    model.train()
    accuracies=[]
    max_acc=0
    for epoch in range(start, epochs):
        losses = []
        kld_fs = []
        kld_zs = []
        cross_entropies = []
        print("Running Epoch: {}".format(epoch+1))
        for item in tqdm(dataset):
            x, y, subject = item
            y = torch.argmax(y, dim=1)  # one hot back to int
            optim.zero_grad()
            f_mean, f_logvar, f, z_post_mean, z_post_logvar, z, z_prior_mean,z_prior_logvar, yhat = model(x)
            loss, kld_f, kld_z, cross_entropy = loss_fn(y, yhat, f_mean, f_logvar,
                                                       z_post_mean, z_post_logvar, z_prior_mean, z_prior_logvar)
            loss.backward()
            optim.step()
            losses.append(loss.item())
            kld_fs.append(kld_f.item())
            kld_zs.append(kld_z.item())
            cross_entropies.append(cross_entropy.item())

        # training_accuracy = check_accuracy(model, dataset)
        test_accuracy = check_accuracy(model, test)
        meanloss = np.mean(losses)
        meanf = np.mean(kld_fs)
        meanz = np.mean(kld_zs)
        mean_cross_entropies = np.mean(cross_entropies)
        accuracies.append(test_accuracy)
        print("Epoch {} : Average Loss: {} KL of f : {} KL of z : {} "
              "Cross Entropy: {} Test Accuracy: {}".format(epoch + 1, meanloss, meanf, meanz, mean_cross_entropies,
                                                           test_accuracy))
        if test_accuracy>max_acc:
            max_acc=test_accuracy            
            save_model(model, optim, epoch, path)
    return accuracies,max_acc

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    model = DisentangledEEG(factorized=True, nonlinearity=True)
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train_data = Miir(DATA_PATH, True)
    test_data = Miir(DATA_PATH, False)
    loader = data.DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0)
    loader_test = data.DataLoader(test_data, batch_size=60, shuffle=True, num_workers=0)
    c,max_acc=train_classifier(model=model, optim=optim, dataset=loader, epochs=100,
                     path='./checkpoint_disentangled_classifier.pth', test=loader_test)

