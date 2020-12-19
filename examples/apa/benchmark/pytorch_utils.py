import os
import pandas as pd
import numpy as np
from collections import defaultdict,OrderedDict
from itertools import combinations
from itertools import chain
from collections import namedtuple
import pickle
import os.path
import shutil
import inspect
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pas_utils import *


class MouseGeneAPADataset():
    def __init__(self, X, X_indices_original, Y, pas_numbers, batch_size, shuffle):
        'Initialization'
        assert batch_size > 0
        self.X = X
        self.Y = Y
        self.pas_numbers = pas_numbers
        self.X_indices_original = X_indices_original
        self.pas_numbers_cumsum = np.insert(pas_numbers.cumsum(), 0, 0)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.pas_numbers)

    def __getitem__(self, index):
        'Generates one sample of data'
        if isinstance(index, slice):
            assert index.step is None
            assert index.start >= 0 and index.stop <= len(self)
            indices1 = self.pas_numbers_cumsum[index.start]
            indices2 = self.pas_numbers_cumsum[index.stop]
            X = self.X[indices1:indices2]
            y = self.Y[indices1:indices2]
            pas_number = self.pas_numbers[index]
            return X, y, pas_number

        elif isinstance(index, int):
            indices1 = self.pas_numbers_cumsum[index]
            indices2 = self.pas_numbers_cumsum[index+1]
            X = self.X[indices1:indices2]
            y = self.Y[indices1:indices2]
            pas_number = indices2-indices1
            return X, y, pas_number

    def __iter__(self):
        self.idx = np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(self.idx)
        self.current = 0
        return self

    def __next__(self):
        if self.current >= len(self):
            raise StopIteration
        else:
            indices = self.idx[self.current:min(
                self.current+self.batch_size, len(self))]
            pas_numbers = np.array([self.pas_numbers[i] for i in indices])
            max_pas_numbers = np.max(pas_numbers)
            X_padded = np.zeros(
                (len(indices), max_pas_numbers, *self.X.shape[1:]))
            Y_padded = np.zeros((len(indices), max_pas_numbers))
            for i, ind in enumerate(indices):
                mark1 = self.pas_numbers_cumsum[ind]
                mark2 = self.pas_numbers_cumsum[ind+1]
                X_padded[i, :self.pas_numbers[ind]] = self.X[mark1:mark2]
                Y_padded[i, :self.pas_numbers[ind]] = self.Y[mark1:mark2]
            self.current += self.batch_size
            return (torch.from_numpy(X_padded).type(torch.float32),
                    torch.from_numpy(Y_padded).type(torch.float32),
                    torch.from_numpy(pas_numbers).type(torch.int64))

    def set_shuffle(self, shuffle):
        self.shuffle = shuffle


class APAModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.shapes = {}
        # before_input_reshape (num_genes,max_num_pas,params["seq_len"],4)
        # after_input_reshape (num_genes*max_num_pas,params["seq_len"],4)
        # after transpose (num_genes*max_num_pas,4,params["seq_len"])
        if params["net_type"] == "Single-Conv-Net":
            self.conv1d = nn.Conv1d(
                4, params["conv1d_out_dim"], params["conv1d_kernel_size"])
            self.shapes["after_conv1d"] = (
                None, params["conv1d_out_dim"], (params["seq_len"]-params["conv1d_kernel_size"])+1)
            # after maxpool1d (num_genes*max_num_pas,params["conv1d_out_dim"],(params["seq_len"]-params["pool_size"])//params["pool_size"]+1)
            s = self.shapes["after_conv1d"]
            self.batchnorm = nn.BatchNorm1d(s[1])
            self.shapes["after_maxpool1d"] = (
                None, params["conv1d_out_dim"], (s[2]-params["pool_size"])//params["pool_size"]+1)
            s = self.shapes["after_maxpool1d"]
            linear1_dim = s[1]*s[2]
        elif params["net_type"] == "Multi-Conv-Net":
            self.conv1d_1 = nn.Conv1d(
                4, params["conv1d_out_dim_1"], params["conv1d_kernel_size"]
            )
            self.shapes["after_conv1d_1"] = (
                None, params["conv1d_out_dim_1"], (params["seq_len"]-params["conv1d_kernel_size"]+1))
            s = self.shapes["after_conv1d_1"]
            self.batchnorm_1 = nn.BatchNorm1d(s[1])
            self.shapes["after_maxpool1d_1"] = (
                None, params["conv1d_out_dim_1"], (s[2]-params["pool_size_1"])//params["pool_size_1"]+1)
            s = self.shapes["after_maxpool1d_1"]
            self.conv1d_2 = nn.Conv1d(
                params["conv1d_out_dim_1"], params["conv1d_out_dim_2"], params["conv1d_kernel_size"])
            self.shapes["after_conv1d_2"] = (
                None, params["conv1d_out_dim_2"], (s[2]-params["conv1d_kernel_size"])+1)
            s = self.shapes["after_conv1d_2"]
            self.batchnorm_2 = nn.BatchNorm1d(s[1])
            self.shapes["after_maxpool1d_2"] = (
                None, params["conv1d_out_dim_2"], (s[2]-params["pool_size_2"])//params["pool_size_2"]+1)
            s = self.shapes["after_maxpool1d_2"]
            linear1_dim = s[1]*s[2]
        elif params["net_type"] == "Feature-Net":
            self.shapes["feature_output_size"] = (None, params["feature_size"])
            s = self.shapes["feature_output_size"]
            linear1_dim = s[1]
        self.linear1 = nn.Linear(linear1_dim, params["linear1_dim"])
        assert params["lstm_output_size"] % 2 == 0
        self.lstm = nn.LSTM(
            input_size=params["linear1_dim"],
            hidden_size=params["lstm_output_size"]//2,
            batch_first=True,
            bidirectional=True)
        s = (None, params["lstm_output_size"])
        self.params = params
        self.dropout = nn.Dropout(p=params["dropout_rate"])
        self.linear2 = nn.Linear(s[1], 1)

    def forward(self, values, pas_numbers):
        params = self.params

        num_genes, max_num_pas = values.shape[0], values.shape[1]

        if params["net_type"] == "Single-Conv-Net":
            values = values.view(num_genes*max_num_pas, params["seq_len"], 4)
            values = values.transpose(1, 2)
            values = F.relu(self.batchnorm(self.conv1d(values)))
            values = F.max_pool1d(values, kernel_size=params["pool_size"])
            s = self.shapes["after_maxpool1d"]
            values = values.view(num_genes*max_num_pas, s[1]*s[2])

        elif params["net_type"] == "Multi-Conv-Net":
            values = values.view(num_genes*max_num_pas, params["seq_len"], 4)
            values = values.transpose(1, 2)
            values = F.relu(self.batchnorm_1(self.conv1d_1(values)))
            values = F.max_pool1d(values, kernel_size=params["pool_size_1"])
            values = F.relu(self.batchnorm_2(self.conv1d_2(values)))
            values = F.max_pool1d(values, kernel_size=params["pool_size_2"])
            s = self.shapes["after_maxpool1d_2"]
            values = values.view(num_genes*max_num_pas, s[1]*s[2])

        elif params["net_type"] == "Feature-Net":
            pass

        values = F.relu(self.linear1(values))
        values = values.view(num_genes, max_num_pas, params["linear1_dim"])
        # argsort pas_numbers
        pas_numbers_sorted, perm_idx = torch.sort(pas_numbers, descending=True)
        values = values[perm_idx]
        values = pack_padded_sequence(
            values, pas_numbers_sorted, batch_first=True)

        values, lstm_hidden = self.lstm(values)
        values, _ = pad_packed_sequence(values, batch_first=True)
        _, unperm_idx = torch.sort(perm_idx)
        values = values[unperm_idx]
        values.view(num_genes*max_num_pas, params["lstm_output_size"])
        values = self.dropout(values)
        values = self.linear2(values)
        values = values[..., 0]
        return values


def loss_function(logits, target, pas_numbers, model, params):
    #     mask=torch.zeros_like(logits)
    for i in range(len(pas_numbers)):
        logits[i, pas_numbers[i]:] = -1e10
    target_multiplies_softmax = - target * F.log_softmax(logits, -1)
    loss = torch.sum(target_multiplies_softmax, -1)
    reg_loss = regularization_loss(model, params)
    return loss.mean()+reg_loss


def regularization_loss(model, params):
    reg_loss = torch.tensor(0.).to(params["device"])
    for k, v in model.named_parameters():
        k1, k2 = k.split('.')
        if k1 in ["conv1d_1", "conv1d_2", "linear1", "linear2", "conv1d"] and k2 == "weight":
            if k == "conv2d.weight":
                reg_loss += torch.sum(v**2)*params["beta"]
            elif k in ("linear1.weight", "linear2.weight"):
                reg_loss += torch.sum(v**2)*params["beta"]
    return reg_loss
# TODO: should be modified to exclude padding


def mae_loss(logits, target, pas_numbers, return_pred=False):
    for i in range(len(pas_numbers)):
        logits[i, pas_numbers[i]:] = -1e32
    softmax = F.softmax(logits, -1)
    loss = 0
    for i in range(len(pas_numbers)):
        loss += F.l1_loss(softmax[i][:pas_numbers[i]],
                          target[i][:pas_numbers[i]], reduction="sum")
    loss /= torch.sum(pas_numbers).item()
    if return_pred:
        return loss, softmax
    else:
        return loss


def predict(logits, pas_numbers):
    for i in range(len(pas_numbers)):
        logits[i, pas_numbers[i]:] = -1e32
    softmax = F.softmax(logits, -1)
    return softmax


def model_predict(model, data_set, params, softmax=True):
    with torch.no_grad():
        model.eval()
        data_set.set_shuffle(False)
        predictions = []
        for local_batch, local_labels, local_pas_numbers in data_set:
            local_batch = (local_batch).to(params["device"])
            local_labels = (local_labels).to(params["device"])
            local_pas_numbers = (local_pas_numbers).to(params["device"])
            local_outputs = model(local_batch, local_pas_numbers)
            if softmax:
                local_pred = predict(local_outputs, local_pas_numbers)
            else:
                local_pred = local_outputs
            for i in range(len(local_pas_numbers)):
                predictions += local_pred[i][:local_pas_numbers[i]].tolist()

        predictions = np.array(predictions)
        return predictions


def mae_eval(model, epoch, name, data_set, params):
    with torch.no_grad():
        model.eval()
        loss_validation = 0
        data_set.set_shuffle(False)
        loss_mae = 0
        total_num_pas = 0
        # mae evaluation
        for local_batch, local_labels, local_pas_numbers in data_set:
            local_batch = (local_batch).to(params["device"])
            local_labels = (local_labels).to(params["device"])
            local_pas_numbers = (local_pas_numbers).to(params["device"])

            local_outputs = model(local_batch, local_pas_numbers)
            local_loss, local_pred = mae_loss(local_outputs, local_labels.type(
                torch.float32), local_pas_numbers, return_pred=True)
            local_total_num_pas = torch.sum(local_pas_numbers).item()
            loss_mae += local_total_num_pas*local_loss
            total_num_pas += local_total_num_pas
        loss_mae = loss_mae/total_num_pas
        if epoch>=0:
            print("[epoch %d] %s\tloss_mae=%.3f" % (epoch, name, loss_mae))
        else:
            print("%s\tloss_mae=%.3f"%(name,loss_mae))
        return loss_mae


def comparison_eval(model, epoch, name, data_set, data_dict, params, phase):
    with torch.no_grad():
        model.eval()
        data_set.set_shuffle(False)
        # comparison accuracy
        accuracy_comparison = 0
        usage_pred_unrolled = []

        for local_batch, local_labels, local_pas_numbers in data_set:
            local_batch = (local_batch).to(params["device"])
            local_labels = (local_labels).to(params["device"])
            local_pas_numbers = (local_pas_numbers).to(params["device"])

            local_outputs = model(local_batch, local_pas_numbers)
            local_pred = predict(local_outputs, local_pas_numbers)
            for i, pas_number in enumerate(local_pas_numbers.cpu().numpy()):
                usage_pred_unrolled += local_pred[i, :pas_number].tolist()

        usage_pred_unrolled = np.array(usage_pred_unrolled)

        assert len(usage_pred_unrolled) == len(
            data_dict["X_%s" % (phase)])
        comparison_pred = usage_pred_unrolled[data_dict["X1_indices_comparison_%s" % (
            phase)]] > usage_pred_unrolled[data_dict["X2_indices_comparison_%s" % (phase)]]
        # comparison_ground_truth = (
        # data_dict["Y_comparison_%s" % (phase)].argmax(axis=1) == 0)
        comp1 = data_dict["Y_usage_%s" %
                          (phase)][data_dict["X1_indices_comparison_%s" % (phase)]]
        comp2 = data_dict["Y_usage_%s" %
                          (phase)][data_dict["X2_indices_comparison_%s" % (phase)]]
        comparison_ground_truth = comp1 >= comp2

        accuracy_comparison = np.mean(
            comparison_pred == comparison_ground_truth)
        
        if epoch>=0:
            print("[epoch %d] %s\taccuracy_comparison=%.3f" %
                (epoch, name, accuracy_comparison))
        else:
            print("%s\taccuracy_comparison=%.3f"%(name,accuracy_comparison))
        return accuracy_comparison


def cross_entropy_eval(model, epoch, name, data_set, params):
        # cross entropy evaluation
    with torch.no_grad():
        model.eval()
        data_set.set_shuffle(False)
        loss_cross_entropy = 0
        for local_batch, local_labels, local_pas_numbers in data_set:
            local_batch = (local_batch).to(params["device"])
            local_labels = (local_labels).to(params["device"])
            local_pas_numbers = (local_pas_numbers).to(params["device"])
            local_outputs = model(local_batch, local_pas_numbers)
            local_loss = loss_function(local_outputs, local_labels.type(
                torch.float32), local_pas_numbers, model, params)
            loss_cross_entropy += local_loss.item()*len(local_pas_numbers)/len(data_set)
        if epoch>=0:
            print("[epoch %d] %s\tcross_entropy=%.3f" %
                (epoch, name, loss_cross_entropy))
        else:
            print("%s\tcross_entropy=%.3f"%(name,loss_cross_entropy))
        return loss_cross_entropy


def max_pred_eval(model, epoch, name, data_set, params):
    with torch.no_grad():
        model.eval()
        data_set.set_shuffle(False)
        # max_pred evaluation
        accuracy_max_pred = 0
        for local_batch, local_labels, local_pas_numbers in data_set:
            local_batch = (local_batch).to(params["device"])
            local_labels = (local_labels).to(params["device"])
            local_pas_numbers = (local_pas_numbers).to(params["device"])
            local_outputs = model(local_batch, local_pas_numbers)
            local_pred = predict(local_outputs, local_pas_numbers)
            local_max_pred = local_pred.argmax(dim=1)
            local_max_pred_ground_truth = local_labels.argmax(dim=1)
            local_max_pred_accuracy = torch.mean(
                (local_max_pred == local_max_pred_ground_truth).type(torch.float32))
            accuracy_max_pred += local_max_pred_accuracy * \
                len(local_pas_numbers)/len(data_set)
        if epoch>=0:
            print("[epoch %d] %s\tmax_pred_accuracy=%.3f" %
                (epoch, name, accuracy_max_pred))
        else:
            print("%s\tmax_pred_accuracy=%.3f"%(name,accuracy_max_pred))
        return accuracy_max_pred


def metric_by_pas_number(model, epoch, name, data_set, params):
    with torch.no_grad():
        model.eval()
        data_set.set_shuffle(False)
        loss_mae = defaultdict(float)
        acc_max_pred=defaultdict(float)
        total_num_pas = defaultdict(int)
        total_num_genes = defaultdict(int)
        # mae evaluation
        for local_batch, local_labels, local_pas_numbers in data_set:
            local_batch = (local_batch).to(params["device"])
            local_labels = (local_labels).to(params["device"])
            local_pas_numbers = (local_pas_numbers).to(params["device"])

            local_outputs = model(local_batch, local_pas_numbers)
            local_loss, local_pred = mae_loss(local_outputs, local_labels.type(
                torch.float32), local_pas_numbers, return_pred=True)
            local_max_pred = local_pred.argmax(dim=1)
            local_max_pred_ground_truth = local_labels.argmax(dim=1)
            local_max_pred_res = (local_max_pred == local_max_pred_ground_truth).type(torch.float32)

            local_pas_numbers = local_pas_numbers.cpu().numpy()
            local_pred = local_pred.cpu().numpy()
            local_labels = local_labels.cpu().numpy()
            local_max_pred_res=local_max_pred_res.cpu().numpy()
            base_num = 0
            for i in range(len(local_pas_numbers)):
                total_num_pas[local_pas_numbers[i]] += local_pas_numbers[i]
                total_num_genes[local_pas_numbers[i]] += 1
                pred = local_pred[base_num:base_num+local_pas_numbers[i]]
                label = local_labels[base_num:base_num+local_pas_numbers[i]]
                loss_mae[local_pas_numbers[i]] += np.abs(pred-label).sum()
                acc_max_pred[local_pas_numbers[i]]+=local_max_pred_res[i]
                base_num += local_pas_numbers[i]
        loss_mae = OrderedDict({k:loss_mae[k]/total_num_pas[k] for k in sorted(loss_mae.keys())})
        acc_max_pred= OrderedDict({k:acc_max_pred[k]/total_num_genes[k] for k in sorted(acc_max_pred.keys())})
    return loss_mae,acc_max_pred
