import matplotlib.pyplot as plt
import os

from torch.autograd import Variable

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F

from tensorflow import summary

from .encoder import Encoder
from .decoder import Decoder


class DA_rnn(nn.Module):
    """da_rnn."""

    def __init__(self, X, y, T,
                 encoder_num_hidden,
                 decoder_num_hidden,
                 batch_size,
                 learning_rate,
                 epochs,
                 loss_func,
                 train_size,
                 parallel=False):
        """da_rnn initialization."""
        super(DA_rnn, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden
        self.decoder_num_hidden = decoder_num_hidden
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T
        self.X = X
        self.y = y
        self.train_size = train_size

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.Encoder = Encoder(input_size=X.shape[1],
                               encoder_num_hidden=encoder_num_hidden,
                               T=T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T).to(self.device)

        # Loss function
        self.criterion = loss_func()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)

        # Training set
        self.train_timesteps = int(self.X.shape[0] * self.train_size)
        # self.y = self.y - np.mean(self.y[:self.train_timesteps])
        self.input_size = self.X.shape[1]

    def train(self, train_summary_writer):
        """training process."""
        iter_per_epoch = int(np.ceil(self.train_timesteps * 1. / self.batch_size))
        self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        self.epoch_losses = np.zeros(self.epochs)

        n_iter = 0

        for epoch in range(self.epochs):
            if self.shuffle:
                ref_idx = np.random.permutation(self.train_timesteps - self.T)
            else:
                ref_idx = np.array(range(self.train_timesteps - self.T))

            idx = 0

            while (idx < self.train_timesteps):
                # get the indices of X_train
                indices = ref_idx[idx:(idx + self.batch_size)]
                # x = np.zeros((self.T - 1, len(indices), self.input_size))
                x = np.zeros((len(indices), self.T - 1, self.input_size))
                y_prev = np.zeros((len(indices), self.T - 1))
                y_gt = self.y[indices + self.T]

                # format x into 3D tensor
                for bs in range(len(indices)):
                    x[bs, :, :] = self.X[indices[bs]:(indices[bs] + self.T - 1), :]
                    y_prev[bs, :] = self.y[indices[bs]: (indices[bs] + self.T - 1)]

                loss = self.train_forward(x, y_prev, y_gt)
                self.iter_losses[int(epoch * iter_per_epoch + idx / self.batch_size)] = loss

                idx += self.batch_size
                n_iter += 1

                if n_iter % 4000 == 0 and n_iter != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                self.epoch_losses[epoch] = np.mean(self.iter_losses[range(
                    epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])
            
            y_train_pred = self.test(on_train=True)
            y_test_pred = self.test(on_train=False)[1:]

            y_train_true = self.y[self.T : (len(y_train_pred) + self.T)]
            y_test_true = self.y[self.T + len(y_train_pred) : (len(self.y) + 1)]

            err_train = np.abs(y_train_true - y_train_pred)
            mae_train = np.mean(err_train)
            rmse_train = np.sqrt(np.mean(err_train ** 2))

            err_test = np.abs(y_test_true - y_test_pred)
            mae_test = np.mean(err_test)
            rmse_test = np.sqrt(np.mean(err_test ** 2))
            
            y_pred = np.concatenate((y_train_pred, y_test_pred))

            err_total = np.abs(self.y[self.T:]-y_pred)
            mae_total = np.mean(err_total)
            rmse_total = np.sqrt(np.mean(err_total ** 2))

            with train_summary_writer.as_default():
                # summary.scalar('Loss', loss, step=epoch)
                summary.scalar('Epoch Average Loss', self.epoch_losses[epoch], step=epoch)
                summary.scalar('MAE_total', mae_total, step=epoch)
                summary.scalar('RMSE_total', rmse_total, step=epoch)
                summary.scalar('MAE_train', mae_train, step=epoch)
                summary.scalar('RMSE_train', rmse_train, step=epoch)
                summary.scalar('MAE_test', mae_test, step=epoch)
                summary.scalar('RMSE_test', rmse_test, step=epoch)
                

            print("Epochs: ", epoch, " Iterations: ", n_iter,
                    " Epoch Loss: ", self.epoch_losses[epoch], " Train MAE", mae_train, 
                    " Train RMSE", rmse_train, " Test MAE: ", mae_test, " Test RMSE", rmse_test)

        y_train_pred = self.test(on_train=True)
        y_test_pred = self.test(on_train=False)

        y_pred = np.concatenate((y_train_pred, y_test_pred))
        plt.ioff()
        plt.figure()
        plt.plot(range(1, 1 + len(self.y)), self.y, label="True")
        plt.plot(range(self.T, len(y_train_pred) + self.T),
                    y_train_pred, label='Predicted - Train')
        plt.plot(range(self.T + len(y_train_pred), len(self.y) + 1),
                    y_test_pred, label='Predicted - Test')
        plt.legend(loc='upper left')
        plt.savefig("test.png")
        img = plt.imread("test.png")
        os.remove("test.png")

            # # Save files in last iterations
            # if epoch == self.epochs - 1:
            #     np.savetxt('../loss.txt', np.array(self.epoch_losses), delimiter=',')
            #     np.savetxt('../y_pred.txt',
            #                np.array(self.y_pred), delimiter=',')
            #     np.savetxt('../y_true.txt',
            #                np.array(self.y_true), delimiter=',')
        
        with torch.no_grad():
            # input_attn = self.Encoder.encoder_attn.weight.cpu().numpy()
            # temp_attn = self.Decoder.attn_layer[2].weight.cpu().numpy()
            # print(input_attn.shape)
            # print(temp_attn.shape)
            with train_summary_writer.as_default():
                # summary.histogram('Input Attention Weights', input_attn, step=1)
                # summary.histogram('Temporal Attention Weights', temp_attn, step=1)
                summary.image('Final Prediction Plot', np.expand_dims(img, 0), step=1)


    def train_forward(self, X, y_prev, y_gt):
        """
        Forward pass.

        Args:
            X:
            y_prev:
            y_gt: Ground truth label

        """
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
        y_pred = self.Decoder(input_encoded, Variable(
            torch.from_numpy(y_prev).type(torch.FloatTensor).to(self.device)))

        y_true = Variable(torch.from_numpy(
            y_gt).type(torch.FloatTensor).to(self.device))

        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()


    def test(self, on_train=False):
        """test."""

        if on_train:
            y_pred = np.zeros(self.train_timesteps - self.T + 1)
        else:
            y_pred = np.zeros(self.X.shape[0] - self.train_timesteps)

        i = 0
        while i < len(y_pred):
            batch_idx = np.array(range(len(y_pred)))[i: (i + self.batch_size)]
            X = np.zeros((len(batch_idx), self.T - 1, self.X.shape[1]))
            y_history = np.zeros((len(batch_idx), self.T - 1))

            for j in range(len(batch_idx)):
                if on_train:
                    X[j, :, :] = self.X[range(
                        batch_idx[j], batch_idx[j] + self.T - 1), :]
                    y_history[j, :] = self.y[range(
                        batch_idx[j], batch_idx[j] + self.T - 1)]
                else:
                    X[j, :, :] = self.X[range(
                        batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1), :]
                    y_history[j, :] = self.y[range(
                        batch_idx[j] + self.train_timesteps - self.T, batch_idx[j] + self.train_timesteps - 1)]

            y_history = Variable(torch.from_numpy(
                y_history).type(torch.FloatTensor).to(self.device))
            _, input_encoded = self.Encoder(
                Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
            y_pred[i:(i + self.batch_size)] = self.Decoder(input_encoded,
                                                           y_history).cpu().data.numpy()[:, 0]
            i += self.batch_size

        return y_pred
