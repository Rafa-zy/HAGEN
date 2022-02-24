import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from lib import utils
from model.hagen_model import HAGENModel
from model.loss import cross_entropy, cal_hr_loss
from sklearn import metrics as metrics_sk
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def sigmoid(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            array[i][j] = 1/(1 + np.exp(-array[i][j]))
    return array


class HAGENSupervisor:
    def __init__(self, month, **kwargs):
        self.month = month
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1)) 

        hagen_model = HAGENModel(self._logger, **self._model_kwargs)
        print(hagen_model)
        self.hagen_model = hagen_model
        self._logger.info("Model created")
        self._epoch_num = self._train_kwargs.get('epoch', 0)
        self._lmd = self._train_kwargs.get('lmd', 0.01)

        train_iterator = self._data['train_loader'].get_iterator()
        val_iterator = self._data['val_loader'].get_iterator()
        ys = []
        for _, (x, y) in enumerate(train_iterator):
            x, y = self._prepare_data(x, y)
            ys.append(y.cpu())
        for _, (x, y) in enumerate(val_iterator):
            x, y = self._prepare_data(x, y)
            ys.append(y.cpu())
        ys1 = np.reshape(self._data['y_train'][:, :, :, :], (-1, self._model_kwargs['output_dim']))
        ys2 = np.reshape(self._data['y_val'][:, :, :, :], (-1, self._model_kwargs['output_dim']))
        ys = np.concatenate([ys1, ys2])
        self._threshold = 1 - np.mean(ys)

    def _get_log_dir(self, kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            subgraph_size = kwargs['model'].get('subgraph_size')
            alpha = kwargs['model'].get('tanhalpha')
            patience = kwargs['train'].get('patience')
            thistime = time.strftime('%m%d%H%M%S')
            log_dir = f'./logs/{self.month}/ds{max_diffusion_step}_rl{num_rnn_layers}_gs{subgraph_size}_alpha{alpha}_pa{patience}_{thistime}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        config = dict(self._kwargs)
        config['model_state_dict'] = self.hagen_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, f'{self._log_dir}/model.tar')
        return f'{self._log_dir}/model.tar'

    def _setup_graph(self):
        with torch.no_grad():
            self.hagen_model = self.hagen_model.eval()
            val_iterator = self._data['val_loader'].get_iterator()
            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.hagen_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0, flag=None):
        with torch.no_grad():
            self.hagen_model = self.hagen_model.eval()
            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            l1s = []
            l2s = []
            y_truths = []
            y_preds = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output, adj_mx = self.hagen_model(x, y, batches_seen)
                loss, l1, l2 = self._compute_loss(y, output, 'train', adj_mx, self._lmd)
                losses.append(loss.item())
                l1s.append(l1.item())
                l2s.append(l2.item())
                y_truths.append(y.cpu())
                y_preds.append(output.cpu())


            mean_loss = np.mean(losses)
            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)
            y_preds = np.concatenate(y_preds, axis=1)
            y_preds = np.transpose(y_preds, (1, 0, 2))
            y_preds = np.reshape(y_preds, (y_preds.shape[0], y_preds.shape[1], self._model_kwargs['num_nodes'], -1))
            y_truth = self._data[f'y_{dataset}'][:, :, :, :]
            y_pred = y_preds[:y_truth.shape[0], :, :, :]
            y_truth_reshape = np.reshape(y_truth, (-1, self._model_kwargs['output_dim']))
            y_pred_reshape = np.reshape(y_pred, (-1, self._model_kwargs['output_dim']))
            y_pred_reshape_sigmoid = sigmoid(y_pred_reshape)
            ss = MinMaxScaler(feature_range=(0, 1))
            y_pred_reshape_sigmoid = ss.fit_transform(y_pred_reshape_sigmoid)
            threshold = np.quantile(y_pred_reshape_sigmoid, self._threshold)
            y_pred_reshape_sigmoid[y_pred_reshape_sigmoid >= threshold] = 1
            y_pred_reshape_sigmoid[y_pred_reshape_sigmoid < threshold] = 0
            
            macro_f1 = metrics_sk.f1_score(y_truth_reshape, y_pred_reshape_sigmoid, average = 'macro')
            micro_f1 = metrics_sk.f1_score(y_truth_reshape, y_pred_reshape_sigmoid, average = 'micro')
            cur_flag =  macro_f1 * 0.6 + micro_f1 * 0.4
            self._logger.info('{}: The average macro-F1 score is {:.5f}, average micro-F1 score is {:.5f}'.format(dataset, macro_f1, micro_f1))
            if dataset == 'test':
                return mean_loss, macro_f1, micro_f1
            else:
                update_best = False
                if(cur_flag >= flag): 
                    update_best = True
                return mean_loss, update_best, np.mean(l1s), np.mean(l2s), macro_f1, micro_f1

    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=1, epsilon=1e-8, **kwargs):
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.hagen_model.parameters(), lr=base_lr, eps=epsilon)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=lr_decay_ratio)
        self._logger.info('Start training ...')
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))
        batches_seen = num_batches * self._epoch_num

        best_performance = 0.0
        micro_best = 0.0
        macro_best = 0.0
        best_epoch = 0

        for epoch_num in range(self._epoch_num, epochs):
            self.hagen_model = self.hagen_model.train()
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            l1s = []
            l2s = []
            start_time = time.time()
            for _, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()
                x, y = self._prepare_data(x, y)
                output, adj_mx = self.hagen_model(x, y, batches_seen)
                if batches_seen == 0:
                    optimizer = torch.optim.Adam(self.hagen_model.parameters(), lr=base_lr, eps=epsilon)
                loss, l1, l2 = self._compute_loss(y, output, 'train', adj_mx, self._lmd)
                self._logger.debug(loss.item())
                losses.append(loss.item())
                l1s.append(l1.item())
                l2s.append(l2.item())
                batches_seen += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.hagen_model.parameters(), self.max_grad_norm)
                optimizer.step()
            lr_scheduler.step()
            val_loss, update, valCE, valHR, valmacro, valmicro = self.evaluate(dataset='val', batches_seen=batches_seen, flag=best_performance)
            end_time = time.time()
            self._writer.add_scalar('training loss', np.mean(losses), batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, ({:.4f} / {:.4f}) val_loss: {:.4f}, ({:.4f} / {:.4f}) lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen, np.mean(losses), np.mean(l1s), np.mean(l2s), 
                                           val_loss, valCE, valHR, lr_scheduler.get_lr()[0], (end_time - start_time))
                self._logger.info(message)
            test_loss, macro, micro = self.evaluate(dataset='test', batches_seen=batches_seen)
            
            if update:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val F1 increase from {:.4f} to {:.4f}, '
                        'saving to {}'.format(best_performance, valmicro * 0.4 + valmacro * 0.6, model_file_name))
            else:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

            if update == True:
                best_epoch = epoch_num
                micro_best = micro
                macro_best = macro
                best_performance = valmicro * 0.4 + valmacro * 0.6

        self._logger.info('For the best epoch{}: The average macro-F1 score is {:.5f}, '
                          'average micro-F1 score is {:.5f}'.format(best_epoch, macro_best, micro_best))

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y
    
    def _compute_loss(self, y_true, y_predicted, loss_type='train', adj_mx=None, beta=0.01):
        loss_seq = cross_entropy(y_predicted, y_true)
        if(loss_type=='train'):
            loss_hr = 0.0
            y_true = y_true.squeeze(0)
            y_predicted = y_predicted.squeeze(0)
            for y in y_true:
                y = y.reshape([self.num_nodes, self.output_dim])
                loss_hr += cal_hr_loss(y, adj_mx, self.input_dim) 
            loss = 1 * loss_seq + beta * loss_hr
        else:
            loss = loss_seq
            loss_hr = 0.0
        return loss, loss_seq, loss_hr 
