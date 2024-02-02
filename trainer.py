import os
import csv
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from my_utils.train.trainer import Trainer
from typing import *
from rawdata_process import sparse2dense

class MyTrainer(Trainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.interval = 25 # for check point to save '.pt' file
        if self.args['pre_split'] == True and self.args['task_type'] == 'classification':
            self.best_recorder.initial(['train loss', 'test loss', 'test acc', 'val loss', 'val acc'], ['min', 'min', 'max', 'min', 'max'])
        elif self.args['pre_split'] == True and self.args['task_type'] == 'regression':
            self.best_recorder.initial(['train loss', 'test loss', 'val loss'], ['min', 'min', 'min'])
        else:
            self.best_recorder.initial(['train loss', 'test loss', 'test acc'], ['min', 'min', 'max'])

    def train(self, train_mask=None, test_mask=None, val_mask=None, scheduler=None) -> None:
        '''
            Train the model for several epochs
        '''
        for epoch in range(self.args['epochs']):
            train_loss = self.train_batch(self.dataset[train_mask])
            test_loss, test_acc = self.test(self.dataset[test_mask])
            val_loss, val_acc = self.valid(self.dataset[val_mask])

            if self.args['pre_split'] == True and self.args['task_type'] == 'classification':
                record_data = [train_loss, test_loss, test_acc, val_loss, val_acc]
                message = f'epoch:{epoch}, train_loss:{train_loss:.4f}, test_loss:{test_loss:.4f}, test_acc:{test_acc:.4f}, val_loss:{val_loss:.4f}, val_acc:{val_acc:.4f}'
            elif self.args['pre_split'] == True and self.args['task_type'] == 'regression':
                record_data = [train_loss, test_loss, val_loss]
                message = f'epoch:{epoch}, train_loss:{train_loss:.4f}, test_loss:{test_loss:.4f}, val_loss:{val_loss:.4f}'
            else:
                record_data = [train_loss, test_loss, test_acc]
                message = f'epoch:{epoch}, train_loss:{train_loss:.4f}, test_loss:{test_loss:.4f}, test_acc:{test_acc:.4f}'

            self.best_recorder.cal_best(epoch, record_data)
            self.logger.log(message, record_data)
            self.checkpoint(epoch)

            if self.args['pre_split'] == True:
                if self.early_stop(val_loss, 'min'):
                    break
            else:
                if self.early_stop(test_loss, 'min'):
                    break
            if scheduler is not None:
                scheduler.step()
    
    def train_batch(self, data_list: List[torch_geometric.data.Data]) -> float:
        '''
            Train the model for 1 batch
            Parameters:
                param1: Data obj
            Return:
                loss value
        '''
        self.model.train()

        data_list = DataLoader(data_list, self.args['batch_size'], shuffle=self.args['pre_split'])
        total_loss = 0
        for data in data_list:
            self.optimizer.zero_grad()
            if data.x.is_sparse: # torch api to_dense() encounter some bugs
                data.x = sparse2dense(data.x)
            y_pred = self.model(data.x, data.edge_index, data.batch, data.node_centrality, data.edge_centrality)
            loss = self.criterion(y_pred, data.y)
            total_loss += loss.item() / len(data_list)
            loss.backward()
            self.optimizer.step()
            
        # if self.args['dataset'] == 'ZINC_full' or self.args['dataset'] == 'ZINC_subset':
        #     predicts = predicts.view(-1)
        # elif self.args['dataset'] == 'QM9':
        #     # TODO
        #     ...

        return total_loss
    
    def predict(self, data_list) -> list:
        '''
            Inference the model
        '''
        self.model.eval()

        with torch.no_grad():
            for data in data_list:
                ...

        return None
    
    def test(self, data_list: List[torch_geometric.data.Data]) -> Tuple[float, float]:
        '''
            Test the model on test set
        '''
        self.model.eval()

        predicts_l = []
        labels_l = []
        data_list = DataLoader(data_list, self.args['batch_size'], shuffle=False)
        with torch.no_grad():
            for data in data_list:
                # Actually, model.eval() will disable Dropout, which may allowing sparse tensors input without converting
                if data.x.is_sparse:
                    data.x = sparse2dense(data.x)
                y_pred = self.model(data.x, data.edge_index, data.batch, data.node_centrality, data.edge_centrality)
                predicts_l.append(y_pred)
                labels_l.append(data.y)

            predicts = torch.cat(predicts_l).float()
            labels = torch.cat(labels_l)
            if self.args['dataset'] == 'ZINC_full' or self.args['dataset'] == 'ZINC_subset':
                predicts = predicts.view(-1)
                acc = torch.tensor(0.0)
            elif self.args['dataset'] == 'QM9':
                # TODO
                ...
                acc = torch.tensor(0.0)
            else:
                acc = (torch.argmax(predicts, dim=-1) == labels).sum() / predicts.shape[0]
            loss = self.criterion(predicts, labels)

        return loss.item(), acc.item()
    
    def valid(self, data_list: List[torch_geometric.data.Data]) -> Tuple[float, float]:
        '''
            Valid the model on test set
        '''
        if len(data_list) == 0:
            return 0.0, 0.0
        
        self.model.eval()

        predicts_l = []
        labels_l = []
        data_list = DataLoader(data_list, self.args['batch_size'], shuffle=False)
        with torch.no_grad():
            for data in data_list:
                # Actually, model.eval() will disable Dropout, which may allowing sparse tensors input without converting
                if data.x.is_sparse:
                    data.x = sparse2dense(data.x)
                y_pred = self.model(data.x, data.edge_index, data.batch, data.node_centrality, data.edge_centrality)
                predicts_l.append(y_pred)
                labels_l.append(data.y)

            predicts = torch.cat(predicts_l).float()
            labels = torch.cat(labels_l)
            if self.args['dataset'] == 'ZINC_full' or self.args['dataset'] == 'ZINC_subset':
                predicts = predicts.view(-1)
                acc = torch.tensor(0.0)
            elif self.args['dataset'] == 'QM9':
                # TODO
                acc = torch.tensor(0.0)
                ...
            else:
                acc = (torch.argmax(predicts, dim=-1) == labels).sum() / predicts.shape[0]
            loss = self.criterion(predicts, labels)

        return loss.item(), acc.item()

    def end_custom(self) -> List[float]:
        '''
            Record loss and acc.
        '''
        # TODO different ...
        test_best = self.best_recorder.get_best()[2][1]
        
        if self.args['pre_split'] == True and self.args['task_type'] == 'classification':
            test_best = self.best_recorder.get_best()[2][1]
            val_best = self.best_recorder.get_best()[4][1]
        elif self.args['pre_split'] == True and self.args['task_type'] == 'regression':
            test_best = self.best_recorder.get_best()[1][1]
            val_best = self.best_recorder.get_best()[2][1]
        else:
            test_best = self.best_recorder.get_best()[2][1]
            val_best = 0
        best = [test_best, val_best]

        if not os.path.exists('records'):
            os.mkdir('records')
        
        try:
            file = open(self.args['record_path'], 'a')
        except:
            file = open('./records/record.csv', 'a')

        writer = csv.writer(file)
        records = best + list(self.args.values())
        writer.writerow(records)
        file.close()

        return best