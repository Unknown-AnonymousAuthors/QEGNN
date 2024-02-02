import os
import abc
from math import ceil
import time
from my_utils.train import log
import torch
from torch.utils.data import DataLoader

class Trainer(metaclass=abc.ABCMeta):
    def __init__(self, args, model, dataset, criterion, optimizer, topic, device, scheduler=None, **kwargs):
        self.args = args
        assert 'epochs' in self.args.keys()
        assert 'batch_size' in self.args.keys()
        assert 'animator_output' in self.args.keys()
        
        self.model = model
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=self.args['batch_size'])
        self.criterion = criterion      # loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.topic = topic              # e.g. TransE_WN18RR
        # **kwargs                      # e.g. {'num_nodes'=123...}

        self.dir_path = log.get_log_dir(self.args['log_path'])
        if self.args['animator_output'] == True:
            self.animator = log.Animator(self.topic, dir_path=self.dir_path)
        self.interval = 50
        self.logger = log.Logger(dir_path=self.dir_path)
        self.best_recorder = BestRecorder()

    @abc.abstractmethod
    def train(self):
        '''
        self.model.train()
        for epoch in range(self.args['epochs']):
            loss = 0
            for data in self.dataloader:
                l = self.train_batch(data)
                loss += l
            
            self.test()
            ...
            if self.args['animator_output'] == True:
                self.animator.step(epoch + 1, loss)
            message = f''
            self.log(message, data=())
            self.checkpoint(epoch)
            self.best_recorder.cal_best(epoch, (loss, acc))
        '''
        ...
    
    @abc.abstractmethod
    def train_batch(self, data):
        '''
        self.optimizer.zero_grad()

        out1, out2 = self.model(data)
        loss = self.criterion(out1, out2)
        
        loss.backward()
        self.optimizer.step()

        return loss
        '''
        ...

    @abc.abstractmethod
    def test(self):
        # self.model.eval()
        ...
        
    @abc.abstractmethod
    def valid(self):
        ...

    def test_epoch_once(self):
        epochs = self.args['epochs']
        self.args['epochs'] = 1
        self.train()
        self.args['epochs'] = epochs

    def __repr__(self) -> str:
        s = ''
        s += f'Hyper Parameters:\n'
        for k, v in self.args.items():
            s += f'    {k}: {v}\n'
        s += f'Model: {self.model}\n'
        s += f'Dataloader: {self.dataloader}\n'
        s += f'Criterion: {self.criterion}\n'
        s += f'Optimizer: {self.optimizer}\n'
        s += f'Scheduler: {self.scheduler}\n' if self.scheduler is not None else ''
        # s += f': {self.}\n'
        return s

    def early_stop(self, value, max_or_min): # for loss, not accuracy. (increase/decrease)
        assert 'early_stop' in self.args.keys()
        assert max_or_min in ['max', 'min']

        if not hasattr(self, 'early_stop_value'): # first time
            self.early_stop_value = value
            self.early_stop_count = 0

        if max_or_min == 'min':
            if value >= self.early_stop_value: # bad
                self.early_stop_count += 1
            else:
                self.early_stop_value = value  # good
                self.early_stop_count = 0
        else:   # max
            if value <= self.early_stop_value: # bad
                self.early_stop_count += 1
            else:
                self.early_stop_value = value  # good
                self.early_stop_count = 0
        
        if self.early_stop_count >= self.args['early_stop']:
            return True
        else:
            return False

    def __call__(self, **kwargs):
        self.start()
        self.train(**kwargs)
        ret = self.end()
        return ret

    def start(self, debug=False):
        self.start_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        if debug:
            print('=' * 10, 'START TRAINING', '=' * 10)
        message = str(self)
        self.log(message)

        message = f'start at {self.start_time}'
        if debug:
            print(message)
        self.log(message)

    def end(self, debug=False):
        self.end_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        if debug:
            print('=' * 10, 'TRAINING COMPLETE', '=' * 10)
        
        message = f'start at {self.start_time}, end at {self.end_time}'
        if debug:
            print(message)
        self.log(message)
        
        if hasattr(self, 'best_recorder'):  #[variable, data, epoch, max_or_min]
            message = "Best Record:\n"
            for record in self.best_recorder.get_best():
                variable = record[0]
                data = record[1]
                epoch = record[2]
                max_or_min = record[3]
                message += f'{variable} has a {max_or_min} record {data} at epoch {epoch}.\n'
            if debug:
                print(message)
            self.log(message)
        
        self.del_checkpoint()

        ret = self.end_custom()
        return ret
        
    
    def end_custom(self):
        # self.valid()
        # self.visualization(data)
        # self.send_email()

        # test_best = self.best_recorder.get_best()[2][1]
        # f = open(self.record_path, 'a')
        # f.write(f"{self.args['att_name']};{self.args['att_input']};{self.args['dropout']};{test_best:.4f}\n")
        # f.close()
        # return test_best

        ret = None
        return ret

    def visulization(self, data):
        ...
    
    def log(self, message, data=None):
        self.logger.log(message, data)

    def checkpoint(self, epoch):
        if epoch % self.interval != 0 and epoch != self.args['epochs']-1:
            return
        
        filename = self.dir_path + '/' + self.topic
        filename = filename + f'_epoch{epoch}.pt'
        torch.save(self.model, filename)
    
    def del_checkpoint(self):
        record = self.best_recorder.get_best()[2] # test acc
        filenames = []
        epoch = record[2]
        epoch = ceil(epoch / self.interval) * self.interval
        filename = f'{self.topic}_epoch{epoch}.pt'
        last_checkpoint = f"{self.topic}_epoch{self.args['epochs']-1}.pt"
        filenames.append(filename)
        filenames.append(last_checkpoint)
        filenames.append(f'learning_curve.png')
        filenames.append(f'training_status.log')
        filenames.append(f'training_status.csv')

        all_files = os.listdir(self.dir_path)
        for file in all_files:
            if file not in filenames:
                os.remove(f'{self.dir_path}/{file}')

    def send_email(self):
        filenames = []
        email_title = self.topic
        all_files = os.listdir(self.dir_path)
        for file in all_files:
            filenames.append(f'{self.dir_path}/{file}')
        # checkpoint.save_checkpoint(email_title, filenames)

class BestRecorder:
    def __init__(self):
        ...

    def initial(self, data_names, max_or_min):
        if not isinstance(max_or_min, (tuple, list)):
            max_or_min = [max_or_min]
        for v in max_or_min:
            assert v in ['max', 'min']

        self.data_names = data_names if isinstance(data_names, (tuple, list)) else [data_names]
        self.len = len(self.data_names)
        self.data = [ [float('-inf'), -1, float('inf'), -1] for _ in range(self.len)] # shape: [N, 4]
        self.max_or_min = max_or_min
    
    def clear(self):
        self.data = []
    
    def cal_best(self, epoch, data):
        data_len = len(data) if isinstance(data, (tuple, list)) else 1
        assert self.len == data_len
        
        for i, record in enumerate(self.data):
            if data[i] > record[0]:
                self.data[i][0] = data[i]
                self.data[i][1] = epoch
            if data[i] < record[2]:
                self.data[i][2] = data[i]
                self.data[i][3] = epoch
    
    def get_best(self):
        arr = [[name,                                        # 0.variable name
                data[0] if max_or_min == 'max' else data[2], # 1.data
                data[1] if max_or_min == 'max' else data[3], # 2.epoch
                max_or_min,                                  # 3.max or min
                ] for (data, name, max_or_min) in zip(self.data, self.data_names, self.max_or_min)]
        
        return arr  # [ [variable, data, epoch, max_or_min] * n ]
    
    def __repr__(self):
        arr = self.get_best()
        s = "\nBest record:\n"
        for record in arr:
            s += f'{record[0]:.4f} has a {record[3]:.4f} record {record[1]:.4f} at epoch {record[2]:.4f}.\n'

        return s

if __name__ == '__main__':
    '''
    class ZZ(Trainer):
        def __init__(self, args, model, dataloader, criterion, optimizer, topic):
            super().__init__(args, model, dataloader, criterion, optimizer, topic)
        
        def train(self):
            ...

        def train_batch(self):
            ...

        def test(self):
            ...

        def valid(self):
            ...
        
    zz = ZZ({'a':3,'b':4}, 1, 1, 1, 1, 'zz')
    print(zz)
    '''
    recorder = BestRecorder(['xx', 'yy', 'zz'], ['max', 'min', 'max'])
    recorder.cal_best(0, [123, 456, 789])
    recorder.cal_best(1, [-789, 123, 7348])
    recorder.cal_best(2, [346, -274, 739])
    print(recorder)