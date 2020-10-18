"""
File with definitions of components, such as model parts and pipelines
"""
import pandas as pd
import numpy as np
import torch
from torch import nn

class WindowGenerator():
    def __init__(self, input_width, target_width, 
                 train_df, val_df, test_df,
                 ignore_columns=None,
                 target_columns=None, 
                 encoder_columns=None, 
                 decoder_columns=None):

        self.columns = sorted(list(set(train_df.columns) - set(ignore_columns)))
    
        self.data = {'train' : train_df[self.columns], 
                     'val' : val_df[self.columns], 
                     'test' : test_df[self.columns], 
                     'custom' : None}

        self.tgt_cols = {name: i for i, name in enumerate(self.data['train'].columns) if name in target_columns}
        self.enc_cols = {name: i for i, name in enumerate(self.data['train'].columns) if name in encoder_columns}
        self.dec_cols = {name: i for i, name in enumerate(self.data['train'].columns) if name in decoder_columns}
        
        self.tgt_col_idxs = [i for (name, i) in self.tgt_cols.items()]
        self.enc_col_idxs = [i for (name, i) in self.enc_cols.items()]
        self.dec_col_idxs = [i for (name, i) in self.dec_cols.items()]
        
        self.input_width = input_width
        self.target_width = target_width
        self.window_width = input_width + target_width
        
        self.batch_size = 1
        
    def __repr__(self):
        return '\n'.join([
            f'Encoder columns : {[name for name in self.enc_cols]}',
            f'Decoder columns : {[name for name in self.dec_cols]}',
            f'Target columns : {[name for name in self.tgt_cols]}',
            f'Input size: {self.input_width}',
            f'Target size: {self.target_width}'])
    
    
    def batch(self, idx, source='train'):
        """
        Batch contains window_width points in X. The target columns are shifted by 1,
        so that every day we know today's features and yesterdays target features.
        """
        # For external batching, for example random sampling
        start = max(idx, 0)
            
        batch_size = min(self.batch_size, len(self.data[source]) - start - self.window_width)
                
        end = start + self.window_width + batch_size + 1
        
        data = self.data[source].iloc[start:end, :]
        
        # Data, including target for the same day, for past "input_width days"
        enc = []
        dec = [] # Today's features + all yesterdays data
        tgt = [] # Today's targets
        
        for i in range(batch_size):
            enc.append(data.iloc[i:self.input_width+i, self.enc_col_idxs].values)
            tgt.append(data.iloc[self.input_width+i:self.window_width+i, self.tgt_col_idxs].values)
            dec_batch = data.iloc[self.input_width+i:self.window_width+i, self.dec_col_idxs].values 
            
            #tgt_shift = data.iloc[self.input_width+i-1:self.window_width+i-1, self.tgt_col_idxs].values
            dec.append(dec_batch)#dec.append(np.concatenate((tgt_shift, dec_batch), axis = 1))
        
        return np.array(enc), np.array(dec), np.array(tgt)
    
    
    def gen_batches(self, batch_size = 1, random = False, source = "train"):
        """
        Return batch generator. Setting random to True gets random positions of batches.
        Withing the batch windows are moved forward by 1 time step. Batches do not overlap.
        
        TODO: generate walk forward split train-valid batches.
        """
        start = 0
        end = len(self.data[source]) - self.window_width
        
        num_wind = batch_size
        self.batch_size = batch_size
        
        start_ids = np.array([i for i in range(start, end, num_wind)])

        if random: 
            np.random.shuffle(start_ids)
            
            
        return start_ids
    
    
    def number_of_batches(self, batch_size, source = 'train'):    
        """
        Number of batches.
        """
        self.batch_size = batch_size
        num_batches = len(self.data[source] - self.window_width) // self.batch_size
        rem = len(self.data[source] - self.window_width) % self.batch_size
            
        if rem == 0:
            return num_batches
        else:
            return num_batches + 1
        
        
    def prep_extern_data(self, df, set_source = 'custom'):
        """
        Set df as a custom datasource. Use source = 'custom' in functions to access it.
        """
        self.data[set_source] = df[self.columns]
        
        
        
class Encoder(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers=1, droupout=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            num_layers=num_layers,
            input_size=num_features,
            hidden_size=hidden_size,
            batch_first=True,
            dropout=droupout)
        
    def forward(self, x, hidden):
        # x should have a type torch.FloatTensor
        output, hidden = self.rnn(x, hidden)
        
        return output, hidden.squeeze(0)
    
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden
    
    
    
class Decoder(nn.Module):
    def __init__(self, num_features, hidden_size, droupout=0.2):
        super(Decoder, self).__init__()
        self.rnn_cell = nn.GRUCell(
            input_size = num_features,
            hidden_size = hidden_size
        )
        self.l1 = nn.Linear(hidden_size, 1)
        self.d1 = nn.Dropout(droupout)
        
    def forward(self, x, hidden):
        hidden = self.rnn_cell(x, hidden)
        hidden = self.d1(hidden)
        output = self.l1(hidden)
        
        return output, hidden
    
    
    
    
def save_checkpoint(path, model, valid_loss):
    torch.save({'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}, path)

    
def load_checkpoint(path, model, device):    
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    
    return state_dict['valid_loss']


def save_metrics(path, train_loss_list, valid_loss_list, global_steps_list):   
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, path)


def load_metrics(path, device):    
    state_dict = torch.load(path, map_location=device)
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']