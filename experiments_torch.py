try:
    from comet_ml import Experiment
    USE_COMET = True
except ImportError:
    USE_COMET = False

import numpy as np
from artificial_data import permuted_pattern, permuted_pattern_consecutive
from chrono_layers_torch import LayerSumSimple, eprint, cumsum, cummax
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from prettytable import PrettyTable
import torchmetrics
import operator
import datetime

try:
    import seaborn as sns; sns.set()
except ImportError:
    pass
from matplotlib import pyplot as plt

class CNNNet(nn.Module):
    def __init__(self, N=200, flip=True):
        super().__init__()
        self.flip = flip
        self.N = N
        self.conv1 = nn.Conv1d(1, 11, 3, stride=1)
        self.pool = nn.MaxPool1d(N-2)
        self.slicer = Slicer()
        self.dense  = nn.Linear(11,2)

    def forward(self, x):
        if self.flip:
            x = torch.transpose(x,-2,-1)
        x = F.relu(self.conv1(x))
        x = self.pool( x )
        x = x[:,:,-1]
        x = self.dense( x )
        return x

def get_model_CNN(IN_SHAPE=(200,1), OUT_DIM=2, flip=True):
    return CNNNet(N=IN_SHAPE[0],flip=flip)


def get_model_FCN(IN_SHAPE=(200,1), OUT_DIM=2):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(IN_SHAPE[0],40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, OUT_DIM),
        )
    return model


class Slicer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:,-1,:]

def get_model_ISS_fns(IN_SHAPE=(200,1), OUT_DIM=2, k_fn=3, dim_fn=[5,5,11]):
    fns = [] 
    for k in range(k_fn):
        tmp = [nn.Linear(IN_SHAPE[-1],dim_fn[0]), nn.ReLU() ]
        for i in range(1,len(dim_fn)):
            tmp.append( nn.Linear(dim_fn[i-1], dim_fn[i]) )
            tmp.append( nn.ReLU() )
        seq = nn.Sequential( *tmp )
        fns.append(seq)

    ls     = LayerSumSimple(nn.ModuleList(fns))
    slicer = Slicer()
    dense  = nn.Linear(dim_fn[-1],2)
    model  = nn.Sequential(ls, slicer, dense)
    return model

def experiment_pattern(epochs=100):
    sigma = 1.
    N = 50
    x_train, y_train = permuted_pattern_consecutive(N, 1000, pattern=[2,-3,16],sigma=sigma)
    x_val, y_val     = permuted_pattern_consecutive(N,  100, pattern=[2,-3,16],sigma=sigma)
    x_test, y_test   = permuted_pattern_consecutive(N, 1000, pattern=[2,-3,16],sigma=sigma)
    model = get_model_ISS_fns(IN_SHAPE=x_train.shape[1:])
    #model = get_model_FCN(IN_SHAPE=x_train.shape[1:])
    #model = get_model_CNN(IN_SHAPE=x_train.shape[1:])
    ret = train_evaluate(comet_experiment(), model,x_train,y_train,x_test,y_test,x_val=x_val,y_val=y_val,epochs=epochs)
    return ret

project_name = f"tss-{datetime.datetime.now()}"
def comet_experiment(parameters_to_log=None):
    if USE_COMET:
        experiment = Experiment(
            #api_key="",                 Is read from environment variable COMET_API_KEY
            project_name=project_name
        )
        if parameters_to_log:
            experiment.log_parameters( parameters_to_log )
    else:
        from contextlib import contextmanager
        class DummyExperiment:

            @contextmanager
            def train(self):
                yield None

            @contextmanager
            def test(self):
                yield None

            def set_epoch(self,_):
                pass

            def log_metrics(self,m):
                print(m)
            
            def log_text(self,t):
                print(t)

            log_curve = None

        experiment = DummyExperiment()
        if parameters_to_log:
            print( parameters_to_log )
    return experiment

def experiment_parameter_search(experiment_factory=comet_experiment):
    epochs  = 100
    #pattern = [2,16]
    pattern = [2,-3,16]

    #for sigma in [0.01,0.1,0.3,0.7,1.0,3.0]:
    #    for N in [10,30,70,100,300,700,1000,3000]:
    #        x_train, y_train = permuted_pattern_consecutive(N, 1000, pattern=pattern,sigma=sigma)
    #        x_val, y_val     = permuted_pattern_consecutive(N,  100, pattern=pattern,sigma=sigma)
    #        x_test, y_test   = permuted_pattern_consecutive(N, 1000, pattern=pattern,sigma=sigma)
    #        train_evaluate(comet_experiment({'sigma':sigma,'N':N, 'type':'FCN'}), get_model_FCN(IN_SHAPE=x_train.shape[1:]),x_train,y_train,x_test,y_test,x_val=x_val,y_val=y_val)
    #        train_evaluate(comet_experiment({'sigma':sigma,'N':N, 'type':'CNN'}), get_model_CNN(IN_SHAPE=x_train.shape[1:]),x_train,y_train,x_test,y_test,x_val=x_val,y_val=y_val)

    for sigma in [0.01,0.1,0.3,0.7,1.0,3.0]:
        for N in [10,30,70,100,300,700,1000,3000]:
            x_train, y_train = permuted_pattern(N, 1000, pattern=pattern,sigma=sigma)
            x_val, y_val     = permuted_pattern(N,  100, pattern=pattern,sigma=sigma)
            x_test, y_test   = permuted_pattern(N, 1000, pattern=pattern,sigma=sigma)
            train_evaluate(comet_experiment({'sigma':sigma,'N':N, 'type':'FCN-NC'}), get_model_FCN(IN_SHAPE=x_train.shape[1:]),    x_train,y_train,x_test,y_test,x_val=x_val,y_val=y_val,epochs=epochs)
            #train_evaluate(comet_experiment({'sigma':sigma,'N':N, 'type':'CNN-NC'}), get_model_CNN(IN_SHAPE=x_train.shape[1:]),    x_train,y_train,x_test,y_test,x_val=x_val,y_val=y_val,epochs=epochs)
            for dim_fn in [ [5,5,11], [5,11], [3,3,3] ]:
                train_evaluate(comet_experiment({'sigma':sigma,'N':N, 'type':'ISS-NC', 'dim_fn':dim_fn}), get_model_ISS_fns(IN_SHAPE=x_train.shape[1:]), x_train,y_train,x_test,y_test,x_val=x_val,y_val=y_val,epochs=epochs)


def train_loop(batch, model, optim, loss_fn):
    for (X, y) in batch:
        pred = model(X)
        loss = loss_fn(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

def pretty_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    #print(table)
    #print(f"Total Trainable Params: {total_params}")
    #return total_params
    return f"{table}\nTotal Trainable Params: {total_params}"

def train_evaluate(experiment, model, x_train, y_train, x_test, y_test, x_val=None, y_val=None, epochs=100, batch_size=16):
    with experiment.train():
        if USE_COMET:
            ts = list(range(len(x_train[0])))
            for i in range(10):
                plt.bar(ts, x_train[i,:,0], label='class=0')
                plt.legend()
                plt.savefig('tmp.png')
                plt.clf()
                experiment.log_image('tmp.png', name=f"class=0_example_{i}")

                plt.bar(ts, x_train[-i,:,0], label='class=1')
                plt.legend()
                plt.savefig('tmp.png')
                plt.clf()
                experiment.log_image('tmp.png', name=f"class=1_example_{i}")

        x_train, y_train = torch.tensor(x_train,dtype=torch.float32), torch.tensor(y_train,dtype=torch.long)
        x_test, y_test = torch.tensor(x_test,dtype=torch.float32), torch.tensor(y_test,dtype=torch.long)
        x_val, y_val = torch.tensor(x_test,dtype=torch.float32), torch.tensor(y_test,dtype=torch.long)

        d_train = TensorDataset(x_train, y_train)
        d_test = TensorDataset(x_test, y_test)
        d_val = TensorDataset(x_val, y_val)

        X_train = DataLoader(d_train, batch_size=batch_size, shuffle=True)
        X_test = DataLoader(d_test, batch_size=batch_size, shuffle=True)
        X_val = DataLoader(d_val, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        metric = torchmetrics.Accuracy()

        experiment.log_text( pretty_parameters(model) )

        for e in range(epochs):
            experiment.set_epoch( e )
            model.train()
            train_loop(X_train, model, optim, loss_fn)
            model.eval()
            m = metric(model(x_train), y_train)
            validation_results = metric(model(x_val), y_val)
            print(f"epoch: {e+1}, metric: {m}, validation: {validation_results}")
            experiment.log_metrics({'epoch_metric': m, 'epoch_validation': validation_results})

    with experiment.test():
        results = metric(model(x_test), y_test)
        print(f"test acc: {results}, {results.shape}")
        experiment.log_metrics({'accuracy': results})
    return results

if __name__ == '__main__':
    #experiment_pattern()
    experiment_parameter_search()
