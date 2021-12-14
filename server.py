import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
from torch.distributed.nn.api.remote_module import RemoteModule
from client import client

class server(object):
    def __init__(self,
                 model,
                 rank,
                 world_size):

        self.n_clients = world_size - 1
        self.model = model
        self.client_rrefs = []
        self.rank = rank
        self.world_size = world_size

        self.initialize_client_modules()

        print("Initialized Server")

    def initialize_client_modules(self):
        print("Initialize Clients")
        for rank in range(self.world_size-1):
            self.client_rrefs.append(
                                    rpc.remote(f"worker{rank+1}",
                                           client,
                                           args=(self.model,rank+1,self.world_size))
                                )

    def send_global_model(self):
       print("Sending Global Parameters")
       check_global = [client_rref.remote().load_global_model(self.model.state_dict().copy()) for client_rref in self.client_rrefs]
       for check in check_global:
           check.to_here()

    def train(self):
        print("Initializing Trainig")
        check_train = [client_rref.remote(timeout=0).train() for client_rref in self.client_rrefs]
        for check in check_train:
            check.to_here(timeout=0)

    def evaluate(self):
        print("Initializing Evaluation")
        total = []
        num_corr = []
        check_eval = [client_rref.remote(timeout=0).evaluate() for client_rref in self.client_rrefs]
        for check in check_eval:
            corr,tot = check.to_here(timeout=0)
            total.append(tot)
            num_corr.append(corr)

        print("Accuracy over all data: {:.3f}".format(sum(num_corr)/sum(total)))

    def aggregate(self):
        print("Aggregating Models")
        check_n_sample = [client_rref.rpc_async().send_num_train() for client_rref in self.client_rrefs]
        n_samples = [check.wait() for check in check_n_sample]
        n_total = sum(n_samples)

        check_params = [client_rref.rpc_async().send_local_model() for client_rref in self.client_rrefs]
        client_params = [check.wait() for check in check_params]

        global_model_state_dict = self.model.state_dict().copy()

        for name,param in self.model.named_parameters():
            global_model_state_dict[name] = torch.zeros_like(global_model_state_dict[name])
            for n_train,client_param in zip(n_samples,client_params):
                global_model_state_dict[name] = global_model_state_dict[name] + n_train/n_total * client_param[name]

        self.model.load_state_dict(global_model_state_dict)

    def update_clients(self):
        self.aggregate()
        self.send_global_model()