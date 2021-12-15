# FedAVG attempt using PyTorch and `torch.distributed.rpc`

## client.py
* client.py contains the `client` object
* logs client actions and metrics to `client{rank}.log`

## server.py
* server.py contains the `server` object
* logs server actions and metrics to `server{rank}.log`

## data/iris_data_generator.py
* generates non-iid dataloader for each client as `.pt` file. Loaded onto the `client` when initialized.
* Latent Dirichlet Allocation sampling from `FedML` repo

Run `python fedavg.py` in terminal to simulate FedAVG with 2 clients and 1 server for the Iris dataset.

Will add more notes when ironed out , and understand `torch.distributed.rpc` better.