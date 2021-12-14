# FedAVG attempt using PyTorch and `torch.distributed.rpc`

## client.py
* client.py contains the `client` object

## server.py
* server.py contains the `server` object

## data/iris_data_generator.py
* generates non-iid dataloader for each client as `.pt` file. Loaded onto the `client` when initialized.
* Latent Dirichlet Allocation sampling from `FedML` repo

Will add more notes when ironed out , and understand `torch.distributed.rpc` better.