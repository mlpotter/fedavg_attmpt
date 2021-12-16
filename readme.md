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

## arguments for script
```optional arguments:
  -h, --help            show this help message and exit 

  --world_size WORLD_SIZE
                        The world size which is equal to 1 server + (world size - 1) clients
  --epochs EPOCHS       The number of epochs to run on the client training each iteration
  --iterations ITERATIONS
                        The number of iterations to communication between clients and server
  --batch_size BATCH_SIZE
                        The batch size during the epoch training
  --partition_alpha PARTITION_ALPHA
                        Number to describe the uniformity during sampling (heterogenous data generation for LDA)
```

Will add more notes when ironed out , and understand `torch.distributed.rpc` better.