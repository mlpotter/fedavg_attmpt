from client import client
from server import server
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from model import *
import os

def init_env():
    print("Initialize Meetup Spot")
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = "5682"

def example(rank,world_size):
    init_env()
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    if rank == 0:
        Server = server(LogisticRegression(4,3),
                        rank,
                        world_size)

        Server.update_clients()

        for iter in range(1000):
            Server.train()
            Server.update_clients()
            Server.evaluate()
        rpc.shutdown()
    else:
        print("Client")
        rpc.shutdown()

def main():

    world_size = 3
    mp.spawn(example,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()