import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
from collections import Counter

class client(object):
    def __init__(self,
                 model,
                 rank,
                 world_size):
        self.rank = rank
        self.world_size = world_size
        self.load_data_local(f"data/data_worker{rank}_")
        self.model = model
        self.optimizer = optim.SGD(model.parameters(),
                                   lr=0.001)

        self.criterion = nn.CrossEntropyLoss()


        print(f"Initialized Client {rank}")

    def load_global_model(self,global_params):
        print(f"Client {self.rank} Loading Global Weights")
        self.model.load_state_dict(global_params)

    def send_local_model(self):
        print(f"Client {self.rank} Sending Local Weights")
        return self.model.state_dict()

    def send_num_train(self):
        return self.n_train

    def train(self):
        for epoch in range(3):  # loop over the dataset multiple times

            # running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # # print statistics
                # running_loss += loss.item()
                # if i % 2000 == 1999:  # print every 2000 mini-batches
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch + 1, i + 1, running_loss / 2000))
                #     running_loss = 0.0

        print('Finished Training')

    def evaluate(self):
        print(f"Client {self.rank} Evaluating Data")
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct,total

    def load_data_local(self,datapath):
        self.trainloader = torch.load(datapath+"train.pt")
        self.testloader = torch.load(datapath+"test.pt")

        self.n_train = len(self.trainloader.dataset)
        print("Local Data Statistics:")
        print("Dataset Size: {:.2f}".format(self.n_train))
        print(dict(Counter(self.trainloader.dataset[:][1].numpy().tolist())))