from relational_table_preprocessor import relational_table_preprocess_dl
import pandas as pd
import numpy as np
import torch
from collections import Counter

class args(object):
    def __init__(self):
        pass

iris_df = pd.read_csv("bezdekIris.data", header=None)
iris_df.loc[:,4] = iris_df.loc[:,4].map({"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2})
label_list = iris_df.pop(4).tolist()
data = iris_df.values

print(iris_df.head())

args.client_num_in_total = 2
args.class_num = 3
args.partition_alpha = 0.5
args.batch_size=16

[_, _, _, _,_, train_data_local_dict, test_data_local_dict, args.class_num] = relational_table_preprocess_dl(args,
                                                                                                             data,
                                                                                                             label_list,
                                                                                                             test_partition=0.2)
for key in train_data_local_dict.keys():
    torch.save(train_data_local_dict[key], f"data_worker{key+1}_train.pt")
    print(dict(Counter(train_data_local_dict[key].dataset[:][1].numpy().tolist())))
    torch.save(test_data_local_dict[key], f"data_worker{key+1}_test.pt")