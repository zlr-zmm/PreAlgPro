import torch
import os

from bio_embeddings.embed import ESM1bEmbedder, PLUSRNNEmbedder, BeplerEmbedder

# CLS averaging processing function
def getCls(vector):
    vector = vector.mean(axis=0)
    return vector


# CLS data generation and writing functions
def data_write(input_data, output_file_name, embedder):
    k=0
    for i in input_data:
        print(k)
        print(i[0])
        k = k+1
        embedding = embedder.embed(i[0])
        cls = getCls(embedding)
        # print(cls)
        if not os.path.exists(output_file_name):
            os.system(r"touch {}".format(output_file_name))
        with open(output_file_name, 'a') as f:
            a = []
            for j in cls:
                a.append(float(j))
            f.write(str(a) + " ")
            f.write("\n")

import pandas as pd
# dataset reading
path_to_train = "dataset/case_data.csv"
path_to_train_cls1 = "dataset/Alg_real_Data_PLUSRNNEmbedder.csv"
dataset_train = []
train_datasets = pd.read_csv(path_to_train).iloc[:,:].values.tolist()
print(len(train_datasets))
data_write(train_datasets, path_to_train_cls1, PLUSRNNEmbedder())
path_to_train_cls2 = "dataset/Alg_real_Data_ESM1bEmbedder.csv"
data_write(train_datasets, path_to_train_cls2, ESM1bEmbedder())
path_to_train_cls3 = "dataset/Alg_real_Data_BeplerEmbedder.csv"
data_write(train_datasets, path_to_train_cls3, BeplerEmbedder())