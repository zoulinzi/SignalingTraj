# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim
# from torch.nn.parallel import DistributedDataParallel as DDP
# from tqdm import tqdm
# from torch_geometric.datasets import Planetoid
# import torch_geometric.transforms as T
# from torch_geometric.nn import GCNConv, GATConv
# from torch_geometric.utils import train_test_split_edges
# from torch_geometric.nn import VGAE
# import torch.multiprocessing as mp
# import numpy as np
# from torch_geometric.data import Data
# import os


# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)


# def cleanup():
#     dist.destroy_process_group()


# def decode(model, epochs, out_channels, data, device):
#     num_features = data.num_node_features
#     x = data.x.to(device)
#     edge_index = data.edge_index.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     model.train()
#     loss = 0
#     for epoch in tqdm(range(1, epochs + 1)):  # train
#         all_loss = []
#         optimizer.zero_grad()
#         z = model.module.encode(x, edge_index)
#         recon_loss = model.module.recon_loss(z, edge_index)
#         loss = recon_loss + 20 * (1 / data.num_nodes) * model.module.kl_loss()
#         all_loss.append(loss)
#         loss.backward()
#         if (epoch % 10 == 0):
#             print(f"Rank {rank}: epoch is {epoch} and loss is {loss}")
#         optimizer.step()
#         torch.cuda.empty_cache()  # 释放 GPU 缓存
#     model.eval()
#     with torch.no_grad():  # pred
#         z = model.module.encode(x, edge_index)
#         print(f"Rank {rank}: z", z)
#         print(f"Rank {rank}: z.shape", z.shape)
#     return z


# class VGAE_gat(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, heads):
#         super(VGAE_gat, self).__init__()
#         self.conv1 = GATConv(in_channels, out_channels, heads[0])
#         self.conv2 = GATConv(heads[0] * out_channels, out_channels, heads[1])
#         self.conv_mu = GATConv(heads[1] * out_channels, out_channels, heads[2], concat=False)
#         self.conv_logstd = GATConv(heads[1] * out_channels, out_channels, heads[2], concat=False)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.conv2(x, edge_index)
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# def main(rank, world_size):
#     setup(rank, world_size)
#     device = torch.device(f'cuda:{rank}')
#     torch.cuda.set_device(device)

#     # 数据为边和每条边的特征
#     with open("./edges_index_length_25_60.txt") as f:
#         edges1 = [list(map(int, [line.split()[0], line.split()[1]])) for line in f.readlines()]
#         edges2 = [list(map(int, [line.split()[1], line.split()[0]])) for line in f.readlines()]
#         edges = edges1 + edges2
#     edges = torch.tensor(edges, dtype=torch.long)
#     edges = edges.transpose(0, 1)

#     lac_features = np.load('./lac_onehot.npy')
#     out_dims = 64
#     data = Data(x=torch.tensor(lac_features, dtype=torch.float), edge_index=edges)
#     num_features = data.num_node_features

#     model = VGAE(VGAE_gat(num_features, out_dims, [2, 2, 1]))
#     model = model.to(device)
#     model = DDP(model, device_ids=[rank])

#     z_gat = decode(model, 100, out_dims, data, device)
#     if rank == 0:
#         np.save('./embedding', z_gat.cpu().numpy())


# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import numpy as np
# from tqdm import tqdm
# from torch_geometric.nn import GATConv, VGAE
# from torch_geometric.data import Data, DataLoader
# from torch.nn.parallel import DistributedDataParallel as DDP
# import os


# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)


# def cleanup():
#     dist.destroy_process_group()


# class VGAE_gat(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, heads):
#         super(VGAE_gat, self).__init__()
#         self.conv1 = GATConv(in_channels, out_channels, heads[0])
#         self.conv2 = GATConv(heads[0] * out_channels, out_channels, heads[1])
#         self.conv_mu = GATConv(heads[1] * out_channels, out_channels, heads[2], concat=False)
#         self.conv_logstd = GATConv(heads[1] * out_channels, out_channels, heads[2], concat=False)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = self.conv2(x, edge_index)
#         return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# def decode(model, epochs, data_loader, device, rank):
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     model.train()
#     for epoch in tqdm(range(1, epochs + 1)):  # train
#         for batch in data_loader:  # mini-batch over edges
#             optimizer.zero_grad()
#             x = batch.x.to(device)
#             edge_index = batch.edge_index.to(device)
#             z = model.module.encode(x, edge_index)
#             recon_loss = model.module.recon_loss(z, edge_index)
#             loss = recon_loss + 20 * (1 / batch.num_nodes) * model.module.kl_loss()
#             loss.backward()
#             optimizer.step()
#             torch.cuda.empty_cache()  # 清理显存

#             if epoch % 10 == 0 and rank == 0:
#                 print(f"Rank {rank}: Epoch {epoch}, Loss: {loss.item()}")

#     model.eval()
#     z = None
#     with torch.no_grad():  # Inference
#         for batch in data_loader:
#             x = batch.x.to(device)
#             edge_index = batch.edge_index.to(device)
#             z = model.module.encode(x, edge_index)
#     return z


# def main(rank, world_size):
#     setup(rank, world_size)
#     device = torch.device(f'cuda:{rank}')
#     torch.cuda.set_device(device)

#     # 数据加载
#     with open("./edges_index_length_25_60.txt") as f:
#         edges1 = [list(map(int, [line.split()[0], line.split()[1]])) for line in f.readlines()]
#         edges2 = [list(map(int, [line.split()[1], line.split()[0]])) for line in f.readlines()]
#         edges = edges1 + edges2
#     edges = torch.tensor(edges, dtype=torch.long).t()

#     # 加载节点特征
#     lac_features = np.load('./lac_onehot.npy')
#     out_dims = 32  # 调整为 64
#     data = Data(x=torch.tensor(lac_features, dtype=torch.float), edge_index=edges)

#     # 将数据切分为 mini-batch
#     batch_size = 1000  # 根据显存大小调整 batch size
#     data_loader = DataLoader([data], batch_size=batch_size, shuffle=True)

#     # 模型初始化
#     num_features = data.num_node_features
#     model = VGAE(VGAE_gat(num_features, out_dims, [2, 2, 1]))
#     model = model.to(device)
#     model = DDP(model, device_ids=[rank])

#     # 模型训练
#     z_gat = decode(model, epochs=50, data_loader=data_loader, device=device, rank=rank)

#     # 保存嵌入
#     if rank == 0:
#         np.save('./embedding', z_gat.cpu().numpy())

#     cleanup()


# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)



# original code
import torch
from tqdm import tqdm
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import VGAE

import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def decode(model,epochs,out_channels,data,device):
    num_features = data.num_node_features
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    loss = 0
    for epoch in tqdm(range(1, epochs + 1)): # train
        all_loss = []
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        recon_loss = model.recon_loss(z, edge_index) # how to calculate the loss
        #print("recon loss is ",float(loss))
        loss = recon_loss +20* (1 / data.num_nodes) * model.kl_loss()  # new line
        #print("all loss is ",float(loss))
        all_loss.append(loss)
        loss.backward()
        if(epoch%10==0):
            print("epoch is {} and loss is {}".format(epoch,loss))
        optimizer.step()
    model.eval()
    with torch.no_grad(): # pred
        z= model.encode(x, edge_index)
        print("z",z)
        print("z.shape",z.shape) # torch.Size([42372, 128])
    return z

class VGAE_gat(torch.nn.Module):
    def __init__(self, in_channels, out_channels,heads):
        super(VGAE_gat, self).__init__()      
        self.conv1 = GATConv(in_channels, out_channels,heads[0]) # cached only for transductive learning
        self.conv2 = GATConv(heads[0]*out_channels, out_channels,heads[1]) # cached only for transductive learning
        self.conv_mu = GATConv(heads[1]* out_channels, out_channels,heads[2],concat = False)
        self.conv_logstd = GATConv(heads[1]* out_channels, out_channels, heads[2],concat =False)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

# 数据为边和每条边的特征
# with open("base_station_info/mobile_trans_202209.txt") as f:
# with open("./preprocess_data/control_distribution_25_60/embedding/edges_index_length_25_60.txt") as f:
with open("./edges_index_length_25_60.txt") as f:
    edges1 = [list(map(int,[line.split()[0],line.split()[1]])) for line in f.readlines()]
    edges2 = [list(map(int,[line.split()[1],line.split()[0]])) for line in f.readlines()]
    edges = edges1+edges2
    # print(len(edges))
    # print((edges1))

edges = torch.tensor(edges,dtype=torch.long)
print(edges.shape)
edges = edges.transpose(0,1)
print(edges.shape) # torch.Size([2, 189299])
 # ['a','b','c'] 
from torch_geometric.data import Data
import numpy as np
# lac_features = np.load('./preprocess_data/control_distribution_25_60/embedding/lac_onehot.npy')
lac_features = np.load('./lac_onehot.npy') # (42374, 246)
# glove_features = np.load('base_station_info/vgae_embedding_results/glove_features.npy')

# features = np.concatenate([lac_features,glove_features],axis=1) # 列concat

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

out_dims = 64
data = Data(x= torch.tensor(lac_features,dtype=torch.float),edge_index=edges) # 修改其中的features

# print(data.edge_index.to(device),'data.edge_index')
# print(max(data.edge_index.to(device)[0]))
# print(max(data.edge_index.to(device)[1]))

num_features = data.num_node_features

model = VGAE(VGAE_gat(num_features, out_dims,[2,2,1]))  # new line 修改其中的num_features
model = model.to(device)
z_gat= decode(model,100,out_dims,data,device) # 修改其中的data
    
np.save('./embedding_npy2txt/embedding', z_gat)
