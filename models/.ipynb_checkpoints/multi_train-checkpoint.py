import numpy as np
import random

import torch
import torch.nn as nn
from tqdm import tqdm
from models.model_utils import touli, get_constraint_mask
from models.loss_fn import cal_id_acc, check_rn_dis_loss, cal_id_acc_train
from models.trajectory_graph import build_graph,search_road_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('multi_task device', device)


def init_weights(self):
    """
    Here we reproduce Keras default initialization weights for consistency with Keras version
    Reference: https://github.com/vonfeng/DeepMove/blob/master/codes/model.py
    """
    ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
    hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
    b = (param.data for name, param in self.named_parameters() if 'bias' in name)

    for t in ih:
        nn.init.xavier_uniform_(t)
    for t in hh:
        nn.init.orthogonal_(t)
    for t in b:
        nn.init.constant_(t, 0)

def next_batch(indexes):
    length = len(indexes)
    for i in range(length):
        yield indexes[i]
        
def train(model, spatial_A_trans, SE, all_index_seqs, optimizer, log_vars, parameters, diffusion_hyperparams):
# def train(model, spatial_A_trans, SE, all_eids, all_rates, optimizer, log_vars, parameters, diffusion_hyperparams):
    model.train()  # not necessary to have this line but it's safe to use model.train() to train model

    criterion_reg = nn.MSELoss()
    criterion_ce = nn.NLLLoss()
    criterion_ce1 = nn.NLLLoss()
    epoch_ttl_loss = 0
    epoch_const_loss = 0
    epoch_diff_loss = 0
    epoch_x0_loss = 0
    
    all_length = []
    shuffle_all_index_seqs = {}
    all_batch_index = []
    for k, index_seqs_group in all_index_seqs.items():
        all_length.append(k)
        traj_num = len(index_seqs_group) // parameters.batch_size
        for i in range(traj_num):
            all_batch_index.append(all_index_seqs[k][i*parameters.batch_size:(i+1)*parameters.batch_size])
        if traj_num * parameters.batch_size != len(index_seqs_group):
            all_batch_index.append(all_index_seqs[k][traj_num*parameters.batch_size:])
    
    
    # 打乱数据
    # zipped_list = list(all_batch_index)
    # random.shuffle(zipped_list)
    # all_batch_index = zip(*zipped_list)

    random.shuffle(all_batch_index)


    SE = SE.to(device)
    
    cnt = 0
    # for ids, rates in next_batch(all_batch_id, all_batch_rate):
    for indexs in next_batch(all_batch_index):
        cnt += 1
        import time
        
        src_index_seqs = torch.tensor(indexs).long().to(device)
        
        spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
        curr_time = time.time()
        optimizer.zero_grad()
        diff_loss, const_loss, x0_loss = model(spatial_A_trans, SE, src_index_seqs)  # T, B, id_size   and    T, B, 1
        
        ttl_loss = diff_loss + const_loss + x0_loss 

        curr_time = time.time()
        ttl_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), parameters.clip)  # log_vars are not necessary to clip
        optimizer.step()
        
        epoch_ttl_loss += ttl_loss.item()
        epoch_const_loss += const_loss.item()
        epoch_diff_loss += diff_loss.item()
        epoch_x0_loss += x0_loss.item()
        
    return log_vars, epoch_ttl_loss / cnt, epoch_const_loss / cnt, epoch_diff_loss / cnt, epoch_x0_loss / cnt


def generate_data(model, spatial_A_trans, index_uli_dict,index_areacode_dict, parameters, SE):
    model.eval()  
    SE = SE.to(device)
    spatial_A_trans = torch.tensor(spatial_A_trans, dtype=torch.float).to(device)
    if parameters.dataset == "Chengdu":
        length2num = np.load("/data/WeiTongLong/data/traj_gen/A_new_dataset/Chengdu/gen_all/length_distri.npy")
        start_length, end_length = 20, 60
    if parameters.dataset.startswith("operator"):
        length2num = np.load(f"/root/autodl-tmp/data/{parameters.dataset}/gen_all/length_distri.npy")
        start_length, end_length = 20, 80
    # batchsize=1
    traj_all_num = int(length2num.sum())
    # batchsize=1
    with torch.no_grad():  # this line can help speed up evaluation
        for i in tqdm(range(start_length, end_length+1)):
            curr_length =i
            curr_batch = int(length2num[i] / length2num.sum() * traj_all_num)

            while True:
                if curr_batch > 256: 
                    _curr_batch = 256
                else:
                    _curr_batch = curr_batch
                ids = model.generate_data(spatial_A_trans, SE, _curr_batch, curr_length, parameters.pre_trained_dim)

                output_seqs = touli(index_uli_dict, index_areacode_dict,ids, parameters, save_txt_num = i)
                curr_batch = curr_batch - _curr_batch
                if curr_batch <= 0: break
                # exit()
            # print(output_seqs)
        # exit()

