import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(preds, labels):
    lossfn = nn.BCELoss()
    predss = torch.sigmoid(preds)
    loss = lossfn(predss, labels)
    return loss


def cal_hr_loss(Y, adj_mx, num_category):
    loss = 0
    sim_ls = torch.zeros([num_category])
    for i in range(num_category):
        Y_i = Y[:, i]
        sum_sim = 0
        for region_id, label in enumerate(Y_i):
            region_neighbor = adj_mx[region_id]
            sum_sim += torch.sum(region_neighbor[Y_i == label])
        sim_ls[i] = sum_sim
    h_ls = torch.zeros([num_category])
    for i in range(num_category):
        h_ls[i] = sim_ls[i]/torch.sum(torch.sum(adj_mx))
    e = torch.ones([num_category])
    diff = h_ls - e
    diff_2 = torch.mul(diff,diff)
    loss = torch.sum(diff_2)
    return loss
