import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self, reduction='mean', norm='L2'):
        super().__init__()
        self.reduction = reduction
        self.norm = norm.upper()

    def forward(self, inputs, targets):

        if self.norm == "L2":
            loss = ((inputs - targets) ** 2).mean(dim=1)
        else:
            loss = (inputs - targets).mean(dim=1)

        if self.reduction == 'mean':
            outputs = torch.mean(loss)
        elif self.reduction == 'sum':
            outputs = torch.sum(loss)
        else:
            outputs = loss

        return outputs

class CosineLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.crit = nn.CosineSimilarity(dim=1)

    def forward(self, x, y):
        loss = 1 - self.crit(x, y)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = loss
        return loss



class Incremental_base_loss:
    '''
    Args:
        Prototypes loss for background.
        :param old_feat: [batch_size, feat_dim, N]
        :param new_feat: [batch_size, feat_dim, N]
        :param label: [batch_size, feat_dim, N]

    Returns:
        :return: loss
    '''
    def __init__(self, step, mode):
        '''
        :param step: current step
        :param mode: choose loss function. {cosine, MSELoss, }
        '''
        self.step = step
        self.step_classes = {0: [1, 2, 3, 4, 5, 6],
                             1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                             2: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}

        if mode == 'cosine':
            self.loss_kd_prototype = CosineLoss(reduction='mean')
        elif mode == 'MSELoss':
            self.loss_kd_prototype = MSELoss(reduction='mean')


    def __call__(self, new_feat, old_feat, label):
        B, C, N = new_feat.shape
        label_ = label.view(-1, N)
        loss_incremental_proto = 0
        old_class = self.step_classes[self.step - 1]
        old_proto_list = []
        new_proto_list = []
        y_label_ = torch.unique(label_)

        for old_class_idx in y_label_:
            if old_class_idx in old_class:
                mask_idx = label_ == old_class_idx
                old_proto = old_feat.transpose(1, 2)[mask_idx].mean(dim=0, keepdim=True)
                new_proto = new_feat.transpose(1, 2)[mask_idx].mean(dim=0, keepdim=True)
                old_proto_list.append(old_proto)
                new_proto_list.append(new_proto)
            else:
                continue
        old_prototypes = torch.stack(old_proto_list, dim=0).squeeze(dim=1)
        new_prototypes = torch.stack(new_proto_list, dim=0).squeeze(dim=1)
        loss_incremental_proto += self.loss_kd_prototype(new_prototypes, old_prototypes)

        return loss_incremental_proto