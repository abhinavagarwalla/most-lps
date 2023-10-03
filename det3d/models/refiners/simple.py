from torch.nn import functional as F
from ..registry import REFINERS
from torch import gt, nn
import numpy as np
import torch
from ..utils import build_norm_layer
from ..losses import segmentation_losses


@REFINERS.register_module
class SimplePointVoxelBEV(nn.Module):
    def __init__(self, in_channels, num_classes, norm_cfg=None):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.model = nn.Sequential(
            nn.Linear(in_channels[0], in_channels[1]),
            build_norm_layer(norm_cfg, in_channels[1])[1],
            nn.ReLU(inplace=True),

            nn.Linear(in_channels[1], num_classes)
        )

    def forward(self, example):
        
        points_coor = example['points_coor']

        #can use bilinear interpolation, but simplicity for now. #by 4 as conv3 #adding 1 as first index spconv=batch index
        points_coor[:, 1:] = (points_coor[:, 1:]//8)
        points_coor[:, 1] = points_coor[:, 1] + 1

        points_coor = torch.clamp(points_coor, max=179)
        
        bev_feat = example['bev_feature'][points_coor[:, 0], :, points_coor[:, 2], points_coor[:, 3]]

        x = torch.cat([example['points_feature'][:, 1:], bev_feat], dim=1)
        return {"sem_preds": self.model(x)}
    
    def loss(self, example, preds_dicts, test_cfg, **kwargs):
        device = preds_dicts['sem_preds'].device
        gts = []
        for b_i in range(len(example['pt_sem_labs'])):
            gt = example['pt_sem_labs'][b_i][example['pt_sem_labs'][b_i]!=-1]
            valid_mask = example['valid_index_mask'][b_i][example['base_index_mask'][b_i]==1]
            gts.append(torch.tensor(gt[valid_mask.astype(bool)], device=device).long())

        gt = torch.cat(gts, dim=0)
        assert(-1 not in torch.unique(gt))
        loss = F.cross_entropy(preds_dicts['sem_preds'], gt, reduction='mean')
        return {'loss': loss}
    
    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        #now distribute the predictions to all points. clipped or not..
        device = preds_dicts['sem_preds'].device
        preds_list = []
        ind = 0
        for b_i in range(len(example['points'])):
            pred = torch.zeros((example['base_index_mask'][b_i]==1).sum(), device=device).long()
            valid_mask = example['valid_index_mask'][b_i][example['base_index_mask'][b_i]==1]
            pred[valid_mask.astype(bool)] = preds_dicts['sem_preds'][ind: ind+valid_mask.sum()].argmax(axis=1)
            ind += valid_mask.sum()
            preds_list.append({'sem_preds_pt': pred})
        assert(ind == preds_dicts['sem_preds'].shape[0])
        return preds_list

@REFINERS.register_module
class VoxelPredRefiner(nn.Module):
    def __init__(self, in_channels, num_classes, norm_cfg=None):
        super().__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        self.model = nn.Sequential(
            nn.Linear(in_channels[0], in_channels[1]),
            build_norm_layer(norm_cfg, in_channels[1])[1],
            nn.ReLU(inplace=True),

            nn.Linear(in_channels[1], num_classes)
        )

    def forward(self, example):
        
        points_coor = example['points_coor']
        points_coor[:, 1:] = (points_coor[:, 1:].clone()//8)
        points_coor = torch.clamp(points_coor, max=179)
        bev_feat = example['bev_feature'][points_coor[:, 0], :, points_coor[:, 2], points_coor[:, 3]]

        x = torch.cat([example['points_feature'][:, 1:], bev_feat], dim=1)
        return {"sem_preds": self.model(x), "voxel_preds": example['voxel_feature'].dense()}
    
    def loss(self, example, preds_dicts, test_cfg, **kwargs):
        device = preds_dicts['sem_preds'].device
        gts = []
        for b_i in range(len(example['pt_sem_labs'])):
            gt = example['pt_sem_labs'][b_i][example['pt_sem_labs'][b_i]!=-1]
            gts.append(torch.tensor(gt, device=device).long())            

        gt = torch.cat(gts, dim=0)
        pt_loss = F.cross_entropy(preds_dicts['sem_preds'], gt, ignore_index=0, reduction='mean')

        voxel_gt = torch.tensor(example['voxel_sem_labs'], device=device).long()
        voxel_loss = F.cross_entropy(preds_dicts['voxel_preds'], voxel_gt, ignore_index=0, reduction='mean')
        
        loss = pt_loss + voxel_loss
        return {'loss': loss, 'voxel_loss': voxel_loss}
    
    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        #now distribute the predictions to all points. clipped or not..
        device = preds_dicts['sem_preds'].device
        preds_list = []
        ind = 0
        for b_i in range(len(example['points'])):
            pred = preds_dicts['sem_preds'].argmax(axis=1)
            preds_list.append({'sem_preds_pt': pred})
        return preds_list

@REFINERS.register_module
class VoxelPredRefinerGeo(VoxelPredRefiner):
    def loss(self, example, preds_dicts, test_cfg, **kwargs):
        device = preds_dicts['sem_preds'].device
        gts = []
        for b_i in range(len(example['pt_sem_labs'])):
            gt = example['pt_sem_labs'][b_i][example['pt_sem_labs'][b_i]!=-1]
            gts.append(torch.tensor(gt, device=device).long())            


        gt = torch.cat(gts, dim=0)
        pt_loss = F.cross_entropy(preds_dicts['sem_preds'], gt, ignore_index=0, reduction='mean')

        voxel_gt = torch.tensor(example['voxel_sem_labs'], device=device).long()
        voxel_loss = F.cross_entropy(preds_dicts['voxel_preds'], voxel_gt, ignore_index=0, reduction='mean')
        
        lovasz_loss = segmentation_losses.lovasz_softmax(F.softmax(preds_dicts['voxel_preds'], dim=1).detach(), voxel_gt, ignore=0)
        geo_loss = segmentation_losses.geo_loss(voxel_gt, preds_dicts['voxel_preds'], ignore_label=0)


        loss = pt_loss + voxel_loss + lovasz_loss + geo_loss
        return {'loss': loss, 'voxel_loss': voxel_loss}

@REFINERS.register_module
class JustVoxel(nn.Module):
    def __init__(self, in_channels, num_classes, norm_cfg=None):
        super().__init__()

    def forward(self, example):
        
        return {"voxel_preds": example['voxel_feature'].dense()}
    
    def loss(self, example, preds_dicts, test_cfg, **kwargs):
        device = preds_dicts['voxel_preds'].device

        voxel_gt = torch.tensor(example['voxel_sem_labs'], device=device).long()
        voxel_loss = F.cross_entropy(preds_dicts['voxel_preds'], voxel_gt, ignore_index=0, reduction='mean')

        loss = voxel_loss
        return {'loss':loss, 'voxel_loss': voxel_loss}
    
    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        device = preds_dicts['voxel_preds'].device
        preds_list = []
        ind = 0
        for b_i in range(len(example['points'])):
            pred = torch.zeros((example['base_index_mask'][b_i]==1).sum(), device=device).long()
            pred = preds_dicts['sem_preds'][ind: ind + pred.sum()].argmax(axis=1)
            ind += pred.sum()
            preds_list.append({'sem_preds_pt': pred})
        assert(ind == preds_dicts['sem_preds'].shape[0])
        return preds_list
