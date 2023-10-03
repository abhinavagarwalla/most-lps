from ..registry import DETECTORS
from .. import builder
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 

@DETECTORS.register_module
class EncoderDecoderNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck=None,
        bbox_head=None,
        refiner=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(EncoderDecoderNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, None
        )

        if refiner is not None:
            self.refiner = builder.build_refiner(refiner)

        self.init_weights(pretrained=pretrained)
        
    def extract_feat(self, data):
        if 'voxels' not in data:
            # extracting point features only for current frame and within range specified by detection
            output = self.reader(data['points'], deepcopy(data['base_index_mask']))
            voxels, coors, points_feat, points_coor, valid_index_mask, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels,
                points_feature=points_feat,
                points_coor=points_coor,
                valid_index_mask=valid_index_mask,
            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_features = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_features, data

    def combine_losses(self, bbox_loss, refinement_loss):
        bbox_loss['loss'][0] = bbox_loss['loss'][0] + refinement_loss['loss']
        return bbox_loss

    def combine_predictions(self, bbox_list, sem_pt_list):
        for i in range(len(bbox_list)):
            bbox_list[i]['sem_preds_pt'] = sem_pt_list[i]['sem_preds_pt']
        return bbox_list

    def forward(self, example, return_loss=True, **kwargs):
        x, voxel_features, voxel_layout = self.extract_feat(example)
        preds, bev_conv_features = self.bbox_head(x)

        example['points_feature'] = voxel_layout['points_feature']
        example['points_coor'] = voxel_layout['points_coor']
        example['valid_index_mask'] = voxel_layout['valid_index_mask']
        example['bev_feature'] = x #preds
        example['voxel_feature'] = voxel_features
        refined_sem_preds = self.refiner(example)
        
        if return_loss:
            bbox_loss = self.bbox_head.loss(example, preds, self.test_cfg)
            refinement_loss = self.refiner.loss(example, refined_sem_preds, self.test_cfg)
            return self.combine_losses(bbox_loss, refinement_loss)
        else:
            sem_preds_pt_list = self.refiner.predict(example, refined_sem_preds, self.test_cfg)
            box_preds_list = self.bbox_head.predict(example, preds, self.test_cfg)
            return self.combine_predictions(box_preds_list, sem_preds_pt_list)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None

@DETECTORS.register_module
class EncoderDecoderNetV2(EncoderDecoderNet):
        
    def extract_feat(self, data):
        if 'voxels' not in data:
            # extracting point features only for current frame and within range specified by detection
            output = self.reader(data['points'], data['base_index_mask'])
            voxels, coors, points_feat, points_coor, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels,
                points_feature=points_feat,
                points_coor=points_coor,
            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_features = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_features, data

    def combine_losses(self, bbox_loss, refinement_loss):
        bbox_loss['voxel_loss'] = []
        for i in range(len(bbox_loss['loss'])):
            bbox_loss['voxel_loss'].append(refinement_loss['voxel_loss'])
        bbox_loss['loss'].append(refinement_loss['loss'])
        return bbox_loss

    def combine_predictions(self, bbox_list, sem_pt_list):
        for i in range(len(bbox_list)):
            bbox_list[i]['sem_preds_pt'] = sem_pt_list[i]['sem_preds_pt']
        return bbox_list

    def vox_feature_to_pt_pred(self, example, voxel_logits, points_coor):

        #currently voxel is in by2 scale 
        points_coor[:, 1:] = (points_coor[:, 1:]//2)
        points_coor = torch.clamp(points_coor, max=719)
        points_coor[:, 1] = torch.clamp(points_coor[:, 1], max=19)
        voxel_preds = voxel_logits.argmax(dim=1)

        device = voxel_logits.device
        preds_list = []
        for b_i in range(len(example['points'])):
            pred = torch.zeros((example['base_index_mask'][b_i]==1).sum(), device=device).long()
            pred = voxel_preds[b_i, points_coor[:, 3], points_coor[:, 2], points_coor[:, 1]]
            preds_list.append({'sem_preds_pt': pred, 'metadata': example["metadata"][b_i]})
        return preds_list
            
    def forward(self, example, return_loss=True, **kwargs):
        x, voxel_features, voxel_layout = self.extract_feat(example)
        preds, bev_conv_features = self.bbox_head(x)

        example['points_feature'] = voxel_layout['points_feature']
        example['points_coor'] = deepcopy(voxel_layout['points_coor']) #refiner below modifies points_coor in_place
        example['bev_feature'] = x #preds
        example['voxel_feature'] = voxel_features
        refined_sem_preds = self.refiner(example)
        
        if return_loss:
            bbox_loss = self.bbox_head.loss(example, preds, self.test_cfg)
            refinement_loss = self.refiner.loss(example, refined_sem_preds, self.test_cfg)
            return self.combine_losses(bbox_loss, refinement_loss)
        else:
            sem_preds_pt_list = self.refiner.predict(example, refined_sem_preds, self.test_cfg)
            box_preds_list = self.bbox_head.predict(example, preds, self.test_cfg)
            vox_sem_preds_pt_list = self.vox_feature_to_pt_pred(example, refined_sem_preds["voxel_preds"], voxel_layout['points_coor'])
            return self.combine_predictions(box_preds_list, vox_sem_preds_pt_list)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 

@DETECTORS.register_module
class EncoderDecoderNetReduced(EncoderDecoderNet):
        
    def extract_feat(self, data):
        if 'voxels' not in data:
            output = self.reader(data['points'], data['base_index_mask'])
            voxels, coors, points_coor, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels,
                points_coor=points_coor,

            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x, voxel_features = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)

        return x, voxel_features, data

    def combine_losses(self, bbox_loss, refinement_loss):
        bbox_loss['voxel_loss'] = []
        for i in range(len(bbox_loss['loss'])):
            bbox_loss['voxel_loss'].append(refinement_loss['voxel_loss'])
        bbox_loss['loss'].append(refinement_loss['loss'])
        return bbox_loss

    def combine_predictions(self, bbox_list, sem_pt_list):
        for i in range(len(bbox_list)):
            bbox_list[i]['sem_preds_pt'] = sem_pt_list[i]['sem_preds_pt']
        return bbox_list

    def vox_feature_to_pt_pred(self, example, voxel_logits, points_coor):

        #currently voxel is in by2 scale 
        points_coor[:, 1:] = (points_coor[:, 1:]//2)
        points_coor = torch.clamp(points_coor, max=719)
        points_coor[:, 1] = torch.clamp(points_coor[:, 1], max=19)
        voxel_preds = voxel_logits.argmax(dim=1)

        device = voxel_logits.device
        preds_list = []
        for b_i in range(len(example['points'])):
            pred = torch.zeros((example['base_index_mask'][b_i]==1).sum(), device=device).long()
            pred = voxel_preds[b_i, points_coor[:, 3], points_coor[:, 2], points_coor[:, 1]]
            preds_list.append({'sem_preds_pt': pred, 'metadata': example["metadata"][b_i]})
        return preds_list
            
    def forward(self, example, return_loss=True, **kwargs):
        x, voxel_features, voxel_layout = self.extract_feat(example)
        preds, bev_conv_features = self.bbox_head(x)

        example['points_coor'] = deepcopy(voxel_layout['points_coor']) #refiner below modifies points_coor in_place
        example['bev_feature'] = x #preds
        example['voxel_feature'] = voxel_features
        refined_sem_preds = self.refiner(example)
        
        if return_loss:
            bbox_loss = self.bbox_head.loss(example, preds, self.test_cfg)
            refinement_loss = self.refiner.loss(example, refined_sem_preds, self.test_cfg)
            return self.combine_losses(bbox_loss, refinement_loss)
        else:
            box_preds_list = self.bbox_head.predict(example, preds, self.test_cfg)
            vox_sem_preds_pt_list = self.vox_feature_to_pt_pred(example, refined_sem_preds["voxel_preds"], voxel_layout['points_coor'])
            return self.combine_predictions(box_preds_list, vox_sem_preds_pt_list)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 
