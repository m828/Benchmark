# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved. 
#
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#
#     http://www.apache.org/licenses/LICENSE-2.0    
#
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import paddle
import paddle.nn as nn
import typing
from .workspace import register, create


__all__ = ['YOLOv5']    

@register
class BaseArch(nn.Layer):
    def __init__(self, data_format='NCHW', use_extra_data=False):
        super(BaseArch, self).__init__()
        self.data_format = data_format
        self.inputs = {}
        self.fuse_norm = False
        self.use_extra_data = use_extra_data

    def load_meanstd(self, cfg_transform):
        scale = 1.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        for item in cfg_transform:
            if 'NormalizeImage' in item:
                mean = np.array(
                    item['NormalizeImage']['mean'], dtype=np.float32)
                std = np.array(item['NormalizeImage']['std'], dtype=np.float32)
                if item['NormalizeImage'].get('is_scale', True):
                    scale = 1. / 255.
                break
        if self.data_format == 'NHWC':
            self.scale = paddle.to_tensor(scale / std).reshape((1, 1, 1, 3))
            self.bias = paddle.to_tensor(-mean / std).reshape((1, 1, 1, 3))
        else:
            self.scale = paddle.to_tensor(scale / std).reshape((1, 3, 1, 1))
            self.bias = paddle.to_tensor(-mean / std).reshape((1, 3, 1, 1))

    def forward(self, inputs):
        if self.data_format == 'NHWC':
            image = inputs['image']
            inputs['image'] = paddle.transpose(image, [0, 2, 3, 1])

        if self.fuse_norm:
            image = inputs['image']
            self.inputs['image'] = image * self.scale + self.bias
            self.inputs['im_shape'] = inputs['im_shape']
            self.inputs['scale_factor'] = inputs['scale_factor']
        else:
            self.inputs = inputs

        self.model_arch()

        if self.training:
            out = self.get_loss()
        else:
            inputs_list = []
            # multi-scale input
            if not isinstance(inputs, typing.Sequence):
                inputs_list.append(inputs)
            else:
                inputs_list.extend(inputs)
            outs = []
            for inp in inputs_list:
                if self.fuse_norm:
                    self.inputs['image'] = inp['image'] * self.scale + self.bias
                    self.inputs['im_shape'] = inp['im_shape']
                    self.inputs['scale_factor'] = inp['scale_factor']
                else:
                    self.inputs = inp
                outs.append(self.get_pred())

            # multi-scale test
            if len(outs) > 1:
                out = self.merge_multi_scale_predictions(outs)
            else:
                out = outs[0]
        return out
    
@register
class YOLOv5(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['post_process']
    __shared__ = ['with_mask']

    def __init__(self,
                 backbone='CSPDarkNet',
                 neck='YOLOCSPPAN',
                 yolo_head='YOLOv5Head',
                 post_process='BBoxPostProcess',
                 with_mask=False,
                 for_mot=False):
        """
        YOLOv5, YOLOv6(https://arxiv.org/abs/2209.02976) and YOLOv7(https://arxiv.org/abs/2207.02696)

        Args:
            backbone (nn.Layer): backbone instance
            neck (nn.Layer): neck instance
            yolo_head (nn.Layer): anchor_head instance
            for_mot (bool): whether return other features for multi-object tracking
                models, default False in pure object detection models.
        """
        super(YOLOv5, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.yolo_head = yolo_head
        self.post_process = post_process
        self.for_mot = for_mot
        self.with_mask = with_mask

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        backbone = create(cfg['backbone'])

        # fpn
        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        # head
        kwargs = {'input_shape': neck.out_shape}
        yolo_head = create(cfg['yolo_head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "yolo_head": yolo_head,
        }

    def _forward(self):
        body_feats = self.backbone(self.inputs)
        neck_feats = self.neck(body_feats, self.for_mot)

        if self.training:
            yolo_losses = self.yolo_head(neck_feats, self.inputs)
            return yolo_losses
        else:
            yolo_head_outs = self.yolo_head(neck_feats)
            post_outs = self.yolo_head.post_process(
                yolo_head_outs,
                im_shape=self.inputs['im_shape'],
                scale_factor=self.inputs['scale_factor'],
                infer_shape=self.inputs['image'].shape[2:])

            if not isinstance(post_outs, (tuple, list)):
                # if set exclude_post_process, concat([pred_bboxes, pred_scores]) not scaled to origin
                # export onnx as torch yolo models
                return post_outs
            else:
                if not self.with_mask:
                    # if set exclude_nms, [pred_bboxes, pred_scores] scaled to origin
                    bbox, bbox_num = post_outs  # default for end-to-end eval/infer
                    output = {'bbox': bbox, 'bbox_num': bbox_num}
                else:
                    bbox, bbox_num, mask = post_outs  # default for end-to-end eval/infer
                    output = {'bbox': bbox, 'bbox_num': bbox_num, 'mask': mask}
                    # Note: YOLOv5 Ins models don't support exclude_post_process or exclude_nms
                return output

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()