# ConvMAE: Masked Convolution Meets Masked Autoencoders


[[`arXiv`](https://arxiv.org/abs/2205.03892)]

This repository contains the implementation of the ConvMAE transfer learning for object detection on COCO.

For ImageNet pretraining and pretrained checkpoint, please refer to [ConvMAE](https://github.com/Alpha-VL/ConvMAE).


### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">pre-train</th>
<th valign="bottom">pre-train<br/>epoch</th>
<th valign="bottom">finetune<br/>epoch</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">mask<br/>AP</th>
<th valign="bottom">model</th>
<th valign="bottom">log</th>
<!-- TABLE BODY -->
<!-- ROW: mask_rcnn_vitdet_b_100ep -->
 <tr><td align="center">ViTDet, ViT-B</td>
<td align="center">IN1K, MAE</td>
<td align="center">1600</td>
<td align="center">100</td>
<td align="center">51.6</td>
<td align="center">45.9</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>
<!-- ROW: mask_rcnn_vitdet_l_100ep -->
 <tr><td align="center"><a href="projects/ConvMAEDet/configs/COCO/mask_rcnn_vitdet_convmae_b_25ep.py">ViTDet, ConvMAE-B</a></td>
<td align="center">IN1K, ConvMAE</td>
<td align="center">1600</td>
<td align="center">25</td>
<td align="center">53.9</td>
<td align="center">47.6</td>
<td align="center"><a href="https://drive.google.com/file/d/1YAnoopUpLSorn9ugq8WGfPyhIDcFouTI/view?usp=sharing">model</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1DccgEmvEQs6i_ZVGZESngIARznJVFDLY/view?usp=sharing">log</a></td>
</tr>
</tbody></table>


</tr>
</tbody></table>


## Installation

Please follow [Installation](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to install detectron2.

## Preparing Dataset

```
cd datasets
ln -s /path/to/coco coco
```

## Training
```
python tools/lazyconfig_train_net.py --num-gpus 8 --config-file \ 
projects/ConvMAEDet/configs/COCO/mask_rcnn_vitdet_convmae_b_25ep.py \
train.init_checkpoint=path/to/pretrained_model
```

## Evaluation
```
python tools/lazyconfig_train_net.py --num-gpus 8 --eval-only --config-file \ 
projects/ConvMAEDet/configs/COCO/mask_rcnn_vitdet_convmae_b_25ep.py \
train.init_checkpoint=path/to/model_checkpoint
```

## Acknowledgement
This project is based on [Detectron2](https://github.com/facebookresearch/detectron2) and [VitDet](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet). Thanks for their wonderful work.

## Citation
```
@article{gao2022convmae,
  title={ConvMAE: Masked Convolution Meets Masked Autoencoders},
  author={Gao, Peng and Ma, Teli and Li, Hongsheng and Dai, Jifeng and Qiao, Yu},
  journal={arXiv preprint arXiv:2205.03892},
  year={2022}
}
```
