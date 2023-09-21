"""
author: Chuanpu Li
date: 2023_09_21 11:40
"""

import os
import numpy as np
import torch
import argparse
import PBTC_TransNet
import json
from get_ROI import get_rois_and_mask

parser = argparse.ArgumentParser()
# base
parser.add_argument('--gpu', type=str, default='6', help='which gpu is used')
parser.add_argument('--model_dir', type=str, default='', help='trained model dir')
parser.add_argument('--model_mode', type=str, default='best_val_AUC')
parser.add_argument('--suffix', type=str, default='', help='suffix in the save dir')
parser.add_argument('--dataset_type', type=str, default='test')
parser.add_argument('--fold_num', type=int, default=4, help='0-4')

# data type
parser.add_argument('--num_cls', type=int, default=3, help='number of classes wanted to be classify')
parser.add_argument('--ct_box', type=tuple, default=(12, 96, 96), help='data resize size')
parser.add_argument('--t1_box', type=tuple, default=(12, 96, 96), help='data resize size')
parser.add_argument('--t2_box', type=tuple, default=(12, 96, 96), help='data resize size')
parser.add_argument('--x_box', type=tuple, default=(1, 512, 384), help='data resize size')

# model parameter
parser.add_argument('--basic_dim', type=int, default=32, help='number of channels in the first layer')
parser.add_argument('--trans_dim', type=int, default=512, help='transformer basic dim')
parser.add_argument('--mlp_dim', type=int, default=4096, help='mlp dim in transformer')
parser.add_argument('--intra_depth', type=int, default=1, help='number of transformer block in intra modality')
parser.add_argument('--inter_depth', type=int, default=1, help='number of transformer block among inter modality')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate in the final dropout layer')
parser.add_argument('--clinical_dims', type=int, default=13, help='how many clinical info will be input to the model' )

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
patient = '/home/lcp/data/bonetumor/final/second_full_modality/0000'      # patient dir: including patient.json(clinical info), and the imgs' roi

model = PBTC_TransNet.Model(num_classes=args.num_cls, basic_dim=args.basic_dim,
                            transformer_basic_dim=args.trans_dim,
                            mlp_dim=args.mlp_dim, inter_depth=args.inter_depth,
                            dropout_rate=args.dropout_rate, clinical_dims=args.clinical_dims)

info = json.load(open(os.path.join(patient, patient.split('/')[-1] + '.json')), encoding='utf-8')
ID = patient.split('/')[-1]
label = info['良中恶性三分类']
clinical_info = torch.tensor([info['性别'], float(info['年龄']), info['部位类别'], info['病理性骨折'], info['白细胞是否正常'], info['血红蛋白是否正常'],
     info['碱性磷酸酶是否正常'], info['充血'], info['肿胀'], info['发热'], info['触痛'],
     info['运动障碍'], info['可触及包块']]).unsqueeze(0)

# print(ID)
rois, mask = get_rois_and_mask(name=patient, ct_box=args.ct_box, t1_box=args.t1_box, t2_box=args.t2_box, x_box=args.x_box)

ct_roi = rois['ct'].cuda()
mr_roi = rois['mr'].cuda()
x_roi = rois['x'].cuda()
clinical_info = clinical_info.cuda()
mask = mask.cuda()

model.module.is_training = False
model.eval()
with torch.no_grad():
    fuse_pred, final_outputs = model(ct_roi, mr_roi, x_roi, clinical_info, mask)  # fuse_pred: 没有经过softmax, final_output: 经过softmax
    final_outputs = final_outputs.data.cpu()
    final_outputs = torch.squeeze(final_outputs).numpy()
    final_pred = np.argmax(final_outputs)









