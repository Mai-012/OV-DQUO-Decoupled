#!/bin/bash
set -e

PRETRAIN_CKPT=${1:-"ckpt/OVDQUO_RN50_COCO.pth"}
EPOCHS=5
LR=2e-5

echo "========================================================"
echo "1/3 启动 Ablation A：仅开启【模块一：去偏未知挖掘 + 动态背景 + CEL去偏】"
echo "========================================================"
python main.py \
    --output_dir work_dirs/ablation_a_module1 \
    -c config/OV_COCO/OVDQUO_RN50.py \
    --amp \
    --device cuda:0 \
    --resume ${PRETRAIN_CKPT} \
    --finetune \
    --eval_start_epoch 1 \
    --options epochs=${EPOCHS} lr=${LR} lr_drop=4 use_module1=True use_module2=False

echo "========================================================"
echo "2/3 启动 Ablation B：仅开启【模块二：新类锚点区域袋 + 掩码过滤 + BCE/InfoNCE蒸馏】"
echo "========================================================"
python main.py \
    --output_dir work_dirs/ablation_b_module2 \
    -c config/OV_COCO/OVDQUO_RN50.py \
    --amp \
    --device cuda:0 \
    --resume ${PRETRAIN_CKPT} \
    --finetune \
    --eval_start_epoch 1 \
    --options epochs=${EPOCHS} lr=${LR} lr_drop=4 use_module1=False use_module2=True

echo "========================================================"
echo "3/3 启动 Full Model：完整双轨闭环【模块一 + 模块二】"
echo "========================================================"
python main.py \
    --output_dir work_dirs/full_module1_module2 \
    -c config/OV_COCO/OVDQUO_RN50.py \
    --amp \
    --device cuda:0 \
    --resume ${PRETRAIN_CKPT} \
    --finetune \
    --eval_start_epoch 1 \
    --options epochs=${EPOCHS} lr=${LR} lr_drop=4 use_module1=True use_module2=True

echo " 全部三组消融训练与评测完成！结果保存在 work_dirs/ 下各自目录的 log.txt 中。"
