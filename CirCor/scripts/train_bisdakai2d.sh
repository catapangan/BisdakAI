#!/bin/sh
python train.py --epochs 200 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule_kws20.yaml --model ai85bisdakai2dnet --dataset CirCor2D --save-sample 10 --confusion --device MAX78000 "$@"
