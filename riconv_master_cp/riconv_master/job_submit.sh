#!/bin/sh

#PBS -q inferno -A GT-ywang93 -l mem=32gb -l walltime=12:00:00 -l nodes=1:ppn=8:gpus=1:exclusive_process  

#conda activate /storage/home/hcoda1/5/zwang945/p-ywang93-0/tensorflow-1.13.1-env
conda init /storage/home/hcoda1/5/zwang945/p-ywang93-0/tensorflow-1.13.1-env

echo $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

python3 train_val_cls.py



