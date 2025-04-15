#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

EVAL_FLAGS="--metric euclidean_l2 --model Facenet512 --detector_backend mtcnn"

DIR_FLAGS="--output_dir asr --original_dir ../datasets/CelebA/Img/img_align_celeba_png --reconstructed_dir GMI/attack_res/celeba/reg"

python evaluation_asr.py --attack gmi --prot_scheme unprot --same_r d $EVAL_FLAGS $DIR_FLAGS
python evaluation_asr.py --attack gmi --prot_scheme mrp --same_r d $EVAL_FLAGS $DIR_FLAGS
python evaluation_asr.py --attack gmi --prot_scheme fe --same_r d $EVAL_FLAGS $DIR_FLAGS


DIR_FLAGS="--output_dir asr --original_dir ../datasets/CelebA/Img/img_align_celeba_png --reconstructed_dir DMI/attack_res/celeba/reg"
python evaluation_asr.py --attack dmi --prot_scheme unprot --same_r d $EVAL_FLAGS $DIR_FLAGS 
python evaluation_asr.py --attack dmi --prot_scheme mrp --same_r d $EVAL_FLAGS $DIR_FLAGS 
python evaluation_asr.py --attack dmi --prot_scheme fe --same_r d $EVAL_FLAGS $DIR_FLAGS 

