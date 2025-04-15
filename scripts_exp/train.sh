# !/bin/bash

# --lr_anneal_steps is 25000 for LFW, 50000 for CelebA and 100000 for FFHQ
# possible suffixes: _d_False, _d_True, _s_False, _s_True. d referes to different seed and s refers to same seed. False means no known_r and True means known_r.
# we can use '_facenet' or '_arcface' to specify the model type.
# we can use '_128' or '_256' to specify the output embedding size (MRP) OR size(vector b)/2 (LWE-FE).

# example shown for lfw, facenet, MRP, diff seed per user

MODEL_FLAGS="--image_size 64 --num_channels 192 --num_heads 3 --num_res_blocks 3 --attention_resolutions 32,16,8  --dropout 0.1 --class_cond True --learn_sigma True --resblock_updown True --use_fp16 True --use_new_attention_order True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 64 --ema_rate 0.9999 --lr_anneal_steps 25000 --save_interval 25000"

#baseline
# python3 scripts/image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --data_dir datasets/mrp/lfw_facenet_128_d_False/train --checkpoint_dir models/mrp/lfw_arcface_128_d_False --embedding_size 128 --device_id 0

# pipe
python3 scripts/image_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS --data_dir datasets/mrp/lfw_facenet_128_d_True/train --checkpoint_dir models/mrp/lfw_arcface_128_d_True --embedding_size 512 --device_id 0

echo "training done"

