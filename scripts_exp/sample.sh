# example shown for lfw, facenet, MRP, diff seed per user
# reconstruct 1000 images from embeddings for train and test set

SAMPLE_FLAGS="--image_size 64 --attention_resolutions 32,16,8  --dropout 0.1 --class_cond True --learn_sigma True --num_channels 192 --num_heads 3 --num_res_blocks 3 --resblock_updown True --use_fp16 True --use_new_attention_order True --use_scale_shift_norm True --diffusion_steps 1000 --noise_schedule cosine" 
TEST_FLAGS="--batch_size 100 --num_samples 1000 --timestep_respacing 250"


echo "sampling lfw"
python3 scripts/image_sample.py $SAMPLE_FLAGS $TEST_FLAGS --model_path models/mrp/lfw_facenet_128_d_True/model025000.pt --checkpoint_dir results/mrp/lfw_facenet_128_d_True --embedding_size 128 --embeddings_path datasets/mrp/lfw_facenet_128_d_True/train/embeddings --eval_dset train --device_id 0 &
python3 scripts/image_sample.py $SAMPLE_FLAGS $TEST_FLAGS --model_path models/mrp/lfw_facenet_128_d_True/model025000.pt --checkpoint_dir results/mrp/lfw_facenet_128_d_True --embedding_size 128 --embeddings_path datasets/mrp/lfw_facenet_128_d_True/test/embeddings --eval_dset test --device_id 1 


echo "sampling done"

