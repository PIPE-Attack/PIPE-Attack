python reconstruct_images.py \
    --original_dir DataGen/databases/celeba_pp \
    --generator_checkpoint 200 \
    --eval_dset train \
    --prot_scheme unprotected \
    --same_r none \
    --embedding_size 512 

# python reconstruct_images.py  --original_dir DataGen/databases/celeba_pp --generator_checkpoint 200  --eval_dset train --prot_scheme mrp --same_r true --embedding_size 128 
# python reconstruct_images.py  --original_dir DataGen/databases/celeba_pp --generator_checkpoint 200  --eval_dset train --prot_scheme mrp --same_r false --embedding_size 128 
# python reconstruct_images.py  --original_dir DataGen/databases/celeba_pp --generator_checkpoint 200  --eval_dset train --prot_scheme fe --same_r true --embedding_size 512 
# python reconstruct_images.py  --original_dir DataGen/databases/celeba_pp --generator_checkpoint 200  --eval_dset train --prot_scheme fe --same_r false --embedding_size 512 

# python reconstruct_images.py  --original_dir DataGen/databases/celeba_pp --generator_checkpoint 200  --eval_dset test --prot_scheme unprotected --same_r none --embedding_size 512 
# python reconstruct_images.py  --original_dir DataGen/databases/celeba_pp --generator_checkpoint 200  --eval_dset test --prot_scheme mrp --same_r true --embedding_size 128 
# python reconstruct_images.py  --original_dir DataGen/databases/celeba_pp --generator_checkpoint 200  --eval_dset test --prot_scheme mrp --same_r false --embedding_size 128 
# python reconstruct_images.py  --original_dir DataGen/databases/celeba_pp --generator_checkpoint 200  --eval_dset test --prot_scheme fe --same_r true --embedding_size 512 
# python reconstruct_images.py  --original_dir DataGen/databases/celeba_pp --generator_checkpoint 200  --eval_dset test --prot_scheme fe --same_r false --embedding_size 512 
