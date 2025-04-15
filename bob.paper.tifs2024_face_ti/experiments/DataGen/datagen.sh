# Process the CelebA dataset using 

python GenDataset.py --prot_scheme unprotected --same_r none --data_dir celeba_pp --mode train
python GenDataset.py --prot_scheme unprotected --same_r none --data_dir celeba_pp --mode test

python GenDataset.py --prot_scheme mrp --same_r true --data_dir celeba_pp --mode train
python GenDataset.py --prot_scheme mrp --same_r true --data_dir celeba_pp --mode test

python GenDataset.py --prot_scheme mrp --same_r false --data_dir celeba_pp --mode train
python GenDataset.py --prot_scheme mrp --same_r false --data_dir celeba_pp --mode test

python GenDataset.py --prot_scheme fe --same_r true --data_dir celeba_pp --mode train
python GenDataset.py --prot_scheme fe --same_r true --data_dir celeba_pp --mode test

python GenDataset.py --prot_scheme fe --same_r false --data_dir celeba_pp --mode train
python GenDataset.py --prot_scheme fe --same_r false --data_dir celeba_pp --mode test
