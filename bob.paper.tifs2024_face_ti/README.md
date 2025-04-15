This project is the modified version of the repository for paper: Vulnerability of State-of-the-Art Face Recognition Models to Template Inversion Attack

Kindly refer to README_base.md for the authors' README regarding this project


## Installation
Refere to README_base.md section: Installation

To create conda environment, you can also use `conda env create -f environment.yml`


## Before starting
Copy only the pre-processed 'images' of celebA dataset (obtained using datagen in PIPE) in the folder `experiments/DataGen/celeba_pp`.

Folder celeba_pp should have two subfolders: `experiments/DataGen/celeba_pp/train/images`, `experiments/DataGen/celeba_pp/test/images`.

We have already added and `id_map` and `img_id_map`, for denoting what is the identity of the user present in the give image. Generation of id_map can be seen in file `DataGen/celeba_pp/gen_map.ipynb`.

Copy `eig_vals_data_facenet`, `eig_vecs_data_facenet` from `helpers` directory present in PIPE Attack.

## Running BOB
### Step 1: Generating Training Dataset
Use `GenDataset.py` in `DataGen` folder to create embeddings for given image folder.
Train test data generation is separate commands specified using mode.
Follow `experiments/DataGen/datagen.sh` for syntax.

You can generate embeddings for following cases:
- unprotected: `--prot_scheme unprotected --same_r none`
- MRP protected with same s setting: `--prot_scheme mrp --same_r true`
- MRP protected with diff s setting: `--prot_scheme mrp --same_r false`
- LWE-FE protected with same s setting: `--prot_scheme fe --same_r true`
- LWE-FE protected with diff s setting: `--prot_scheme fe --same_r false`

For example:
```sh
cd experiments/DataGen
python GenDataset.py --prot_scheme unprotected --same_r none --data_dir celeba_pp --mode train
```

The images are stored in `experiments/DataGen/databases/celeba_pp`


### Step 2: Training Face reconstruction model
Next step is to train the reconstruction model

Follow `experiments/TrainNetwork/train.sh` for syntax.

```sh
cd experiments/TrainNetwork
python train.py --prot_scheme {prot_scheme} --same_r {same_r} --data_dir {celeba_pp full path}/{prot_scheme}_{same_r}
```

### Step 3: Sample reconstructed images
After training, before evaluation, obtain reconstructed images from embeddings. Primarily, you would like to look at the test set reconstruction.

Follow `experiments/sample.sh` for syntax.

```sh
cd experiments
python reconstruct_images.py \
    --original_dir DataGen/databases/celeba_pp \
    --generator_checkpoint 200 \
    --eval_dset test \
    --prot_scheme {prot_scheme} \
    --same_r {same_r} \
    --embedding_size {embedding size} 
```
embedding size is 512 in all cases except MRP protection scheme where it is 128.
Reconstructed images are stored in `experiments/DataGen/databases/celeba_pp/{prot_scheme}_{same_r}/test/reconstructed_images`, or in the train folder if eval_dset is train.

### Step 4: Evaluation
After obtaining reconstructed images, we compute ASR.

```sh
EVAL_FLAGS="--metric euclidean_l2 --model Facenet512 --detector_backend mtcnn --output_dir results --original_dir DataGen/databases/celeba_pp --samples 1000"
python evaluation.py $EVAL_FLAGS  --prot_scheme {prot_scheme} --same_r {same_r} --eval_dset test &
```

`results` stores the distance between original and reconstructed image embeddings and a plot depicting the same.