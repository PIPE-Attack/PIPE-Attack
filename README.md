# PIPE-Attack

This repository contains code for the PIPE attack used for inversion of protected embeddings. It also has code for prior attacks such as BOB, KED-MI, and GMI for comparison. This repository is forked from [openai/guided-diffusion](https://github.com/openai/guided-diffusion), modified to use embeddings instead of class as the condition vector. 

## Code base

- `./guided_diffusion` and `./scripts` contains code for conditioned DDPM model
- `./scripts_datagen` contain python scripts and notebooks to prepare data for PIPE
- `./scripts_exp` contain bash scripts to run and evaluate PIPE attack experiments
- `./gan_attacks` contains code for prior attaks such as GMI and KED-MI
- `./bob` contains code for prior attacks sucha as BOB

We also include submodule facenet_pytorch, modified to output unnormalized embedding vectors instead of normalized embedding vectors. 

P.S. In our codebase, we use the keyword `sameR` to represent same seed settings.

## Requirements

We create two separate environments for running inversion attacks (`ddpm`) and evaluating ASR (`deepface`), and can be reproduced with the `environment_*.yml` files. 

The requirements for running attacks:

- blobfile>=1.0.5
- torch
- tqdm

The requirements for Deepface evaluation:
- deepface

## PIPE Attack

The exact commands and parameters we used to run PIPE are given in `scripts_exp`. Here we give an overview of how the scripts are used. 

To generate data, 

- First run the relevant `scripts/*_pp.ipynb` files to preprocess image datasets.
- Then run relevant `scripts/datagen_*.py` files to generate (un)protected image embeddings

To run PIPE attack, 

- Run `scripts/image_train.py` to train the attack model. 
- Run `scripts/image_sample.py` to run the attack model and reconstruct images.
- Run `evaluation.py` to compute ASR of the attack. 

After running the PIPE experiment, results will be stored in the following locations:
- `./models` contains trained diffusion model files
- `./results` contains evaluation results

## Prior attacks

### GAN-based attacks

GMI and KED-MI attacks are provided under `./gan_attack`. This directory is forked from [AlanPeng0897/Defend_MI](https://github.com/AlanPeng0897/Defend_MI/), only includes GMI and DMI attack code for CelebA dataset, modified to run on protected embeddings as well.

The pretrained GMI GAN model (used for unprotected embeddings) is downloaded according to [README](./gan_attacks/README.md) and placed under `GMI/result/models_celeba_gan`.

Our preparations are documented under the `preparation` directory, please run those first. The commands used to run and evaluate the attacks on unprotected and protected embeddings are listed under the `scripts_exp` directory. The ASR values are stored in `asr` and you can load them with `load_asr.ipynb`. 


### BOB attack

The BOB attack is provided under `./bob.paper.tifs2024_face_ti`. This directory is a modified version of `bob.paper.tifs2024_face_ti` project present in the software package: signal-processing and machine learning toolbox [Bob](https://www.idiap.ch/software/bob).

To run the attack, kindly refer to [README](./bob.paper.tifs2024_face_ti/README.md).