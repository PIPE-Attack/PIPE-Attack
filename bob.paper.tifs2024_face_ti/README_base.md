# Vulnerability of State-of-the-Art Face Recognition Models to Template Inversion Attack
This package is part of the signal-processing and machine learning toolbox [Bob](https://www.idiap.ch/software/bob).
It contains the source code to reproduce the following paper:

```BibTeX
@article{tifs2024vulnerability,
  title={Vulnerability of State-of-the-Art Face Recognition Models to Template Inversion Attack},
  author={Shahreza, Hatef Otroshi and Hahn, Vedrana Krivoku{\'c}a and Marcel, S{\'e}bastien},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2024},
  publisher={IEEE}
}
```
## Installation
The installation instructions are based on [conda](https://conda.io/) and works on **Linux systems
only**. Therefore, please [install conda](https://conda.io/docs/install/quick.html#linux-miniconda-install) before continuing.

For installation, please download the source code of this paper and unpack it. Then, you can create a conda
environment with the following command:

```sh
$ cd bob.paper.tifs2024_face_ti

# create the environment
$ conda create --name bob.paper.tifs2024_face_ti --file package-list.txt
# or conda env create -f environment.yml

$ conda activate bob.paper.tifs2024_face_ti  # activate the environment
$ buildout
```

## Downloading the datasets
In our experiments, we use [FFHQ](https://github.com/NVlabs/ffhq-dataset) dataset for training our face reconstruction network.
Also we used [MOBIO](https://www.idiap.ch/dataset/mobio), [LFW](http://vis-www.cs.umass.edu/lfw/), and [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/) datasets for evaluation.
All of these datasets are publicly available. To download the datasets please refer to their websites:
- [FFHQ](https://github.com/NVlabs/ffhq-dataset)
- [MOBIO](https://www.idiap.ch/dataset/mobio)
- [LFW](http://vis-www.cs.umass.edu/lfw/)
- [AgeDB](https://ibug.doc.ic.ac.uk/resources/agedb/)


## Configuring the directories of the datasets
Now that you have downloaded the four databases. You need to set the paths to
those in the configuration files. [Bob](https://www.idiap.ch/software/bob) supports a configuration file
(`~/.bobrc`) in your home directory to specify where the
databases are located. Please specify the paths for the database like below:
```sh
# Setup FFHQ directory
$ bob config set  bob.db.ffhq.directory [YOUR_FFHQ_IMAGE_DIRECTORY]

# Setup MOBIO directories
$ bob config set  bob.db.mobio.directory [YOUR_MOBIO_IMAGE_DIRECTORY]
$ bob config set  bob.db.mobio.annotation_directory [YOUR_MOBIO_ANNOTATION_DIRECTORY]

# Setup LFW directories
$ bob config set  bob.db.lfw.directory [YOUR_LFW_IMAGE_DIRECTORY]
$ bob config set  bob.bio.face.lfw.annotation_directory [YOUR_LFW_ANNOTATION_DIRECTORY]

# Setup AgeDB directories
$ bob config set  bob.db.agedb.directory [YOUR_AGEDB_IMAGE_DIRECTORY]
```

## Running the Experiments
### Step 1: Generating Training Dataset
You can use the `GenDataset.py` in `DataGen` folder to generate training dataset for embeddings of a terget face recognition model. For example, for ArcFace you can use the following commands:
```sh
cd experiments/DataGen
python GenDataset.py --FR_system ArcFace
```
**Note:** If you want to use face recognition models from FaceXZoo, you need to run with `./bin/python`, where `bin` folder should be generated after `buildout` in in the installation step.

### Step 2: Training Face reconstruction model
After the data is generated, you can train the face reconstruction model by running `train.py` in `TrainNetwork` folder. For example, for ArcFace you can use the following commands:
```sh
cd experiments/TrainNetwork
python train.py  --FR_system ArcFace
```
### Step 3: Evaluation
After the model is  trained, you can use it to run evaluation.
For evaluation, you can use `eval` folder in the directory of each model and then open the folder of an evaluation dataset (MOBIO/LFW/AgeDB). Then, you can run the evaluation using `./run_pipeline.sh`. For example, for evaluation of ArcFace on MOBIO dataset, you can use the following commands:
```sh
cd experiments/eval/MOBIO
./run_pipeline.sh
```
You can change the face recognition in `./run_pipeline.sh` script.
After you ran the evaluation pipeline, you can use `eval_hist_plot.py`, `eval_SAR_plot.py`, `eval_SAR_values.py`  to find the score distributions and calculate the vulnerability in terms of Sucess Attack Rate (SAR).


## Contact
For questions or reporting issues to this software package, contact the first author or our
development [mailing list](https://www.idiap.ch/software/bob/discuss).