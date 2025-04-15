import numpy as np
import os
import shutil
import argparse
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# load argvalues
parser = argparse.ArgumentParser()
parser.add_argument('--output_dim', type=int, default=128) 
parser.add_argument('--input_dim', type=int, default=512)
parser.add_argument('--dataset', type=str, default='ffhq')
parser.add_argument('--mode', type=str, nargs='+', default=['train', 'test'])
parser.add_argument('--setting', type=str, help="'s' for same R and 'd' for different R", default = 'd') 
parser.add_argument('--known_r', type=str2bool, help="r is known (yes) or unknown (no)", default = True) 
parser.add_argument('--seed', type=int, help="seed value", default = 1740470466) 

dataset = parser.parse_args().dataset
input_folder = f'datasets/{dataset}_pp'
mode = parser.parse_args().mode
input_dim = parser.parse_args().input_dim
output_dim = parser.parse_args().output_dim
setting = parser.parse_args().setting
known_r = parser.parse_args().known_r
seed = parser.parse_args().seed
folders = ['images', 'embeddings']

suffixes = f"{setting}_{known_r}"
print(dataset, mode, suffixes)


# Create output folders
output_folder = f'datasets/mrp/{dataset}_{output_dim}_{suffixes}'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Initialize seed
np.random.seed(seed)


# Save seed for randomness
if setting == 's':
    R = np.random.randn(output_dim, input_dim)
    np.save(f'{output_folder}/R_{seed}.npy', R)
else:
    np.save(f'{output_folder}/seed_{seed}.npy', seed)

# generate the data
print("Start datagen")
start_time = time.time()
for m in mode:
    for f in folders:
        if f == 'images':
            if os.path.exists(os.path.join(output_folder, m, f)):
                continue
            os.makedirs(os.path.join(output_folder, m, f))
            # copy the images folder to the output folder
            files = os.listdir(os.path.join(input_folder, m, f))
            files.sort()
            if m == 'test':
                files = files[:1000]
            for file in files:
                shutil.copy(os.path.join(input_folder, m, f, file), os.path.join(output_folder, m, f, file))
            continue 

        if not os.path.exists(os.path.join(output_folder, m, f)):
            os.makedirs(os.path.join(output_folder, m, f))
        # load the embeddings
        embedding_list = os.listdir(os.path.join(input_folder, m, f))
        embedding_list.sort()
        if m == 'test':
            embedding_list = embedding_list[:1000] # first thousand only
        i =0
        # project x to eigen value space and run attack procedure
        for e in embedding_list:
            i+=1
            if i % 1000 == 0:
                print(f"Processing {i} embeddings")
            x = np.load(os.path.join(input_folder, m, f, e)).flatten()
            if setting == 'd':
                R = np.random.randn(output_dim, input_dim)
            v = np.matmul(R, x)
            v /= np.sqrt(output_dim)
            if known_r:
                w_inv = np.matmul(np.linalg.pinv(R), v)
                w_inv *= np.sqrt(output_dim)
                np.save(os.path.join(output_folder, m, f, e), w_inv)
                continue
            np.save(os.path.join(output_folder, m, f, e), v)

end_time = time.time()
time_elapsed = end_time - start_time
print(f"Time elapsed for datagen {time_elapsed/60:.2f} min, or {time_elapsed/3600:.2f} hours. " )