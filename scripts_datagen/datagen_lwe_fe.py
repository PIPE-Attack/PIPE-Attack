import numpy as np
import os
import shutil
import argparse
import time
from sympy import Matrix

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
parser.add_argument('--output_dim', type=int, default=128) # n (size of b)
parser.add_argument('--input_dim', type=int, default=512) # m (size of x)
parser.add_argument('--dataset', type=str, default='ffhq')
parser.add_argument('--mode', type=str, nargs='+', default=['train', 'test'])
parser.add_argument('--setting', type=str, help="'s' for same A,b and 'd' for different A,b", default = 'd') 
parser.add_argument('--known_r', type=str2bool, help="r is known (yes) or unknown (no)", default = True) 
parser.add_argument('--seed', type=int, help="seed value", default = 1740470466) 
parser.add_argument('--debug', type=str2bool, help="stores embeddings in original space", default = False)
parser.add_argument('--zq', type=str2bool, help="stores embeddings in eigen AND Z_q space", default = False)


dataset = parser.parse_args().dataset
input_folder = f'datasets/{dataset}_pp'
mode = parser.parse_args().mode
input_dim = parser.parse_args().input_dim
output_dim = parser.parse_args().output_dim
setting = parser.parse_args().setting
known_r = parser.parse_args().known_r
seed = parser.parse_args().seed
debug = parser.parse_args().debug
zq = parser.parse_args().zq
folders = ['images', 'embeddings']

suffixes = f"{setting}_{known_r}{'_debug' if debug else ''}{'_zq' if zq else ''}"
print(dataset, mode, suffixes)


# hardcoded information
model = dataset.split('_')[1]
eig_vecs = np.load(f"comp-FOWH/eig_vecs_data_{model}.npy") # note: generated using ffhq embeddings
eig_vals = np.load(f"comp-FOWH/eig_vals_data_{model}.npy")
k = 49 if model == 'facenet' else 51
print(f"Number of significant eigen values: {k}")
# assert(k==49)
eig_vecs_inv = np.linalg.inv(eig_vecs) # for debugging
min_proj = -20
max_proj = 20
q = 130003 #31013 # arbitrary prime number

# representation function
def zq_A(q, m, n):
    A = np.random.rand(m, n)
    return np.floor(A * q).astype(int) % q

def zq_b(q, n):
    b = np.random.rand(n)
    return np.floor(b * q).astype(int) % q

def zq_x(x, q):
    # almost all entries of x are from min_proj to max_proj
    x -= min_proj
    x /= (max_proj - min_proj)
    x *= q
    return np.floor(x).astype(int) % q

def zq_x_inv(x, q):
    # for debugging
    x = x.astype(float)
    x /= q
    x *= (max_proj - min_proj)
    x += min_proj
    return x


# functions

def eig_x(x, eig_vecs, k):
    x_proj = x@eig_vecs
    x_proj[k:] = 0 # set insignificant values = 0 (eig val < 1)
    return x_proj

def is_invertible(A, q):
    A_ = Matrix(A)
    try:
        A_inv = A_.inv_mod(q)
        return True
    except:
        return False


def modulo_inv(A, q):
    A_ = Matrix(A)
    A_inv = A_.inv_mod(q)
    A_inv = np.matrix(A_inv).astype(int)
    return A_inv

def gen(A, b, x, q):
    c = (((A@b )%q) + x)%q
    return c

def recover_b(A, b1, c, q):
    c1 = c[input_dim- output_dim//2:] # last n/2 elements
    A1 = A[input_dim- output_dim//2:, :output_dim//2]
    A2 = A[input_dim- output_dim//2:, output_dim//2:]
    
    A2_inv = modulo_inv(A2, q)
    b2_recover = (A2_inv@((c1 - (A1@b1)%q - q//2)%q)%q)%q
    b2_recover = np.array(b2_recover)[0]
    b2_recover = b2_recover.astype(int) %q
    b_recover = np.concatenate([b1, b2_recover])
    return b_recover

def recover_x(A, b, c, q):
    x = (c - (A@b)%q)%q
    return x

# Create output folders
output_folder = f'datasets/lwe-fe/{dataset}_{output_dim}_{k}_{suffixes}'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Initialize seed
np.random.seed(seed)


# Save seed for randomness
if setting == 's':
    A = zq_A(q, input_dim, output_dim)
    b = zq_b(q, output_dim)
    np.save(f'{output_folder}/A_{seed}.npy', A)
    np.save(f'{output_folder}/b_{seed}.npy', b)
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

        # project x to eigen value space and run attack procedure
        for e in embedding_list:
            x_original = np.load(os.path.join(input_folder, m, f, e)).flatten()
            x = eig_x(x_original, eig_vecs, k)
            x = zq_x(x, q) 
            if setting == 'd':
                A = zq_A(q, input_dim, output_dim)
                b = zq_b(q, output_dim)
            A2 = A[input_dim- output_dim//2:, output_dim//2:]

            while (not is_invertible(A2, q)):
                print("A2 not invertible, fetching new A")
                A = zq_A(q, input_dim, output_dim)
                b = zq_b(q, output_dim)
                A2 = A[input_dim- output_dim//2:, output_dim//2:]

            b1 = b[:output_dim//2]
            c = gen(A, b, x,q)

            if not known_r:
                c =  c.astype(float)   # default
                c /= q # store as vector with entries [0,1] for easy fp_16 usage
                np.save(os.path.join(output_folder, m, f, e), c)
                continue

            # given A, b1, c: generate b and then x
            b_recovered = recover_b(A, b1, c, q)
            x_recovered = recover_x(A, b_recovered, c, q)
            # debug: print error if recover not successful
            if (np.linalg.norm(x_recovered - x) != 0):
                print(np.linalg.norm(x_recovered - x))
            if not zq:
                x_recovered = zq_x_inv(x_recovered, q) # default
            if debug:
                x_recovered = zq_x_inv(x_recovered, q)
                x_recovered = x_recovered @ eig_vecs_inv

            np.save(os.path.join(output_folder, m, f, e), x_recovered)

end_time = time.time()
time_elapsed = end_time - start_time
print(f"Time elapsed for datagen {time_elapsed/60:.2f} min, or {time_elapsed/3600:.2f} hours. " )