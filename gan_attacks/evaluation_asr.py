import numpy as np
import matplotlib.pyplot as plt
import os
from deepface import DeepFace
import seaborn as sns
import argparse
import time
import cv2
from collections import defaultdict

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)
        
def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def parse_identity(path, attack):
    basename = os.path.basename(path)
    identity_str = basename.split('_')[2].split('|')[0] # e.g., '132' and '2' from 'attack_iden_132|2.png'
    if attack == "gmi":
        return int(identity_str)-1
    else:
        return int(identity_str)

def get_trainset_id2img(trainset_file):
    """
    Reads a trainset.txt file with lines like:
    image_0001.png 132
    image_0002.png 132
    image_0100.png 155

    Returns:
        A dict: id (int) -> list of image filenames
    """
    id2img = defaultdict(list)
    with open(trainset_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, identity = parts
                identity = int(identity)
                id2img[identity].append(filename)
    return dict(id2img)

def reconstruction_accuracy(original_path, reconstructed_path, attack, metric='euclidean_l2', model='Facenet512', detector_backend='mtcnn', generate_plot=False, visualize=False, 
                            distances_path="", plot_path="", result_path=""):
    
    recon_paths = os.listdir(reconstructed_path)
    recon_paths.sort()
    print("Reconstructed images: ", len(recon_paths))

    id2img = get_trainset_id2img(os.path.join(os.getcwd(), "../attack_dataset/CelebA/trainset.txt"))

    # List of 5 min distances for each identity
    distances = defaultdict(list)
    threshold = 0

    for re_img_path in recon_paths:
        if re_img_path.endswith('.txt'):
            continue
        iden = parse_identity(re_img_path, attack)
        re_path = os.path.join(reconstructed_path, re_img_path)
        orig_paths = id2img[iden]
    
        # Compute minimum distance for single reconstructed image, over all relevant original images
        min_dist = float('inf')
        for i in range(len(orig_paths)):
            orig_path = os.path.join(original_path, orig_paths[i])

            res = DeepFace.verify(orig_path, re_path, model_name=model, distance_metric=metric, detector_backend=detector_backend, enforce_detection=False)
            if i == 0:
                old = threshold
                threshold = res['threshold']
                if old != threshold:
                    print('Threshold: ', threshold)
            min_dist = min(min_dist, res['distance'])
            
        distances[iden].append(min_dist)
        print(f"Minimum distance of {iden}: {min_dist}")
        # np.save(os.path.join(result_path, f"dist_{iden}"), distances[iden])
    
    distances = [min(dlist) for dlist in distances.values()]
    distances = np.array(distances)
    acc = np.mean(distances < threshold)
    print(f"Reconstruction Accuracy: {acc:.4f}")

    if generate_plot:
        np.save(distances_path, distances)
        sns.kdeplot(distances)
        plt.axvline(x=threshold, color='r', linestyle='--')
        plt.xlabel('{} Distance'.format(metric))
        plt.ylabel('Density')
        plt.savefig(plot_path)
        #plt.show()
        print(f"Saved distances and plot to {distances_path} and {plot_path}")
        

def reconstruction_accuracy_from_distances(distances_path):
    distances = np.load(distances_path)
    threshold=1.04
    accuracy = np.mean(distances < threshold)
    print('Accuracy: ', accuracy)
        

def main():
    args = create_argparser().parse_args()
    output_dir = args.output_dir
    original_dir = args.original_dir
    reconstructed_dir = args.reconstructed_dir
    detector_backend = args.detector_backend
    prot_scheme = args.prot_scheme
    same_r = args.same_r
    metric = args.metric
    model = args.model
    attack = args.attack
    os.environ["DEVICE_ID"] = args.device_id

    original_path = os.path.join(os.getcwd(), original_dir)
    reconstructed_path = os.path.join(os.getcwd(), reconstructed_dir, f'{prot_scheme}_{same_r}_{attack}', "all")
    result_path = os.path.join(os.getcwd(), output_dir, f'{prot_scheme}_{same_r}_{attack}')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    distances_path = os.path.join(result_path, f"distances")
    plot_path = os.path.join(result_path, f"plot.png")

    print("Evaluating", output_dir, f'{prot_scheme}_{same_r}')
    start_time = time.time()

    reconstruction_accuracy(original_path, reconstructed_path, attack, metric, model, detector_backend, generate_plot=True, visualize=False, distances_path=distances_path, plot_path=plot_path, result_path=result_path)

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Time elapsed for evaluation {time_elapsed/60:.2f} min, or {time_elapsed/3600:.2f} hours. " )

def create_argparser():
    defaults = dict(
        device_id="0",
        eval_dset="test",
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--reconstructed_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--detector_backend", type=str, required=True)
    parser.add_argument("--prot_scheme", type=str, required=True)
    parser.add_argument("--same_r", type=str, required=True)
    parser.add_argument("--attack", type=str, required=True)

    return parser
    
if __name__ == "__main__":
    main()