#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import numpy as np
from subprocess import call
from os.path import expanduser

parser = argparse.ArgumentParser(description='Variational Saccading MNIST HP Search')
parser.add_argument('--num-trials', type=int, default=50,
                    help="number of different models to run for the HP search (default: 50)")
parser.add_argument('--num-titans', type=int, default=15,
                    help="number of TitanXP's (default: 15)")
parser.add_argument('--num-pascals', type=int, default=15,
                    help="number of P100's (default: 15)")
parser.add_argument('--singularity-img', type=str, default=None,
                    help="if provided uses a singularity image (default: None)")
args = parser.parse_args()


def get_rand_hyperparameters():
    return {
        'max-time-steps': np.random.choice([2, 3, 4]),
        'synthetic-upsample-size': 0,
        'downsample-scale': 1,
        'window-size': 32,
        'epochs': 2000,                              # FIXED, but uses ES
        'task': 'crop_dual_imagefolder',                            # FIXED
        'data-dir': os.path.join(expanduser("~"), 'datasets/cluttered_imagefolder_ptiff_v3'),
        'visdom-url': 'http://neuralnetworkart.com', # FIXED
        'visdom-port': 8097,                         # FIXED
        'differentiable-image-size': np.random.choice([300, 200, 100, 500, 1000]),
        'clip': np.random.choice([0, 0.25, 0.5, 0.75, 1.0, 5.0]),
        'latent-size': np.random.choice([64, 128, 256, 512]),
        'max-image-percentage': np.random.choice([0.1, 0.15, 0.2, 0.3]),
        'dense-normalization': np.random.choice(['batchnorm', 'none']),
        'conv-normalization': np.random.choice(['groupnorm', 'batchnorm', 'none']),
        'batch-size': np.random.choice([32, 48, 64, 76]),
        'reparam-type': np.random.choice(['beta', 'mixture', 'isotropic_gaussian']),
        'encoder-layer-type': np.random.choice(['conv', 'dense']),
        'decoder-layer-type': np.random.choice(['conv', 'dense']),
        'discrete-size': np.random.choice([6, 8, 10, 20, 30, 40, 64]),
        'continuous-size': np.random.choice([6, 8, 10, 20, 30, 40, 64]),
        'optimizer': np.random.choice(['adam', 'rmsprop', 'sgd_momentum']),
        'use-noisy-rnn-state': np.random.choice([1, 0]),
        'use-prior-kl': np.random.choice([1, 0]),
        'activation': np.random.choice(['identity', 'selu', 'elu', 'relu', 'softplus']),
        'disable-gated': np.random.choice([1, 0]),
        'kl-reg': np.random.choice([1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.3, 2.0, 3.0]),
        # 'continuous-mut-info': np.random.choice([1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 10.0]),
        # 'discrete-mut-info': np.random.choice([1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0, 10.0]),
        # 'monte-carlo-infogain': np.random.choice([1, 0]),
        # 'generative-scale-var': np.random.choice([1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1]),
        # 'mut-clamp-strategy': np.random.choice(['clamp', 'none', 'norm']),
        # 'mut-clamp-value': np.random.choice([1, 2, 5, 10, 30, 50, 100])
    }

def format_job_str(job_map, run_str):
    singularity_str = "" if args.singularity_img is None \
        else "module load GCC/6.3.0-2.27 Singularity/2.4.2"
    return """#!/bin/bash -l

#SBATCH --job-name={}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition={}
#SBATCH --time={}
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --constraint="COMPUTE_CAPABILITY_6_0|COMPUTE_CAPABILITY_6_1"
echo $CUDA_VISIBLE_DEVICES
{}
srun {}""".format(
    job_map['job-name'],
    job_map['partition'],
    job_map['time'],
    singularity_str,
    # job_map['gpu'],
    run_str
)

def unroll_hp_and_value(hpmap):
    base_str = ""
    no_value_keys = ["disable-gated", "use-pixel-cnn-decoder", "monte-carlo-infogain",
                     "use-noisy-rnn-state", "use-prior-kl"]
    for k, v in hpmap.items():
        if k in no_value_keys and v == 0:
            continue
        elif k in no_value_keys:
            base_str += " --{}".format(k)
            continue

        if k == "mut-clamp-value" and hpmap['mut-clamp-strategy'] != 'clamp':
            continue

        if k == "normalization" and hpmap['layer-type'] == 'dense':
            # only use BN or None for dense layers [GN doesnt make sense]
            base_str += " --normalization={}".format(np.random.choice(['batchnorm', 'none']))
            continue

        base_str += " --{}={}".format(k, v)

    return base_str

def format_task_str(hp):
    hpmap_str = unroll_hp_and_value(hp) # --model-dir=.nonfidmodels
    python_native = os.path.join(expanduser("~"), '.venv3/bin/python')
    python_bin = "singularity exec -B /home/ramapur0/opt:/opt --nv {} python".format(
        args.singularity_img) if args.singularity_img is not None else python_native
    return """{} ../main.py --early-stop {} --uid={}""".format(
        python_bin,
        hpmap_str,
        "{}".format(hp['task']) + "_hp_search{}_"
    ).replace("\n", " ").replace("\r", "").replace("   ", " ").replace("  ", " ").strip()

def get_job_map(idx, gpu_type):
    return {
        'partition': 'kruse-gpu,kalousis-gpu',
        'time': '48:00:00',
        'gpu': gpu_type,
        'job-name': "hp_search{}".format(idx)
    }

def run(args):
    # grab some random HP's and filter dupes
    hps = [get_rand_hyperparameters() for _ in range(args.num_trials)]

    # create multiple task strings
    task_strs = [format_task_str(hp) for hp in hps]
    print("#tasks = ", len(task_strs),  " | #set(task_strs) = ", len(set(task_strs)))
    task_strs = set(task_strs) # remove dupes
    task_strs = [ts.format(i) for i, ts in enumerate(task_strs)]

    # create GPU array and tile to the number of jobs
    gpu_arr = []
    for i in range(args.num_titans + args.num_pascals):
        if i < args.num_titans:
            gpu_arr.append('titan')
        else:
            gpu_arr.append('pascal')

    num_tiles = int(np.ceil(float(args.num_trials) / len(gpu_arr)))
    gpu_arr = [gpu_arr for _ in range(num_tiles)]
    gpu_arr = [item for sublist in gpu_arr for item in sublist]
    gpu_arr = gpu_arr[0:len(task_strs)]

    # create the job maps
    job_maps = [get_job_map(i, gpu_type) for i, gpu_type in enumerate(gpu_arr)]

    # sanity check
    assert len(task_strs) == len(job_maps) == len(gpu_arr), "#tasks = {} | #jobs = {} | #gpu_arr = {}".format(
        len(task_strs), len(job_maps), len(gpu_arr)
    )

    # finally get all the required job strings
    job_strs = [format_job_str(jm, ts) for jm, ts in zip(job_maps, task_strs)]

    # spawn the jobs!
    for i, js in enumerate(set(job_strs)):
        print(js + "\n")
        job_name = "hp_search_{}.sh".format(i)
        with open(job_name, 'w') as f:
            f.write(js)

        call(["sbatch", "./{}".format(job_name)])


if __name__ == "__main__":
    run(args)
