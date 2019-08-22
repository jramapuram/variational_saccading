#!/bin/sh

# pull the project
tar -C $HOME -xvf /tmp/project*.tar.gz

# pull the two-digit-clutter digit identification problem
aws s3 sync s3://jramapuram-datasets/uncluttered_mnist_2sumdigits_120k ~/datasets/uncluttered_mnist_2sumdigits_120k

# Execute the two-digit clutter ID problem (experiment 1 in paper)
cd ~/variational_saccading && sh ./docker/run.sh "python baselines.py --seed=1234 --synthetic-upsample-size=2528 --downsample-scale=1 --use-full-resolution --epochs=1000 --task=clutter --data-dir=/datasets/uncluttered_mnist_2sumdigits_120k --visdom-url=http://neuralnetworkart.com --visdom-port=8097 --lr=1e-03 --batch-size=160 --optimizer=adam --ngpu=8 --uid=baselinesAWS_clutterSUM_0"
