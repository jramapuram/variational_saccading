#!/bin/sh

# pull the project
tar -C $HOME -xvf /tmp/project*.tar.gz

# pull the two-digit-clutter digit identification problem
aws s3 sync s3://jramapuram-datasets/uncluttered_mnist_2sumdigits_120k ~/datasets/uncluttered_mnist_2sumdigits_120k

# Execute the two-digit clutter ID problem (experiment 1 in paper)
cd ~/variational_saccading && sh ./docker/run.sh "python main.py --seed=1234 --max-time-steps=2 --synthetic-upsample-size=100 --downsample-scale=1 --window-size=32 --epochs=2000 --task=clutter --data-dir=/datasets/uncluttered_mnist_2sumdigits_120k --visdom-url=http://neuralnetworkart.com --visdom-port=8097 --lr=1e-05 --clip=0.25 --latent-size=256 --max-image-percentage=0.3 --dense-normalization=none --conv-normalization=batchnorm --batch-size=128 --reparam-type=isotropic_gaussian --nll-type=bernoulli --encoder-layer-type=resnet --decoder-layer-type=dense --continuous-size=6 --optimizer=adam --use-noisy-rnn-state --activation=elu --disable-gated --kl-reg=5.0 --ngpu=1 --uid=best_noclutter_0"
