#!/bin/sh

# pull the project
aws s3 sync s3://jramapuram-projects/variational_saccading ~/variational_saccading

# pull the sc2 dataset
aws s3 sync s3://jramapuram-datasets/starcraft_predictbattle_250k ~/datasets/starcraft_predictbattle_250k

# execute the experiment
cd ~/variational_saccading && sh ./docker/run.sh "python main.py  --max-time-steps=4 --synthetic-upsample-size=1024 --downsample-scale=1 --window-size=64 --epochs=4000 --task=starcraft_predict_battle --data-dir=/datasets/starcraft_predictbattle_250k --visdom-url=http://neuralnetworkart.com --visdom-port=8097 --lr=1e-05 --clip=0.25 --latent-size=256 --max-image-percentage=0.3 --dense-normalization=none --conv-normalization=batchnorm --batch-size=100 --reparam-type=isotropic_gaussian --nll-type=bernoulli --encoder-layer-type=resnet --decoder-layer-type=dense --continuous-size=6 --optimizer=adam --use-noisy-rnn-state --activation=elu --disable-gated --kl-reg=5.0 --ngpu=1 --uid=sc2_v2"
