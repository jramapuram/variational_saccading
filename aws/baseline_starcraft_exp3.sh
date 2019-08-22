#!/bin/sh

# pull the project
tar -C $HOME -xvf /tmp/project*.tar.gz

# pull the sc2 dataset
# aws s3 sync s3://jramapuram-datasets/starcraft_predictbattle_250k ~/datasets/starcraft_predictbattle_250k
aws s3 cp s3://jramapuram-datasets/starcraft_predictbattle_250k.tar.gz ~/datasets/sc2.tar.gz
cd ~/datasets && tar xvf sc2.tar.gz && rm sc2.tar.gz


# execute the experiment
cd ~/variational_saccading && sh ./docker/run.sh "python baselines.py --seed=1234 --synthetic-upsample-size=0 --downsample-scale=1 --use-full-resolution --epochs=1000 --task=starcraft_predict_battle --data-dir=/datasets/starcraft_predictbattle_250k --visdom-url=http://neuralnetworkart.com --visdom-port=8097 --lr=1e-03 --batch-size=32 --optimizer=adam --ngpu=8 --uid=baselinesAWS_Starcraft_weighted_noupsample_0"
