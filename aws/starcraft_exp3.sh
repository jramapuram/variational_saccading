#!/bin/sh

# pull the project
tar -C $HOME -xvf /tmp/project*.tar.gz

# pull the sc2 dataset
aws s3 cp s3://jramapuram-datasets/starcraft_predictbattle_250k.tar.gz ~/datasets/sc2.tar.gz
cd ~/datasets && tar xvf sc2.tar.gz && rm sc2.tar.gz

# sc2BBv2_starcraft_predict_battle_hp_search71__win64_us_dscale1_vrnn_2ts_1ns_1pklstarcraft_predict_battle_resnetdense_mixturecat10gauss128_actsoftplus_crgroupnorm_drnone_klr1.1_gsv1.0_mcig0_mcsnone_input3_64_64_batch64_mut0d0c_filter32_nllbernoulli_lr0.0003_epochs2000_adamnorm_ngpu1

# best so far
#cd ~/variational_saccading && sh ./docker/run.sh "python main.py  --max-time-steps=6 --synthetic-upsample-size=1024 --downsample-scale=1 --window-size=64 --epochs=4000 --task=starcraft_predict_battle --data-dir=/datasets/starcraft_predictbattle_250k --visdom-url=http://neuralnetworkart.com --visdom-port=8097 --lr=1e-05 --clip=0.25 --latent-size=256 --max-image-percentage=0.1 --dense-normalization=none --conv-normalization=groupnorm --batch-size=128 --reparam-type=isotropic_gaussian --nll-type=bernoulli --encoder-layer-type=resnet --decoder-layer-type=dense --continuous-size=6 --optimizer=adam --use-noisy-rnn-state --activation=relu --disable-gated --kl-reg=1.8 --uid=EC2_sc2_v4_weighted_2 --ngpu=4"

# same as above with 1gpu, but also concat pred size
cd ~/variational_saccading && sh ./docker/run.sh "python main.py  --max-time-steps=6 --synthetic-upsample-size=1024 --downsample-scale=1 --window-size=64 --epochs=4000 --task=starcraft_predict_battle --data-dir=/datasets/starcraft_predictbattle_250k --visdom-url=http://neuralnetworkart.com --visdom-port=8097 --lr=1e-05 --clip=0.25 --latent-size=256 --max-image-percentage=0.1 --dense-normalization=none --conv-normalization=groupnorm --batch-size=32 --reparam-type=isotropic_gaussian --nll-type=bernoulli --encoder-layer-type=resnet --decoder-layer-type=dense --continuous-size=6 --optimizer=adam --use-noisy-rnn-state --activation=relu --disable-gated --kl-reg=1.8 --uid=EC2_sc2_v4_weighted_2_concat --ngpu=1 --concat-prediction-size=10"
