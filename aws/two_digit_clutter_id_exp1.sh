#!/bin/sh

# pull the project
tar -C $HOME -xvf /tmp/project*.tar.gz

# pull the two-digit-clutter digit identification problem
aws s3 sync s3://jramapuram-datasets/clutteredmnist2digits120k ~/datasets/clutteredmnist2digits120k

# Execute the two-digit clutter ID problem (experiment 1 in paper)
cd ~/variational_saccading && sh ./docker/run.sh "python main.py --seed=1234 --max-time-steps=4 --synthetic-upsample-size=100 --downsample-scale=1 --window-size=64 --epochs=2000 --task=clutter --data-dir=/datasets/clutteredmnist2digits120k --visdom-url=http://neuralnetworkart.com --visdom-port=8097 --lr=1e-05 --clip=0.25 --latent-size=256 --max-image-percentage=0.3 --dense-normalization=none --conv-normalization=batchnorm --batch-size=100 --reparam-type=isotropic_gaussian --nll-type=bernoulli --encoder-layer-type=resnet --decoder-layer-type=dense --continuous-size=6 --optimizer=adam --use-noisy-rnn-state --activation=elu --disable-gated --kl-reg=5.0 --ngpu=1 --uid=test_994_clamp0.9"


#optimal : { 'activation': 'elu', 'add_img_noise': False, 'batch_size': 100, 'clip': 0.25, 'continuous_mut_info': 0, 'continuous_size': 6, 'conv_normalization': 'batchnorm', 'crop_padding': 6, 'cuda': True, 'data_dir': '/home/jramapuram/datasets/cluttered_mnist_2digits_120k', 'decoder_layer_type': 'dense', 'dense_normalization': 'none', 'detect_anomalies': False, 'disable_gated': True, 'discrete_mut_info': 0, 'discrete_size': 10, 'download': 1, 'downsample_scale': 3, 'early_stop': False, 'encoder_layer_type': 'resnet', 'epochs': 4000, 'filter_depth': 32, 'generative_scale_var': 1.0, 'half': False, 'img_shp': [1, 100, 100], 'kl_reg': 5.0, 'latent_size': 256, 'lr': 1e-05, 'max_image_percentage': 0.3, 'max_time_steps': 4, 'monte_carlo_infogain': False, 'mut_clamp_strategy': 'none', 'mut_clamp_value': 100.0, 'mut_reg': 0, 'ngpu': 1, 'nll_type': 'bernoulli', 'no_cuda': False, 'optimizer': 'adam', 'output_size': None, 'reparam_type': 'isotropic_gaussian', 'restore': None, 'seed': 1234, 'synthetic_rotation': 0, 'synthetic_upsample_size': 2528, 'task': 'clutter', 'uid': 'best_clutter_find2digits_titanv_vs0_ds3_0', 'use_noisy_rnn_state': True, 'use_prior_kl': False, 'vae_type': 'vrnn', 'visdom_port': 8097, 'visdom_url': 'http://neuralnetworkart.com', 'window_size': 64}
