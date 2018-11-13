# variational_saccading

Implementation of variational saccading

# Experiment 1: Two-Digit-Cluttered MNIST digit identification:

``` python
python main.py --seed=1234 --max-time-steps=4 --synthetic-upsample-size=2528 --downsample-scale=1 --window-size=64 --epochs=4000 --task=clutter --data-dir=$HOME/datasets/cluttered_mnist_2digits_120k --visdom-url=http://neuralnetworkart.com --visdom-port=8097 --lr=1e-05 --clip=0.25 --latent-size=256 --max-image-percentage=0.3 --dense-normalization=none --conv-normalization=batchnorm --batch-size=100 --reparam-type=isotropic_gaussian --nll-type=bernoulli --encoder-layer-type=resnet --decoder-layer-type=dense --continuous-size=6 --optimizer=adam --use-noisy-rnn-state --activation=elu --disable-gated --kl-reg=5.0 --ngpu=1 --uid=best_clutter_find2digits_0
```
