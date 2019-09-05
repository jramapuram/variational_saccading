# variational_saccading

Implementation of variational saccading

## Experiment 1: Two-Digit-Cluttered MNIST digit identification:

Be sure to change `YOUR_VISDOM_URL` and `YOUR_VISDOM_PORT` below.

```bash
sh ./docker/run.sh "python main.py --seed=1234 --max-time-steps=4 --synthetic-upsample-size=100 --downsample-scale=6 --window-size=64 --epochs=2000 --task=clutter --data-dir=/cluttered_mnist --visdom-url=http://YOUR_VISDOM_URL --visdom-port=YOUR_VISDOM_PORT --lr=1e-05 --clip=0.25 --latent-size=256 --max-image-percentage=0.3 --dense-normalization=none --conv-normalization=batchnorm --batch-size=100 --reparam-type=isotropic_gaussian --nll-type=bernoulli --encoder-layer-type=resnet --decoder-layer-type=dense --continuous-size=6 --optimizer=adam --use-noisy-rnn-state --activation=elu --disable-gated --kl-reg=5.0 --ngpu=1 --uid=saccadingExp1"
```