# variational_saccading [![DOI](https://zenodo.org/badge/124655768.svg)](https://zenodo.org/badge/latestdoi/124655768)



Implementation of [Variational Saccading: Efficient Inference for Large Resolution Images](https://arxiv.org/abs/1812.03170).  
![poster](poster/Variational_Saccading_BMVC2019.png)  

If you use this code or the ideas therein please cite:

```tex
@article{ramapuram2018variational,
  title={Variational Saccading: Efficient Inference for Large Resolution Images},
  author={Ramapuram, Jason and Diephuis, Maurits and Frantzeska, Lavda and Webb, Russ and Kalousis, Alexandros},
  journal={BMVC},
  year={2019}
}
```

## Experiment 1: Two-Digit-Cluttered MNIST digit identification:

The following runs a smaller version of the Two-Digit-Cluttered MNIST problem from the paper (images are 100x100).  
Be sure to spinup a [visdom server](https://github.com/facebookresearch/visdom) and change `YOUR_VISDOM_URL` and `YOUR_VISDOM_PORT` below to match your IP/hostname and port.

Clone the repo with **submodules**: `git clone --recursive https://github.com/jramapuram/variational_saccading` and run the following from the downloaded repo:

```bash
sh ./docker/run.sh "python main.py --seed=1234 --max-time-steps=4 --synthetic-upsample-size=100 \  
--downsample-scale=6 --window-size=64 --epochs=2000 --task=clutter --data-dir=/cluttered_mnist \  
--visdom-url=http://YOUR_VISDOM_URL --visdom-port=YOUR_VISDOM_PORT --lr=1e-05 --clip=0.25 \  
--latent-size=256 --max-image-percentage=0.3 --dense-normalization=none \  
--conv-normalization=batchnorm --batch-size=100 --reparam-type=isotropic_gaussian \  
--nll-type=bernoulli --encoder-layer-type=resnet --decoder-layer-type=dense \  
--continuous-size=6 --optimizer=adam --use-noisy-rnn-state --activation=elu \  
--disable-gated --kl-reg=5.0 --ngpu=1 --uid=saccadingExp1"
```

![setup](imgs/setup.gif)  
