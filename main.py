import os
import gc
import time
import psutil
import argparse
import numpy as np
import pprint
import torchvision
import torch
import functools
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')


from models.vae.vrnn import VRNN
from models.vae.vrnn_reinforce import VRNNReinforce

from models.vae.parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from models.vae.sequentially_reparameterized_vae import SequentiallyReparameterizedVAE
from helpers.layers import EarlyStopping, Rotate, init_weights
from models.pool import train_model_pool

from models.saccade import Saccader
from models.saccade_reinforce import SaccaderReinforce

from datasets.loader import get_split_data_loaders, get_loader, simple_merger, sequential_test_set_merger
from datasets.utils import normalize_images
from optimizers.adamnormgrad import AdamNormGrad
from optimizers.adamw import AdamW
from optimizers.utils import decay_lr_every
from helpers.grapher import Grapher
from helpers.metrics import softmax_accuracy, bce_accuracy
from helpers.utils import same_type, ones_like, \
    append_to_csv, num_samples_in_loader, expand_dims, \
    dummy_context, register_nan_checks, network_to_half, \
    number_of_parameters

parser = argparse.ArgumentParser(description='Variational Saccading')

# Task parameters
parser.add_argument('--uid', type=str, default="",
                    help="add a custom task-specific unique id; appended to name (default: None)")
parser.add_argument('--task', type=str, default="crop_dual_imagefolder",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter /
                    permuted / crop_dual_imagefolder] (default: crop_dual_imagefolder)""")
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='minimum number of epochs to train (default: 2000)')
parser.add_argument('--download', type=int, default=1,
                    help='download dataset from s3 (default: 1)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--early-stop', action='store_true', default=False,
                    help='enable early stopping (default: False)')

# handle scaling of images and related imgs
parser.add_argument('--synthetic-upsample-size', type=int, default=0,
                    help="""size to upsample image before downsampling to
                    blurry version for synthetic problems (default: 0)""")
parser.add_argument('--synthetic-rotation', type=float, default=0,
                    help='rotate proxy image in degrees (default: 0 degrees)')
parser.add_argument('--max-image-percentage', type=float, default=0.3,
                    help='maximum percentage of the image to look over (default: 0.15)')
parser.add_argument('--window-size', type=int, default=32,
                    help='window size for saccades [becomes WxW] (default: 32)')
parser.add_argument('--crop-padding', type=int, default=6,
                    help='the extra padding around the crop for numerical diff (default: 6)')
parser.add_argument('--downsample-scale', type=int, default=5,
                    help='downscale the image by this scalar, eg: [100 // 5 , 100 // 5] (default: 5)')

# Model parameters
parser.add_argument('--activation', type=str, default='identity',
                    help='default activation function (default: identity)')
parser.add_argument('--latent-size', type=int, default=512, metavar='N',
                    help='sizing for latent layers (default: 256)')
parser.add_argument('--output-size', type=int, default=None,
                    help='output class size [optional: usually auto-discovered] (default: None)')
parser.add_argument('--add-img-noise', action='store_true',
                    help='add scattered noise to  images (default: False)')
parser.add_argument('--filter-depth', type=int, default=32,
                    help='number of initial conv filter maps (default: 32)')
parser.add_argument('--reparam-type', type=str, default='isotropic_gaussian',
                    help='isotropic_gaussian / discrete / beta / mixture / beta_mixture (default: beta)')
parser.add_argument('--encoder-layer-type', type=str, default='conv',
                    help='dense or conv (default: conv)')
parser.add_argument('--decoder-layer-type', type=str, default='conv',
                    help='dense or conv or pixelcnn (default: conv)')
parser.add_argument('--continuous-size', type=int, default=6,
                    help='continuous latent size (6/2 units of this are used for [s, x, y]) (default: 6)')
parser.add_argument('--discrete-size', type=int, default=10,
                    help='discrete latent size (only used for mix + disc) (default: 10)')
parser.add_argument('--nll-type', type=str, default='bernoulli',
                    help='bernoulli or gaussian (default: bernoulli)')
parser.add_argument('--vae-type', type=str, default='vrnn',
                    help='vae type [sequential / parallel / vrnn] (default: parallel)')
parser.add_argument('--disable-gated', action='store_true',
                    help='disables gated convolutional or dense structure (default: False)')
parser.add_argument('--restore', type=str, default=None,
                    help='path to a model to restore (default: None)')

# RNN related
parser.add_argument('--use-prior-kl', action='store_true',
                    help='add a kl on the VRNN prior against the true prior (default: False)')
parser.add_argument('--use-noisy-rnn-state', action='store_true',
                    help='uses a noisy initial rnn state instead of zeros (default: False)')
parser.add_argument('--max-time-steps', type=int, default=4,
                    help='max time steps for RNN (default: 4)')

# (REINFORCE) loss related
parser.add_argument('--use-reinforce', action='store_true',
                    help='Use hard crops and REINFORCE loss (default: False)')
parser.add_argument('--std', type=float, default=0.17, help='standard deviation for locator sampler (default: 0.17)')

# Regularizer related
parser.add_argument('--continuous-mut-info', type=float, default=0,
                    help='-continuous_mut_info * I(z_c; x) is applied (opposite dir of disc)(default: 0.0)')
parser.add_argument('--discrete-mut-info', type=float, default=0,
                    help='+discrete_mut_info * I(z_d; x) is applied (default: 0.0)')
parser.add_argument('--mut-clamp-strategy', type=str, default="none",
                    help='clamp mut info by norm / clamp / none (default: none)')
parser.add_argument('--mut-clamp-value', type=float, default=100.0,
                    help='max / min clamp value if above strategy is clamp (default: 100.0)')
parser.add_argument('--monte-carlo-infogain', action='store_true',
                    help='use the MC version of mutual information gain / false is analytic (default: False)')
parser.add_argument('--mut-reg', type=float, default=0,
                    help='mutual information regularizer [mixture only] (default: 0)')
parser.add_argument('--kl-reg', type=float, default=1.0,
                    help='hyperparameter to scale KL term in ELBO')
parser.add_argument('--generative-scale-var', type=float, default=1.0,
                    help='scale variance of prior in order to capture outliers')
parser.add_argument('--conv-normalization', type=str, default='groupnorm',
                    help='normalization type: batchnorm/groupnorm/instancenorm/none (default: groupnorm)')
parser.add_argument('--dense-normalization', type=str, default='batchnorm',
                    help='normalization type: batchnorm/instancenorm/none (default: batchnorm)')

# Optimizer
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--optimizer', type=str, default="adam",
                    help="specify optimizer (default: adam)")
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping for RNN (default: 0.25)')

# Visdom / tensorboard parameters
parser.add_argument('--visdom-url', type=str, default=None,
                    help='visdom URL for graphs (needs http, eg: http://localhost) (default: None)')
parser.add_argument('--visdom-port', type=int, default=None,
                    help='visdom port for graphs (default: None)')

# Device parameters
parser.add_argument('--detect-anomalies', action='store_true', default=False,
                    help='detect anomalies in the computation graph (default: False)')
parser.add_argument('--seed', type=int, default=None,
                    help='seed for numpy and pytorch (default: None)')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpus available (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--half', action='store_true', default=False,
                    help='enables half precision training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    torch.backends.cudnn.benchmark = True


# handle randomness / non-randomness
if args.seed is not None:
    print("setting seed %d" % args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

# import FP16 optimizer and module
if args.half is True:
    from apex import amp
    from apex.fp16_utils import FP16_Optimizer
    amp_handle = amp.init()


# Global counter
TOTAL_ITER = 0


def build_optimizer(model):
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adamnorm": AdamNormGrad,
        "adamw": AdamW,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "sgd_momentum": lambda params, lr: optim.SGD(params,
                                                     lr=lr,
                                                     weight_decay=1e-4,
                                                     momentum=0.9),
        "lbfgs": optim.LBFGS
    }
    # filt = filter(lambda p: p.requires_grad, model.parameters())
    # return optim_map[args.optimizer.lower().strip()](filt, lr=args.lr)
    optimizer = optim_map[args.optimizer.lower().strip()](
        model.parameters(), lr=args.lr
    )
    if args.half is True:
        return FP16_Optimizer(optimizer, dynamic_loss_scale=True)

    return optimizer


def register_plots(loss, grapher, epoch, prefix='train'):
    ''' helper to register all plots with *_mean and *_scalar '''
    for k, v in loss.items():
        if isinstance(v, map):
            register_plots(loss[k], grapher, epoch, prefix=prefix)

        if 'mean' in k or 'scalar' in k:
            key_name = '-'.join(k.split('_')[0:-1])
            value = v.item() if not isinstance(v, (float, np.float32, np.float64)) else v
            grapher.add_scalar('{}_{}'.format(prefix, key_name), value, epoch)


def register_images(output_map, grapher, prefix='train'):
    ''' helper to register all plots with *_img and *_imgs
        NOTE: only registers 1 image to avoid MILLION imgs in visdom,
              consider adding epoch for tensorboardX though
    '''
    for k, v in output_map.items():
        if isinstance(v, map):
            register_images(output_map[k], grapher, epoch, prefix=prefix)

        if 'img' in k or 'imgs' in k:
            key_name = '-'.join(k.split('_')[0:-1])
            grapher.add_image('{}_{}'.format(prefix, key_name),
                              v.detach(), global_step=0)  # dont use step


def _add_loss_map(loss_tm1, loss_t):
    ''' helper to add two maps and keep counts
        of the total samples for reduction later'''
    if not loss_tm1:  # base case: empty dict
        resultant = {'count': 1}
        for k, v in loss_t.items():
            if 'mean' in k or 'scalar' in k:
                if isinstance(v, torch.Tensor):
                    resultant[k] = v.clone().detach()
                else:
                    resultant[k] = v

        return resultant

    resultant = {}
    for (k, v) in loss_t.items():
        if 'mean' in k or 'scalar' in k:
            if isinstance(v, torch.Tensor):
                resultant[k] = loss_tm1[k] + v.clone().detach()
            else:
                resultant[k] = loss_tm1[k] + v

    # increment total count
    resultant['count'] = loss_tm1['count'] + 1
    return resultant


def _mean_map(loss_map):
    ''' helper to reduce all values by the key count '''
    for k in loss_map.keys():
        if k == 'count':
            continue
        loss_map[k] /= loss_map['count']
    loss_map['count'] = 1
    return loss_map


# create this once
rotator = Rotate(args.synthetic_rotation)


def generate_related(data, x_original, args):
    # handle logic for crop-image-loader & multi-image-folder
    if x_original is not None:
        return x_original, data

    # first downsample the image and then upsample it
    # this creates a 'blurry' related image making the problem tougher
    original_img_size = tuple(data.size()[-2:])
    ds_img_size = tuple(int(i) for i in np.asarray(original_img_size)
                        // args.downsample_scale)  # eg: [12, 12]
    x_downsampled = F.interpolate(
        F.interpolate(data, ds_img_size, mode='bilinear', align_corners=True),  # blur the crap out
        original_img_size, mode='bilinear', align_corners=True)  # of the original data
    x_upsampled = F.interpolate(data, (args.synthetic_upsample_size,
                                       args.synthetic_upsample_size),
                                mode='bilinear', align_corners=True)
    return x_upsampled, rotator(x_downsampled)


def _unpack_data_and_labels(item):
    ''' helper to unpack the data and the labels
        in the presence of a lambda cropper '''
    if isinstance(item[-1], list):    # crop-dual loader logic
        x_original, (x_related, label) = item
    elif isinstance(item[0], list):   # multi-imagefolder logic
        assert len(item[0]) == 2, \
            "multi-image-folder [{} #datasets] unpack > 2 datasets not impl".format(len(item[0]))
        (x_related, x_original), label = item
    else:                             # standard loader
        x_related, label = item
        x_original = None

    return x_original, x_related, label


def cudaize(tensor, is_data_tensor=False):
    if isinstance(tensor, list):
        return tensor

    if args.half is True and is_data_tensor:
        tensor = tensor.half()

    if args.cuda:
        tensor = tensor.cuda()

    return tensor


def execute_graph(epoch, model, data_loader, grapher, optimizer=None,
                  prefix='test', plot_mem=False):
    ''' execute the graph; when 'train' is in the name the model runs the optimizer '''
    start_time = time.time()
    model.eval() if not 'train' in prefix else model.train()
    assert optimizer is not None if 'train' in prefix else optimizer is None
    loss_map, num_samples = {}, 0
    x_original, x_related = None, None

    for item in data_loader:
        # first destructure the data, cuda-ize and wrap in vars
        x_original, x_related, labels = _unpack_data_and_labels(item)
        x_related, labels = cudaize(x_related, is_data_tensor=True), cudaize(labels)

        if 'train' in prefix:  # zero gradients on optimizer
            optimizer.zero_grad()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            with torch.autograd.detect_anomaly() if args.detect_anomalies else dummy_context():
                x_original, x_related = generate_related(x_related, x_original, args)
                x_original = cudaize(x_original, is_data_tensor=True)

                # run the model and gather the loss map
                output_map = model(x_original, x_related)
                loss_t = model.loss_function(x_related, labels, output_map)

                # compute accuracy and aggregate into map
                accuracy_fn = softmax_accuracy if len(labels.shape) == 1 else bce_accuracy
                loss_t['accuracy_mean'] = accuracy_fn(
                    output_map['preds'],
                    labels, size_average=True
                )

                # print("Accuracy mean during run {} ".format(loss_t['accuracy_mean']))
                loss_map = _add_loss_map(loss_map, loss_t)
                # print("Accuracy mean during run {} ".format(loss_map['accuracy_mean']))

                num_samples += x_related.size(0)

        if 'train' in prefix:    # compute bp and optimize
            if args.half is True:
                optimizer.backward(loss_t['loss_mean'])
                # with amp_handle.scale_loss(loss_t['loss_mean'], optimizer,
                #                            dynamic_loss_scale=True) as scaled_loss:
                #     scaled_loss.backward()
            else:
                loss_t['loss_mean'].backward()

            if args.clip > 0:
                # TODO: clip by value or norm? torch.nn.utils.clip_grad_value_
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) \
                torch.nn.utils.clip_grad_value_(model.parameters(), args.clip) \
                    if not args.half is True else optimizer.clip_master_grads(args.clip)

            optimizer.step()

            # add individual loss vectors
            # iter_loss_map = _mean_map(loss_map)
            # print('{}[Epoch {}][{} samples]:\
            #     Average loss: {:.4f}\t Action loss: {:.4f}\t Baseline loss: {:.4f}\t REINFORCE loss :{:.4f}'.format(
            #     prefix, epoch, num_samples,
            #     iter_loss_map['loss_mean'].item(),
            #     iter_loss_map['actions_mean'].item(),
            #     iter_loss_map['baselines_mean'].item(),
            #     iter_loss_map['reinforce_mean'].item()
            #     ))

            del loss_t

    final_loss_map = _mean_map(loss_map)  # reduce the map to get actual means
    correct_percent = 100.0 * final_loss_map['accuracy_mean']
    if correct_percent > 100:
        print(loss_map['count'])
        print(loss_map['accuracy_mean'])
        print('Correct percentage {}'.format(correct_percent))

    # print('{}[Epoch {}][{} samples][{:.2f} sec]:\
    # Average loss: {:.4f}\tKLD: {:.4f}\t\
    # NLL: {:.4f}\tAcc: {:.4f}'.format(
    #     prefix, epoch, num_samples, time.time() - start_time,
    #     loss_map['loss_mean'].item(),
    #     loss_map['kld_mean'].item(),
    #     loss_map['nll_mean'].item(),
    #     correct_percent))

    # add individual loss vectors
    print('{}[Epoch {}][{} samples][{:.2f} sec]:\
    Average loss: {:.4f}\t Action loss: {:.4f}\t Baseline loss: {:.4f}\t REINFORCE loss :{:.4f}\t Acc: {:.4f}'.format(
        prefix, epoch, num_samples, time.time() - start_time,
        final_loss_map['loss_mean'].item(),
        final_loss_map['actions_mean'].item(),
        final_loss_map['baselines_mean'].item(),
        final_loss_map['reinforce_mean'].item(),
        correct_percent))

    # gather scalar values of reparameterizers (if they exist)
    reparam_scalars = model.vae.get_reparameterizer_scalars()

    # add memory tracking
    if plot_mem:
        process = psutil.Process(os.getpid())
        loss_map['cpumem_scalar'] = process.memory_info().rss * 1e-6
        loss_map['cudamem_scalar'] = torch.cuda.memory_allocated() * 1e-6

    # plot all the scalar / mean values
    register_plots({**loss_map, **reparam_scalars},
                   grapher, epoch=epoch, prefix=prefix)
    # plot images, crops, inlays and all relevant images
    input_imgs_map = {'related_imgs': x_related, 'original_imgs': x_original}
    imgs_map = model.get_imgs(x_related.size(0), output_map, input_imgs_map)
    register_images(imgs_map, grapher, prefix=prefix)

    # return this for early stopping
    loss_val = {
        'loss_mean': loss_map['loss_mean'].clone().detach().item(),
        'pred_loss_mean': loss_map['pred_loss_mean'].clone().detach().item(),
        'accuracy_mean': correct_percent
    }

    # delete the data instances, see https://tinyurl.com/ycjre67m
    loss_map.clear(), input_imgs_map.clear(), imgs_map.clear()
    output_map.clear(), reparam_scalars.clear()
    del loss_map
    del input_imgs_map
    del imgs_map
    del output_map
    del reparam_scalars
    del x_related
    del x_original
    del labels
    gc.collect()

    # return loss scalar map
    return loss_val


def train(epoch, model, optimizer, loader, grapher, prefix='train'):
    ''' train loop helper '''
    return execute_graph(epoch, model, loader,
                         grapher, optimizer, 'train',
                         plot_mem=True)


def test(epoch, model, loader, grapher, prefix='test'):
    ''' test loop helper '''
    return execute_graph(epoch, model, loader,
                         grapher, prefix='test',
                         plot_mem=False)


def get_model_and_loader():
    ''' helper to return the model and the loader '''
    aux_transform = None
    if args.synthetic_upsample_size > 0 and args.task == "multi_image_folder":
        def aux_transform(x): return F.interpolate(torchvision.transforms.ToTensor()(x).unsqueeze(0),
                                                   size=(args.synthetic_upsample_size,
                                                         args.synthetic_upsample_size),
                                                   mode='bilinear', align_corners=True).squeeze(0)

    loader = get_loader(args, transform=None,
                        sequentially_merge_test=False,
                        aux_transform=aux_transform,
                        postfix="_large", **vars(args))

    # append the image shape to the config & build the VAE
    args.img_shp = loader.img_shp
    if args.use_reinforce:
        vae = VRNNReinforce(loader.img_shp,
                            n_layers=2,            # XXX: hard coded
                            # bidirectional=True,    # XXX: hard coded
                            bidirectional=False,    # XXX: hard coded
                            kwargs=vars(args))
    else:
        vae = VRNN(loader.img_shp,
                   n_layers=2,            # XXX: hard coded
                   # bidirectional=True,    # XXX: hard coded
                   bidirectional=False,    # XXX: hard coded
                   kwargs=vars(args))

    # build the Variational Saccading module
    # and lazy generate the non-constructed modules
    if args.use_reinforce:
        saccader = SaccaderReinforce(vae, loader.output_size, kwargs=vars(args))
    else:
        saccader = Saccader(vae, loader.output_size, kwargs=vars(args))

    lazy_generate_modules(saccader, loader.train_loader)

    # FP16-ize, cuda-ize and parallelize (if requested)
    saccader = saccader.fp16() if args.half is True else saccader
    saccader = saccader.cuda() if args.cuda is True else saccader
    saccader.parallel() if args.ngpu > 1 else saccader

    # build the grapher object (tensorboard or visdom)
    # and plot config json to visdom
    if args.visdom_url is not None:
        grapher = Grapher('visdom',
                          env=saccader.get_name(),
                          server=args.visdom_url,
                          port=args.visdom_port)
    else:
        grapher = Grapher('tensorboard', comment=saccader.get_name())

    grapher.add_text('config', pprint.PrettyPrinter(indent=4).pformat(saccader.config), 0)

    # register_nan_checks(saccader)
    return [saccader, loader, grapher]


def lazy_generate_modules(model, loader):
    ''' Super hax, but needed for building lazy modules '''
    model.eval()
    model.config['half'] = False  # disable half here due to CPU weights
    for item in loader:
        # first destructure the data and cuda-ize and wrap in vars
        x_original, x_related, _ = _unpack_data_and_labels(item)
        x_original, x_related = generate_related(x_related, x_original, args)
        with torch.no_grad():
            _ = model(x_original, x_related)
            del x_original
            del x_related
            gc.collect()
            break

    # reset half tensors if requested since torch.cuda.HalfTensor has impls
    model.config['half'] = args.half

    # TODO: consider various initializations
    # model = init_weights(model)


def generate(epoch, model, grapher, generate_every=10):
    ''' generate some synthetic samples ever generate_every epoch'''
    if epoch % generate_every == 0:
        # a few time details
        start_time = time.time()
        samples = model.generate(args.batch_size)
        num_samples = len(samples) * np.prod(list(samples[0].shape))
        print("generate[Epoch {}][{} samples][{} sec]".format(
            epoch,
            num_samples,
            time.time() - start_time)
        )

        gen_map = {}  # generate and place in map
        for i, sample in enumerate(samples):
            gen_map['samples{}_imgs'.format(i)] \
                = F.interpolate(sample, (32, 32), mode='bilinear', align_corners=True)

        register_images(gen_map, grapher, prefix="generated")

        # XXX: memory cleanups
        gen_map.clear()
        del samples
        gc.collect()


def scalar_map_to_csvs(scalar_map, prefix='test'):
    ''' iterates over map and writes all items with _mean or _scalar fields to csv'''
    for k, v in scalar_map.items():
        if 'mean' in k or 'scalar' in k:
            append_to_csv([v], "{}_{}_{}.csv".format(args.uid, prefix, k))


def run(args):
    # collect our model and data loader
    model, loader, grapher = get_model_and_loader()
    print("model has {} params".format(number_of_parameters(model)))

    # collect our optimizer
    optimizer = build_optimizer(model)

    # train the VAE on the same distributions as the model pool
    if args.restore is None:
        print("training current distribution for {} epochs".format(args.epochs))
        early = EarlyStopping(model, burn_in_interval=100, max_steps=80) if args.early_stop else None

        test_map = {}
        for epoch in range(1, args.epochs + 1):
            generate(epoch, model, grapher)
            train(epoch, model, optimizer, loader.train_loader, grapher)
            test_map = test(epoch, model, loader.test_loader, grapher)

            if args.early_stop and early(test_map['pred_loss_mean']):
                early.restore()  # restore and test again
                test_map = test(epoch, model, loader.test_loader, grapher)
                break

            # adjust the LR if using momentum sgd
            if args.optimizer == 'sgd_momentum':
                decay_lr_every(optimizer, args.lr, epoch)

        grapher.save()  # save to endpoint after training
    else:
        assert model.load(args.restore), "Failed to load model"
        test_loss, test_acc = test(epoch, model, loader.test_loader, grapher)

    # evaluate one-time metrics
    scalar_map_to_csvs(test_map)

    # cleanups
    grapher.close()


if __name__ == "__main__":
    run(args)
