import os
import gc
import time
import psutil
import argparse
import numpy as np
import pprint
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.autograd import Variable
from torchvision.models.resnet import resnet18
from torchvision.models import vgg16_bn


from models.vae.vrnn import VRNN
from models.vae.parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from models.vae.sequentially_reparameterized_vae import SequentiallyReparameterizedVAE
from helpers.layers import EarlyStopping, init_weights, BWtoRGB
from models.pool import train_model_pool
from models.saccade import Saccader
from datasets.loader import get_split_data_loaders, get_loader, simple_merger, sequential_test_set_merger
from optimizers.adamnormgrad import AdamNormGrad
from optimizers.adamw import AdamW
from optimizers.utils import decay_lr_every
from helpers.grapher import Grapher
from helpers.metrics import softmax_accuracy
from helpers.utils import same_type, ones_like, \
    append_to_csv, num_samples_in_loader, expand_dims, \
    dummy_context, register_nan_checks, network_to_half

parser = argparse.ArgumentParser(description='Variational Saccading')

# Task parameters
parser.add_argument('--use-full-resolution', action='store_true', default=False,
                    help='use the full resolution image instead of the downsampled one (default: False)')
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
parser.add_argument('--early-stop', action='store_true',
                    help='enable early stopping (default: False)')

# handle scaling of images and related imgs
parser.add_argument('--synthetic-upsample-size', type=int, default=0,
                    help="""size to upsample image before downsampling to
                    blurry version for synthetic problems (default: 0)""")
parser.add_argument('--downsample-scale', type=int, default=7,
                    help='downscale the image by this scalar, eg: [100 // 8 , 100 // 8] (default: 8)')

# Model parameters
parser.add_argument('--baseline', type=str, default='resnet18',
                    help='baseline model to use (resnet18/vgg16_bn) (default: resnet18)')
parser.add_argument('--conv-normalization', type=str, default='groupnorm',
                    help='normalization type: batchnorm/groupnorm/instancenorm/none (default: groupnorm)')
parser.add_argument('--dense-normalization', type=str, default='batchnorm',
                    help='normalization type: batchnorm/instancenorm/none (default: batchnorm)')
parser.add_argument('--restore', type=str, default=None,
                    help='path to a model to restore (default: None)')


# Optimizer
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--optimizer', type=str, default="adam",
                    help="specify optimizer (default: adam)")
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping for RNN (default: 0)')

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
    numpy.random.seed(args.seed)
    torch.manual_seed_all(args.seed)

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
        "sgd_momentum": lambda params, lr : optim.SGD(params,
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
                              v.detach(), global_step=0) # dont use step


def _add_loss_map(loss_tm1, loss_t):
    ''' helper to add two maps and keep counts
        of the total samples for reduction later'''
    if not loss_tm1: # base case: empty dict
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
            else:                resultant[k] = loss_tm1[k] + v

    # increment total count
    resultant['count'] = loss_tm1['count'] + 1
    return resultant


def _mean_map(loss_map):
    ''' helper to reduce all values by the key count '''
    for k in loss_map.keys():
        loss_map[k] /= loss_map['count']

    return loss_map


def generate_related(data, x_original, args):
    # handle logic for crop-image-loader
    if x_original is not None:
        return x_original, data

    # first downsample the image and then upsample it
    # this creates a 'blurry' related image making the problem tougher
    original_img_size = tuple(data.size()[-2:])
    ds_img_size = tuple(int(i) for i in np.asarray(original_img_size)
                        // args.downsample_scale)  # eg: [12, 12]
    x_downsampled = F.interpolate(
        F.interpolate(data, ds_img_size, mode='bilinear'), # blur the crap out
        original_img_size, mode='bilinear')             # of the original data
    x_upsampled = F.interpolate(data, (args.synthetic_upsample_size,
                                       args.synthetic_upsample_size), mode='bilinear')
    return x_upsampled, x_downsampled


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
                data_to_infer = x_original if args.use_full_resolution else x_related
                loss_logits_t = model(data_to_infer)
                loss_t = {'loss_mean': F.cross_entropy(
                    input=loss_logits_t, target=labels)}

                # compute accuracy and aggregate into map
                loss_t['accuracy_mean'] = softmax_accuracy(
                    F.softmax(loss_logits_t, -1),
                    labels, size_average=True
                )

                loss_map = _add_loss_map(loss_map, loss_t)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) \
                    if not args.half is True else optimizer.clip_master_grads(args.clip)

            optimizer.step()
            del loss_t

    loss_map = _mean_map(loss_map) # reduce the map to get actual means
    correct_percent = 100.0 * loss_map['accuracy_mean']

    print('''{}[Epoch {}][{} samples][{:.2f} sec]:Average loss: {:.4f}\tAcc: {:.4f}'''.format(
        prefix, epoch, num_samples, time.time() - start_time,
        loss_map['loss_mean'].item(),
        correct_percent))

    # add memory tracking
    if plot_mem:
        process = psutil.Process(os.getpid())
        loss_map['cpumem_scalar'] = process.memory_info().rss * 1e-6
        loss_map['cudamem_scalar'] = torch.cuda.memory_allocated() * 1e-6

    # plot all the scalar / mean values
    register_plots(loss_map, grapher, epoch=epoch, prefix=prefix)

    # plot images, crops, inlays and all relevant images
    input_imgs_map = {
        'related_imgs': F.interpolate(x_related, (32, 32), mode='bilinear'),
        'original_imgs': F.interpolate(x_original, (32, 32), mode='bilinear')
    }
    register_images(input_imgs_map, grapher, prefix=prefix)
    grapher.show()

    # return this for early stopping
    loss_val = {
        'loss_mean': loss_map['loss_mean'].clone().detach().item(),
        'acc_mean': correct_percent
    }

    # delete the data instances, see https://tinyurl.com/ycjre67m
    loss_map.clear(); input_imgs_map.clear()
    del loss_map; del input_imgs_map
    del x_related; del x_original; del labels
    gc.collect()

    # return loss and accuracy
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
    loader = get_loader(args, transform=None,
                        sequentially_merge_test=False,
                        postfix="_large")

    # append the image shape to the config & build the VAE
    args.img_shp = loader.img_shp
    model_map = {
        'vgg16_bn': vgg16_bn,
        'resnet18': resnet18,
    }

    print("using {} baseline model on {}-{} with batch-size {}".format(
        args.baseline,
        args.task,
        "full" if args.use_full_resolution else "truncated",
        args.batch_size
    ))
    model = nn.Sequential(
            BWtoRGB(),
            nn.Upsample(size=[224, 224], mode='bilinear'),
            model_map[args.baseline](num_classes=loader.output_size)
        )

    # FP16-ize, cuda-ize and parallelize (if requested)
    model = model.half() if args.half is True else model
    model = model.cuda() if args.cuda is True else model
    model = nn.DataParallel(model) if args.ngpu > 1 else model

    # build the grapher object (tensorboard or visdom)
    # and plot config json to visdom
    if args.visdom_url is not None:
        grapher = Grapher('visdom',
                          env=get_name(),
                          server=args.visdom_url,
                          port=args.visdom_port)
    else:
        grapher = Grapher('tensorboard', comment=get_name())

    grapher.add_text('config', pprint.PrettyPrinter(indent=4).pformat(vars(args)), 0)
    return [model, loader, grapher]


def get_name():
    return "{}_{}_{}{}_batch{}".format(
        args.uid,
        args.baseline,
        args.task,
        "full" if args.use_full_resolution else "truncated",
        args.batch_size
    )

def run(args):
    # collect our model and data loader
    model, loader, grapher = get_model_and_loader()

    # collect our optimizer
    optimizer = build_optimizer(model)

    # train the VAE on the same distributions as the model pool
    if args.restore is None:
        print("training current distribution for {} epochs".format(args.epochs))
        early = EarlyStopping(model, burn_in_interval=100, max_steps=80) if args.early_stop else None

        test_loss, test_acc = 0.0, 0.0
        for epoch in range(1, args.epochs + 1):
            train(epoch, model, optimizer, loader.train_loader, grapher)
            test_loss = test(epoch, model, loader.test_loader, grapher)

            if args.early_stop and early(test_loss['pred_loss_mean']):
                early.restore() # restore and test+generate again
                test_loss = test(epoch, model, loader.test_loader, grapher)
                break

            # adjust the LR if using momentum sgd
            if args.optimizer == 'sgd_momentum':
                decay_lr_every(optimizer, args.lr, epoch)


        grapher.save() # save to endpoint after training
    else:
        model = torch.load(args.restore)
        test_loss = test(epoch, model, loader.test_loader, grapher)

    # evaluate one-time metrics
    append_to_csv([test_loss['acc_mean']], "{}_test_acc.csv".format(args.uid))


if __name__ == "__main__":
    run(args)
