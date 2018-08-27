import time
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

from models.vae.vrnn import VRNN
from models.vae.parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from models.vae.sequentially_reparameterized_vae import SequentiallyReparameterizedVAE
from helpers.layers import EarlyStopping, init_weights
from models.pool import train_model_pool
from models.saccade import Saccader
from datasets.loader import get_split_data_loaders, get_loader, simple_merger, sequential_test_set_merger
from optimizers.adamnormgrad import AdamNormGrad
from optimizers.adamw import AdamW
from optimizers.utils import adjust_learning_rate
from helpers.grapher import Grapher
from helpers.metrics import softmax_accuracy
from helpers.utils import same_type, ones_like, \
    append_to_csv, num_samples_in_loader, expand_dims, \
    dummy_context, register_nan_checks, network_to_half

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
parser.add_argument('--early-stop', action='store_true',
                    help='enable early stopping (default: False)')

# handle scaling of images and related imgs
parser.add_argument('--synthetic-upsample-size', type=int, default=0,
                    help="""size to upsample image before downsampling to
                    blurry version for synthetic problems (default: 0)""")
parser.add_argument('--max-image-percentage', type=float, default=0.15,
                    help='maximum percentage of the image to look over (default: 0.15)')
parser.add_argument('--window-size', type=int, default=32,
                    help='window size for saccades [becomes WxW] (default: 32)')
parser.add_argument('--downsample-scale', type=int, default=7,
                    help='downscale the image by this scalar, eg: [100 // 8 , 100 // 8] (default: 8)')

# Model parameters
parser.add_argument('--activation', type=str, default='identity',
                    help='default activation function (default: identity)')
parser.add_argument('--filter-depth', type=int, default=32,
                    help='number of initial conv filter maps (default: 32)')
parser.add_argument('--reparam-type', type=str, default='beta',
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
parser.add_argument('--use-noisy-rnn-state', action='store_true',
                    help='uses a noisy initial rnn state instead of zeros (default: False)')
parser.add_argument('--max-time-steps', type=int, default=4,
                    help='max time steps for RNN (default: 4)')

# Regularizer related
parser.add_argument('--continuous-mut-info', type=float, default=0.0,
                    help='-continuous_mut_info * I(z_c; x) is applied (opposite dir of disc)(default: 0.0)')
parser.add_argument('--discrete-mut-info', type=float, default=0.0,
                    help='+discrete_mut_info * I(z_d; x) is applied (default: 0.0)')
parser.add_argument('--mut-clamp-strategy', type=str, default="clamp",
                    help='clamp mut info by norm / clamp / none (default: clamp)')
parser.add_argument('--mut-clamp-value', type=float, default=100.0,
                    help='max / min clamp value if above strategy is clamp (default: 100.0)')
parser.add_argument('--monte-carlo-infogain', action='store_true',
                    help='use the MC version of mutual information gain / false is analytic (default: False)')
parser.add_argument('--mut-reg', type=float, default=0.3,
                    help='mutual information regularizer [mixture only] (default: 0.3)')
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
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--optimizer', type=str, default="adam",
                    help="specify optimizer (default: adam)")
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clipping for RNN (default: 0.25)')

# Visdom parameters
parser.add_argument('--visdom-url', type=str, default="http://localhost",
                    help='visdom URL for graphs (default: http://localhost)')
parser.add_argument('--visdom-port', type=int, default="8097",
                    help='visdom port for graphs (default: 8097)')

# Device parameters
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
if args.half:
    from apex import amp
    amp_handle = amp.init()

    from apex.fp16_utils import FP16_Module
    from apex.fp16_utils import FP16_Optimizer

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
    if args.half:
        return FP16_Optimizer(optimizer)

    return optimizer


def register_images(images, names, grapher, prefix="train"):
    ''' helper to register a list of images '''
    if isinstance(images, list):
        assert len(images) == len(names), "#images[{}] != #names[{}]".format(
            len(images), len(names))
        for im, name in zip(images, names):
            register_images(im, name, grapher, prefix=prefix)
    else:
        images = torch.min(images.detach(), ones_like(images))
        grapher.register_single({'{}_{}'.format(prefix, names): images},
                                plot_type='imgs')


def register_plots(loss, grapher, epoch, prefix='train'):
    ''' helper to register all plots with *_mean and *_scalar '''
    for k, v in loss.items():
        if isinstance(v, map):
            register_plots(loss[k], grapher, epoch, prefix=prefix)

        if 'mean' in k or 'scalar' in k:
            key_name = k.split('_')[0]
            value = v.item() if not isinstance(v, (float, np.float32, np.float64)) else v
            grapher.register_single({'%s_%s' % (prefix, key_name): [[epoch], [value]]},
                                    plot_type='line')


def _add_loss_map(loss_tm1, loss_t):
    ''' helper to add two maps and keep counts
        of the total samples for reduction later'''
    if not loss_tm1: # base case: empty dict
        resultant = {'count': 1}
        for k, v in loss_t.items():
            if 'mean' in k or 'scalar' in k:
                if not isinstance(v, (float, np.float32, np.float64)):
                    resultant[k] = v.detach()
                else:
                    resultant[k] = v

        return resultant

    resultant = {}
    for (k, v) in loss_t.items():
        if 'mean' in k or 'scalar' in k:
            if not isinstance(v, (float, np.float32, np.float64)):
                resultant[k] = loss_tm1[k] + v.detach()
            else:
                resultant[k] = loss_tm1[k] + v

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

    # first upsample the image and then downsample the upsampled version
    # this creates a 'blurry' related image making the problem tougher
    original_img_size = tuple(data.size()[-2:])
    ds_img_size = tuple(int(i) for i in np.asarray([32, 32]) // args.downsample_scale)  # eg: [12, 12]
    downsampled = F.upsample(
        F.upsample(data, ds_img_size, mode='bilinear'),
        (32, 32), mode='bilinear')
    data = F.upsample(data, (args.synthetic_upsample_size,
                             args.synthetic_upsample_size), mode='bilinear')
    return data, downsampled


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

    if args.half and is_data_tensor:
        tensor = tensor.half()

    if args.cuda:
        tensor = tensor.cuda()

    return tensor


def execute_graph(epoch, model, data_loader, grapher, optimizer=None, prefix='test'):
    ''' execute the graph; when 'train' is in the name the model runs the optimizer '''
    start_time = time.time()
    model.eval() if not 'train' in prefix else model.train()
    assert optimizer is not None if 'train' in prefix else optimizer is None
    loss_map, num_samples = {}, 0
    x_original, x_related = None, None

    for item in data_loader:
        # first destructure the data and cuda-ize and wrap in vars
        x_original, x_related, labels = _unpack_data_and_labels(item)
        x_related, labels = cudaize(x_related, is_data_tensor=True), cudaize(labels)

        if 'train' in prefix:  # zero gradients on optimizer
            optimizer.zero_grad()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            x_original, x_related = generate_related(x_related, x_original, args)
            x_original = cudaize(x_original, is_data_tensor=True)

            # run the VAE + the DNN and gather the loss function
            output_map = model(x_original, x_related)
            loss_t = model.loss_function(x_related, labels, output_map)

            # compute accuracy and aggregate into map
            loss_t['accuracy_mean'] = softmax_accuracy(
                F.softmax(output_map['preds'], -1),
                labels, size_average=True
            )

            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += x_related.size(0)

        if 'train' in prefix:    # compute bp and optimize
            if args.half:
            #     optimizer.backward(loss_t['loss_mean'])
                with amp_handle.scale_loss(loss_t['loss_mean'], optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_t['loss_mean'].backward()

            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.clip)
                # torch.nn.utils.clip_grad_value_(model.parameters(),
                #                                 args.clip)

            optimizer.step()

    loss_map = _mean_map(loss_map) # reduce the map to get actual means
    correct_percent = 100.0 * loss_map['accuracy_mean']

    print('{}[Epoch {}][{} samples][{:.2f} sec]: Average loss: {:.4f}\tKLD: {:.4f}\tNLL: {:.4f}\tAcc: {:.4f}'.format(
        prefix, epoch, num_samples, time.time() - start_time,
        loss_map['loss_mean'].item(),
        loss_map['kld_mean'].item(),
        loss_map['nll_mean'].item(),
        correct_percent))

    # gather scalar values of reparameterizers (if they exist)
    reparam_scalars = model.vae.get_reparameterizer_scalars()

    # plot the test accuracy, loss and images
    register_plots({**loss_map, **reparam_scalars}, grapher, epoch=epoch, prefix=prefix)
    images = [F.upsample(x_related, (32, 32), mode='bilinear')]
    img_names = ['related_imgs']
    if isinstance(x_original, torch.Tensor):
        images.append(F.upsample(x_original, (32, 32), mode='bilinear'))
    else:
        z = np.tile(np.array([[1.0, 0.0, 0.0]]), (x_related.shape[0], 1))
        images.append(
            torch.cat([F.upsample(x_original[i](z_i, override=True), (32, 32), mode='bilinear')
                       for i, z_i in enumerate(z)])
        )

    img_names.append('original_imgs')

    for i, (crop, decoded) in enumerate(zip(output_map['crops'], output_map['decoded'])):
        images.append(F.upsample(crop, (32, 32), mode='bilinear'))
        img_names.append('crop_{}'.format(i))
        images.append(F.upsample(model.vae.nll_activation(decoded),
                                 (32, 32), mode='bilinear'))
        img_names.append('decoded_{}'.format(i))

    register_images(images, img_names, grapher, prefix=prefix)
    grapher.show()

    # delete the data instances, see https://tinyurl.com/ycjre67m
    images.clear(), img_names.clear()
    output_map.clear(), reparam_scalars.clear()
    del x_related; del x_original; del labels

    # return this for early stopping
    loss_val = loss_map['loss_mean'].detach().item()
    loss_map.clear()
    return loss_val, correct_percent


def train(epoch, model, optimizer, loader, grapher, prefix='train'):
    ''' train loop helper '''
    return execute_graph(epoch, model, loader,
                         grapher, optimizer, 'train')


def test(epoch, model, loader, grapher, prefix='test'):
     ''' test loop helper '''
     return execute_graph(epoch, model, loader,
                          grapher, prefix='test')


def get_model_and_loader():
    ''' helper to return the model and the loader '''
    loader = get_loader(args, transform=None, sequentially_merge_test=False,
                        window_size=args.window_size,
                        max_img_percent=args.max_image_percentage,
                        postfix="_large")

    # append the image shape to the config & build the VAE
    args.img_shp = loader.img_shp
    vae = VRNN([loader.img_shp[0], 32, 32], #loader.img_shp,
               latent_size=512,            # XXX: hard coded
               normalization="batchnorm",  # XXX: hard coded
               n_layers=2,                 # XXX: hard coded
               bidirectional=True,         # XXX: hard coded
               kwargs=vars(args))

    # build the Variational Saccading module
    # and lazy generate the non-constructed modules
    saccader = Saccader(vae, loader.output_size, kwargs=vars(args))
    lazy_generate_modules(saccader, loader.train_loader)

    if args.half:
        name_fn = saccader.get_name
        loss_fn = saccader.loss_function
        conf = saccader.config
        saccader = FP16_Module(saccader.half())
        setattr(saccader, 'get_name', name_fn)
        setattr(saccader, 'loss_function', loss_fn)
        setattr(saccader, 'config', conf)

        saccader = network_to_half(saccader)
        # register_nan_checks(saccader)

    if args.cuda:
        saccader = saccader.cuda()

    if args.ngpu > 1: # parallelize across GPU
        saccader.parallel()

    # build the grapher object
    grapher = Grapher(env=saccader.get_name(),
                      server=args.visdom_url,
                      port=args.visdom_port)

    # register_nan_checks(saccader)
    return [saccader, loader, grapher]


def lazy_generate_modules(model, loader):
    ''' Super hax, but needed for building lazy modules '''
    model.eval()
    for item in loader:
        # first destructure the data and cuda-ize and wrap in vars
        x_original, x_related, _ = _unpack_data_and_labels(item)
        x_original, x_related = generate_related(x_related, x_original, args)
        with torch.no_grad():
            _ = model(x_original, x_related)
            del x_original; del x_related
            break

    #model = init_weights(model)


def generate(epoch, model, grapher, generate_every=10):
    ''' generate some synthetic '''
    if epoch % generate_every == 0:
        start_time = time.time()
        samples = model.generate(args.batch_size)
        num_samples = len(samples) * np.prod(list(samples[0].shape))
        print("generated[Epoch {}][{} samples][{} sec]".format(
            epoch,
            num_samples,
            time.time() - start_time)
        )
        img_names = ['samples_{}'.format(i) for i in range(len(samples))]
        register_images(samples, img_names, grapher, prefix="generated")
        grapher.show()

        # cleanups
        del samples


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
            generate(epoch, model, grapher)
            train(epoch, model, optimizer, loader.train_loader, grapher)
            test_loss, test_acc = test(epoch, model, loader.test_loader, grapher)

            if args.early_stop and early(test_loss):
                early.restore() # restore and test+generate again
                test_loss, test_acc = test(epoch, model, loader.test_loader, grapher)
                break

            # adjust the LR if using momentum sgd
            if args.optimizer == 'sgd_momentum':
                adjust_learning_rate(optimizer, args.lr, epoch)

        # plot config json to visdom
        grapher.vis.text(pprint.PrettyPrinter(indent=4).pformat(model.config),
                         opts=dict(title="config"))
        grapher.save() # save to json after distributional interval
    else:
        assert model.load(args.restore), "Failed to load model"
        test_loss, test_acc = test(epoch, model, loader.test_loader, grapher)

    # evaluate one-time metrics
    append_to_csv([test_loss], "{}_test_elbo.csv".format(args.uid))
    append_to_csv([test_acc], "{}_test_acc.csv".format(args.uid))


if __name__ == "__main__":
    run(args)
