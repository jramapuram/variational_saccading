import argparse
import numpy as np
import pprint
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch.autograd import Variable

from models.vae.parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from models.vae.sequentially_reparameterized_vae import SequentiallyReparameterizedVAE
from helpers.layers import EarlyStopping, init_weights
from models.pool import train_model_pool
from models.saccade import Saccader
from datasets.loader import get_split_data_loaders, get_loader, simple_merger, sequential_test_set_merger
from optimizers.adamnormgrad import AdamNormGrad
from helpers.grapher import Grapher
from helpers.metrics import softmax_accuracy
from helpers.utils import same_type, ones_like, \
    append_to_csv, num_samples_in_loader, expand_dims, \
    dummy_context, register_nan_checks, network_to_half

parser = argparse.ArgumentParser(description='Variational Saccading')

# Task parameters
parser.add_argument('--uid', type=str, default="",
                    help="add a custom task-specific unique id; appended to name (default: None)")
parser.add_argument('--task', type=str, default="mnist",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / permuted] (default: mnist)""")
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='minimum number of epochs to train (default: 2000)')
parser.add_argument('--download', type=int, default=1,
                    help='download dataset from s3 (default: 1)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--early-stop', action='store_true',
                    help='enable early stopping (default: False)')
parser.add_argument('--window-size', type=int, default=8,
                    help='window size for saccades [becomes WxW] (default: 8)')
parser.add_argument('--upsample-size', type=int, default=3840,
                    help='size to upsample image before downsampling to blurry version (default: 3840)')
parser.add_argument('--downsample-scale', type=int, default=8,
                    help='downscale the image by this scalar, eg: [100 // 8 , 100 // 8] (default: 8)')

# Model parameters
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--filter-depth', type=int, default=32,
                    help='number of initial conv filter maps (default: 32)')
parser.add_argument('--reparam-type', type=str, default='isotropic_gaussian',
                    help='isotropic_gaussian, discrete or mixture [default: isotropic_gaussian]')
parser.add_argument('--layer-type', type=str, default='dense',
                    help='dense or conv (default: dense)')
parser.add_argument('--continuous-size', type=int, default=6,
                    help='continuous latent size (6/2 units of this are used for [s, x, y]) (default: 6)')
parser.add_argument('--discrete-size', type=int, default=10,
                    help='discrete latent size (only used for mix + disc) (default: 10)')
parser.add_argument('--nll-type', type=str, default='bernoulli',
                    help='bernoulli or gaussian (default: bernoulli)')
parser.add_argument('--mut-reg', type=float, default=0.3,
                    help='mutual information regularizer [mixture only] (default: 0.3)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--vae-type', type=str, default='parallel',
                    help='vae type [sequential or parallel] (default: parallel)')
parser.add_argument('--use-relational-encoder', action='store_true',
                    help='uses a relational network as the encoder projection layer')
parser.add_argument('--restore', type=str, default=None,
                    help='path to a model to restore (default: None)')

# Optimizer
parser.add_argument('--optimizer', type=str, default="adamnorm",
                    help="specify optimizer (default: rmsprop)")

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


# Global counter
TOTAL_ITER = 0


def build_optimizer(model):
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adamnorm": AdamNormGrad,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "lbfgs": optim.LBFGS
    }
    # filt = filter(lambda p: p.requires_grad, model.parameters())
    # return optim_map[args.optimizer.lower().strip()](filt, lr=args.lr)
    return optim_map[args.optimizer.lower().strip()](
        model.parameters(), lr=args.lr
    )


def register_images(images, names, grapher, prefix="train"):
    ''' helper to register a list of images '''
    if isinstance(images, list):
        assert len(images) == len(names)
        for im, name in zip(images, names):
            register_images(im, name, grapher, prefix=prefix)
    else:
        images = torch.min(images, ones_like(images))
        grapher.register_single({'{}_{}'.format(prefix, names): images},
                                plot_type='imgs')

def register_plots(loss, grapher, epoch, prefix='train'):
    ''' helper to register all plots with *_mean and *_scalar '''
    for k, v in loss.items():
        if isinstance(v, map):
            register_plots(loss[k], grapher, epoch, prefix=prefix)

        if 'mean' in k or 'scalar' in k:
            key_name = k.split('_')[0]
            value = v.data[0] if not isinstance(v, (float, np.float32, np.float64)) else v
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


def generate_related(data, args):
    # first upsample the image and then downsample the upsampled version
    # this creates a 'blurry' related image making the problem tougher
    x_enlarged = F.upsample(data, (args.upsample_size, args.upsample_size), mode='bilinear')
    original_img_size = tuple(data.size()[-2:])                                                  # eg: [100, 100]
    ds_img_size = tuple(int(i) for i in np.asarray(original_img_size) // args.downsample_scale)  # eg: [12, 12]
    x_downsampled = F.upsample(F.upsample(data, ds_img_size, mode='bilinear'), # blur the crap out
                               original_img_size, mode='bilinear')             # of the original data
    return x_enlarged, x_downsampled


def execute_graph(epoch, model, data_loader, grapher, optimizer=None, prefix='test'):
    ''' execute the graph; when 'train' is in the name the model runs the optimizer '''
    model.eval() if not 'train' in prefix else model.train()
    assert optimizer is not None if 'train' in prefix else optimizer is None
    loss_map, params, num_samples = {}, {}, 0
    x_original, x_related = None, None

    for data, labels in data_loader:
        data = Variable(data).cuda() if args.cuda else Variable(data)
        labels = Variable(labels).cuda() if args.cuda else Variable(labels)
        if args.half:
            data = data.half()

        if 'train' in prefix:
            # zero gradients on optimizer
            optimizer.zero_grad()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            x_original, x_related = generate_related(data, args)

            # run the VAE + the DNN and gather the loss function
            pred_logits, vae_pred_logits, params = model(x_original, x_related)
            loss_t = model.loss_function(vae_pred_logits, pred_logits, x_related, labels, params)

            # compute accuracy and aggregate into map
            loss_t['accuracy_mean'] = softmax_accuracy(F.softmax(pred_logits, -1), labels, size_average=True)
            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += pred_logits.size(0)

        if 'train' in prefix:
            # compute bp and optimize
            loss_t['loss_mean'].backward()
            optimizer.step()

    loss_map = _mean_map(loss_map) # reduce the map to get actual means
    correct_percent = 100.0 * loss_map['accuracy_mean']

    print('{}[Epoch {}][{} samples]: Average loss: {:.4f}\tKLD: {:.4f}\tNLL: {:.4f}\tAcc: {:.4f}'.format(
        prefix, epoch, num_samples,
        loss_map['loss_mean'].data[0],
        loss_map['kld_mean'].data[0],
        loss_map['nll_mean'].data[0],
        correct_percent))

    # gather scalar values of reparameterizers (if they exist)
    reparam_scalars = model.vae.get_reparameterizer_scalars()

    # plot the test accuracy, loss and images
    register_plots({**loss_map, **reparam_scalars}, grapher, epoch=epoch, prefix=prefix)
    images = [data, F.upsample(x_related, (32, 32), mode='bilinear'),
              F.upsample(params['crop_imgs'], (32, 32), mode='bilinear'),
              model.vae.nll_activation(vae_pred_logits)]
    img_names = ['original_imgs', 'related_imgs', 'img_crops', 'vae_reconstructions']
    register_images(images, img_names, grapher, prefix=prefix)
    grapher.show()

    # return this for early stopping
    loss_val = loss_map['loss_mean'].detach().data[0]
    loss_map.clear()
    params.clear()
    return loss_val, correct_percent


def train(epoch, model, optimizer, loader, grapher, prefix='train'):
    ''' train loop helper '''
    return execute_graph(epoch, model, loader,
                         grapher, optimizer, 'train')


def test(epoch, model, loader, grapher, prefix='test'):
     ''' test loop helper '''
     return execute_graph(epoch, model, loader,
                          grapher, prefix='test')


def set_multigpu(saccader):
    ''' workaround for multi-gpu: needs members set '''
    print("NOTE: multi-gpu currently doesnt over much better perf")
    name_fn = saccader.get_name
    loss_fn = saccader.loss_function
    vae_obj = saccader.vae
    saccader = nn.DataParallel(saccader)
    setattr(saccader, 'get_name', name_fn)
    setattr(saccader, 'loss_function', loss_fn)
    setattr(saccader, 'vae', vae_obj)
    return saccader


def get_model_and_loader():
    ''' helper to return the model and the loader '''
    loader = get_loader(args, sequentially_merge_test=False)

    # append the image shape to the config & build the VAE
    args.img_shp =  loader.img_shp
    vae = ParallellyReparameterizedVAE(loader.img_shp,
                                       loader.output_size,
                                       kwargs=vars(args))

    # build the Variational Saccading module
    saccader = Saccader(vae, kwargs=vars(args))
    if args.half:
        saccader = network_to_half(saccader)
        register_nan_checks(saccader)

    if args.ngpu > 1:
        saccader = set_multigpu(saccader)

    # build the grapher object
    grapher = Grapher(env=saccader.get_name(),
                      server=args.visdom_url,
                      port=args.visdom_port)

    return [saccader, loader, grapher]


def lazy_generate_modules(model, img_shp):
    ''' Super hax, but needed for building lazy modules '''
    model.eval()
    chans = img_shp[0]
    downsampled = same_type(args.half, args.cuda)(args.batch_size, *img_shp).normal_()
    upsampled = same_type(args.half, args.cuda)(args.batch_size, chans,
                                                args.upsample_size,
                                                args.upsample_size).normal_()
    model(Variable(upsampled), Variable(downsampled))


def run(args):
    # collect our model and data loader
    model, loader, grapher = get_model_and_loader()

    # since some modules are lazy generated
    # we want to run a single fwd pass
    lazy_generate_modules(model, args.img_shp)

    # collect our optimizer
    optimizer = build_optimizer(model)

    # train the VAE on the same distributions as the model pool
    if args.restore is None:
        print("training current distribution for {} epochs".format(args.epochs))
        early = EarlyStopping(model, max_steps=80) if args.early_stop else None

        test_loss, test_acc = 0.0, 0.0
        for epoch in range(1, args.epochs + 1):
            train(epoch, model, optimizer, loader.train_loader, grapher)
            test_loss, test_acc = test(epoch, model, loader.test_loader, grapher)
            if args.early_stop and early(test_loss):
                early.restore() # restore and test+generate again
                test_loss, test_acc = test(epoch, model, loader.test_loader, grapher)
                break

        # plot config json to visdom
        grapher.vis.text(pprint.PrettyPrinter(indent=4).pformat(model.config),
                         opts=dict(title="config"))
        grapher.save() # save to json after distributional interval
    else:
        model.load(args.restore)
        test_loss, test_acc = test(epoch, model, loader.test_loader, grapher)

    # evaluate one-time metrics
    append_to_csv([test_loss], "{}_test_elbo.csv".format(args.uid))
    append_to_csv([test_acc], "{}_test_acc.csv".format(args.uid))


if __name__ == "__main__":
    run(args)
