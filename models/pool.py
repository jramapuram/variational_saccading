from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict

from optimizers.adamnormgrad import AdamNormGrad
from datasets.loader import get_loader, get_split_data_loaders
from datasets.utils import GenericLoader, simple_merger
from helpers.layers import View, Identity, flatten_layers, EarlyStopping
from helpers.metrics import softmax_accuracy
from helpers.utils import float_type, zeros_like, ones_like, \
    check_or_create_dir, int_type, long_type, num_samples_in_loader


def build_optimizer(parameters, args):
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
        parameters, lr=args.lr
    )


def train(i, epoch, model, optimizer, data_loader, args):
    '''  i : which submodel to train '''
    model.train()
    for batch_idx, (data, target) in enumerate(data_loader.train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # data, target = Variable(data), Variable(target)
        # if len(list(target.size())) > 1:  #XXX: hax
        #     target = torch.squeeze(target)
        data, target = Variable(data), Variable(target)
        if len(list(target.size())) > 1:  #XXX: hax
            target = torch.squeeze(target)

        target = target == i
        target = target.type(long_type(args.cuda))

        optimizer.zero_grad()

        # project to the output dimension
        output = model(data, which_model=i)
        loss = model.loss_function(output, target)
        correct = softmax_accuracy(output, target)

        # compute loss
        loss.backward()
        optimizer.step()

        # log every nth interval
        if batch_idx % args.log_interval == 0:
            # the total number of samples is different
            # if we have filtered using the class_sampler
            if hasattr(data_loader.train_loader, "sampler") \
               and hasattr(data_loader.train_loader.sampler, "num_samples"):
                num_samples = data_loader.train_loader.sampler.num_samples
            else:
                num_samples = len(data_loader.train_loader.dataset)

            print('[POOL_{}]Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(
                i, epoch, batch_idx * len(data), num_samples,
                100. * batch_idx * len(data) / num_samples,
                loss.data[0], correct))


def test(i, epoch, model, data_loader, args):
    model.eval()
    loss = []
    correct = []

    for data, target in data_loader.test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            if len(list(target.size())) > 1:  #XXX: hax
                target = torch.squeeze(target)

            target = target == i
            target = target.type(long_type(args.cuda))

            output = model(data, which_model=i)
            loss_t = model.loss_function(output, target)
            correct_t = softmax_accuracy(output, target)

            loss.append(loss_t.detach().cpu().data[0])
            correct.append(correct_t)

    loss = np.mean(loss)
    acc = np.mean(correct)
    print('\n[POOL_{} | {} samples]Test Epoch: {}\tAverage loss: {:.4f}\tAverage Accuracy: {:.4f}\n'.format(
        i, num_samples_in_loader(data_loader.test_loader), epoch, loss, acc))
    return loss, acc


def train_model_pool(args):
    if args.disable_sequential: # vanilla batch training
        loaders = get_loader(args, sequentially_merge_test=True)
        loaders = [loaders] if not isinstance(loaders, list) else loaders
    else: # classes split
        loaders = get_split_data_loaders(args, num_classes=10, sequentially_merge_test=True)

    # only operate over the number of meta models
    loaders = loaders[0:args.num_meta_models]
    for l in loaders:
        print("pool-train = ", len(l.train_loader.dataset),
              "pool-test = ", len(l.test_loader.dataset))

    model = ModelPool(input_shape=loaders[0].img_shp,
                      output_size=2,#output_size,
                      num_models=args.num_meta_models,
                      kwargs=vars(args))

    if isinstance(loaders, list): # has a sequential loader
        loaders = simple_merger(loaders, args.batch_size, args.cuda)

    if not model.model_exists:
        optimizer = [build_optimizer(list(m_i.parameters()) + list(pr_i.parameters()), args)
                     for m_i, pr_i in zip(model.models, model.project_to_class_models)]
        early_stop = [EarlyStopping(model, max_steps=10)
                      for _ in range(args.num_meta_models)]

        for i in range(args.num_meta_models):
            for epoch in range(1, args.epochs + 1):
                train(i, epoch, model, optimizer[i], loaders, args)
                loss, _ = test(i, epoch, model, loaders, args)
                if early_stop[i](loss):
                    early_stop[i].restore()
                    break

            # save the model
            model.save()

    # if we loaded test again and exit
    #_ = [test(i, -1, model, loaders, args) for i in range(args.num_meta_models)]
    del loaders
    return model


class ModelPool(nn.Module):
    def __init__(self, input_shape, output_size, num_models, activation_fn=F.elu, **kwargs):
        super(ModelPool, self).__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self.num_models = num_models
        self.num_layers = 2 #XXX: parameterize this later
        self.is_color = self.input_shape[0] > 1
        self.chans = 3 if self.is_color else 1
        self.activation_fn = activation_fn

        # grab the meta config
        self.config = kwargs['kwargs']

        # build K simple models, this is a list of lists
        # we will build a separate projection layer though
        self.models, self.project_to_class_models = [], []
        for _ in range(num_models):
            base, proj = self.create_simple_model()
            self.models.append(base)
            self.project_to_class_models.append(proj)

        self.models = nn.ModuleList(self.models)
        self.project_to_class_models = nn.ModuleList(self.project_to_class_models)
        self.model_exists = self.load()

    def create_simple_model(self):
        ''' TODO: maybe consider using conv later? '''
        input_size = int(np.prod(self.input_shape))
        base_model = nn.ModuleList([
            nn.Linear(input_size, input_size),
            nn.Linear(input_size, input_size)
        ])

        # a simple model to project to the space of y
        linear = nn.Sequential(
            nn.Linear(input_size, self.output_size)
        )

        if self.config['ngpu'] > 1:
            base_model = nn.DataParallel(base_model)
            linear = nn.DataParallel(linear)

        if self.config['cuda']:
            base_model.cuda()
            linear.cuda()

        return base_model, linear

    def load(self):
        # load the pool if it exists
        if os.path.isdir(".models"):
            model_filename = os.path.join(".models", self.get_name() + ".th")
            if os.path.isfile(model_filename):
                print("loading existing model pool")
                self.load_state_dict(torch.load(model_filename))
                return True

        return False

    def save(self, overwrite=False):
        # save the model if it doesnt exist
        check_or_create_dir(".models")
        model_filename = os.path.join(".models", self.get_name() + ".th")
        if not os.path.isfile(model_filename) or overwrite:
            print("saving existing model pool")
            torch.save(self.state_dict(), model_filename)

    def get_name(self):
        full_hash_str = "_dense_act{}_input{}_output{}_batch{}_lr{}_ngpu{}".format(
            str(self.activation_fn.__name__),
            str(self.input_shape),
            str(self.output_size),
            str(self.config['batch_size']),
            str(self.config['lr']),
            str(self.config['ngpu'])
        )

        full_hash_str = full_hash_str.strip().lower().replace('[', '')  \
                                                     .replace(']', '')  \
                                                     .replace(' ', '')  \
                                                     .replace('{', '') \
                                                     .replace('}', '') \
                                                     .replace(',', '_') \
                                                     .replace(':', '') \
                                                     .replace('(', '') \
                                                     .replace(')', '') \
                                                     .replace('\'', '')
        return 'pool_{}x{}_{}'.format(
            str(self.config['task']),
            str(self.num_models),
            full_hash_str
        )

    def loss_function(self, pred, target):
        return F.cross_entropy(input=pred, target=target)

    def gather_submodel_hard(self, single_cat):
        index = single_cat.type(int_type(self.config['cuda']))
        single_model = []
        for i, model in enumerate(self.models):
            for j, layer in enumerate(model):
                if index[j, i] == 1:
                    single_model.append(layer)

        return nn.ModuleList(single_model)

    def gather_submodel_soft(self, single_soft_cat):
        # build a single model and zero it out
        single_model, _ = self.create_simple_model()
        nn.utils.vector_to_parameters(nn.utils.parameters_to_vector(single_model.parameters())*0,
                                      single_model.parameters())

        # add the contribution of all the models
        for i, model in enumerate(self.models):  # iterate over all models
            for j, layer in enumerate(model):    # iterate over all the layers
                single_model_single_layer_params \
                    = nn.utils.parameters_to_vector(single_model[j].parameters())
                layer_params = nn.utils.parameters_to_vector(layer.parameters())
                update = single_model_single_layer_params + single_soft_cat[j, i] *layer_params
                nn.utils.vector_to_parameters(update, single_model[j].parameters())

        return single_model

    def gather_batch_of_submodels(self, categorical_tensor):
        ''' expects a categoical of [B, N, M]
            where N = #models, M = #layers

            returns: a batch of models '''
        if self.config['hard_model_selection']:
            batch_of_models = [self.gather_submodel_hard(row) for row in categorical_tensor]
        else:
            batch_of_models = [self.gather_submodel_soft(row) for row in categorical_tensor]

        return batch_of_models

    def forward_with_model(self, x, model):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        for layer in model:
            x = layer(x)
            x = self.activation_fn(x)

        return x

    def forward(self, x, which_model):
        model = self.models[which_model]
        x = self.forward_with_model(x, model)
        return self.project_to_class_models[which_model](x)
