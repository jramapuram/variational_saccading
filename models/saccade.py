import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from copy import deepcopy

from helpers.utils import expand_dims, check_or_create_dir, \
    zeros_like, int_type, nan_check_and_break, zeros, get_dtype, \
    plot_tensor_grid, add_noise_to_imgs
from helpers.layers import View, Identity, get_encoder, str_to_activ_module
from .image_state_projector import ImageStateProjector
from .numerical_diff_module import NumericalDifferentiator
from .spatial_transformer import SpatialTransformer
from datasets.crop_dual_imagefolder import CropLambdaPool, USE_LIB


class Saccader(nn.Module):
    def __init__(self, vae, output_size, **kwargs):
        super(Saccader, self).__init__()
        self.vae = vae
        self.output_size = output_size
        self.config = kwargs['kwargs']

        # build the pool object to multi-process images
        if USE_LIB == 'rust':
            self.pool = CropLambdaPool(window_size=self.config['window_size'],
                                       chans=self.vae.chans,
                                       max_image_percentage=self.config['max_image_percentage'])
        else:
            self.pool = CropLambdaPool(num_workers=self.config['batch_size'])
            #self.pool = CropLambdaPool(num_workers=1)  # disables
            #self.pool = CropLambdaPool(num_workers=-1) # auto-pool
            #self.pool = CropLambdaPool(num_workers=24) # fixed pool


        # build the projection to softmax from RNN state
        self.latent_projector = ImageStateProjector(output_size=self.output_size,
                                                    config=self.config)

        # build the inlay projector
        self.numgrad = NumericalDifferentiator(self.config)

        # build the spatial transformer
        self.spatial_transformer = SpatialTransformer(self.config)

        # build the projector from the posterior to the ST params
        self.posterior_to_st = nn.Sequential(self._get_dense('posterior_to_st_proj')(
            self.vae.reparameterizer.output_size, 3,
            normalization_str=self.config['dense_normalization'],
            activation_fn=str_to_activ_module(self.config['activation'])
        )) if self.vae.reparameterizer.output_size != 3 else Identity()

        # self.posterior_to_st = nn.Linear(self.vae.reparameterizer.output_size, 3) \
        #     if self.vae.reparameterizer.output_size != 3 else Identity()

    def _get_dense(self, name):
        config = deepcopy(self.config)
        config['encoder_layer_type'] = 'dense'
        return get_encoder(config, name=name)

    def fp16(self):
        self.latent_projector.fp16()
        self.numgrad.fp16()
        self.vae.fp16()
        self.spatial_transformer = self.spatial_transformer.half()

    def parallel(self):
        self.latent_projector.parallel()
        self.numgrad.parallel()
        self.vae.parallel()
        self.spatial_transformer = nn.DataParallel(self.spatial_transformer)

    def get_name(self):
        return "{}_win{}_us{}_dscale{}_{}".format(
            str(self.config['uid']),
            str(self.config['window_size']),
            str(self.config['synthetic_upsample_size']) if self.config['synthetic_upsample_size'] > 0 else "",
            str(self.config['downsample_scale']),
            self.vae.get_name()
        )

    def load(self, filename=None):
        def _load(model_filename):
            # load the pool if it exists
            if os.path.isfile(model_filename):
                print("loading existing saccader model from {}".format(
                    model_filename
                ))
                self = torch.load(model_filename)
                return True

            return False

        if filename is None:
            assert os.path.isdir(".models")
            model_filename = os.path.join(".models", self.get_name() + ".th")
            return _load(model_filename)

        return _load(filename)

    def save(self, overwrite=False):
        # save the model if it doesnt exist
        check_or_create_dir(".models")
        model_filename = os.path.join(".models", self.get_name() + ".th")
        if not os.path.isfile(model_filename) or overwrite:
            print("saving existing saccader model")
            torch.save(self, model_filename)

    def get_input_imgs(self, batch_size, input_imgs_map, resize=(128, 128)):
        ''' helper to take the input image map and return visdom friendly plots '''
        return_map = {}
        for name, img in input_imgs_map.items():
            if isinstance(img, torch.Tensor):
                return_map[name] = F.interpolate(img, resize, mode='bilinear')
            else: # we are in the case of lambda operands
                z = np.tile(np.array([[1.0, 0.0, 0.0]]), (batch_size, 1))
                #original_full_img, _, _ = self._pool_to_imgs(z, img, override=True)
                original_full_img = self._pool_to_imgs(z, img, override=True)
                return_map[name] = F.interpolate(original_full_img, resize, mode='bilinear')

        return return_map


    def get_auxiliary_imgs(self, output_map, crop_resize=(64, 64), decode_resize=(32, 32)):
        ''' simple helper to go through the output map and grab all the images and labels'''
        imgs_map = {}
        assert 'crops' in output_map and isinstance(output_map['crops'], list) # sanity
        for i, (crop, decoded) in enumerate(zip(output_map['crops'],
                                                output_map['decoded'])):
            imgs_map['softcrop{}_imgs'.format(i)] = F.interpolate(
                crop, crop_resize, mode='bilinear') # grab softcrops
            imgs_map['decoded{}_imgs'.format(i)] = F.interpolate(
                self.vae.nll_activation(decoded),   # grab decodes
                decode_resize, mode='bilinear')

        # add the inlays and hard-crops here because they might not exist
        # for the case of just using a vanilla spatial-transformer
        for i, (crop_true, inlay) in enumerate(zip(output_map['crops_true'],
                                                   output_map['inlays'])):
            imgs_map['hardcrop{}_imgs'.format(i)] = F.interpolate(
                crop_true, crop_resize, mode='bilinear')   # grab hardcrops
            imgs_map['inlay{}_imgs'.format(i)] = F.interpolate(
                inlay, decode_resize, mode='bilinear')     # grab the inlays

        return imgs_map

    def get_imgs(self, batch_size, output_map, input_imgs_map):
        ''' simple helper to concat the output-map images with the input images'''
        return{**self.get_auxiliary_imgs(output_map),
               **self.get_input_imgs(batch_size, input_imgs_map)}

    def loss_function(self, x, labels, output_map):
        ''' loss is: L_{classifier} * L_{VAE} '''
        vae_loss_map = self.vae.loss_function(output_map['decoded'],
                                              [x.clone() for _ in range(len(output_map['decoded']))],
                                              output_map['params'])

        # get classifier loss, use BCE with logits if multi-dimensional
        loss_fn = F.binary_cross_entropy_with_logits \
            if len(labels.shape) > 1 else F.cross_entropy
        pred_loss = loss_fn(
            input=output_map['preds'],
            target=labels,
            reduction='none'
        )
        pred_loss = torch.sum(pred_loss, -1) if len(pred_loss.shape) > 1 else pred_loss
        nan_check_and_break(pred_loss, "pred_loss")

        # the crop prediction loss [ only for lambda images ]
        # crop_loss = self.numgrad.loss_function(
        #     output_map['crops'],
        #     output_map['crops_true']
        # ) if len(output_map['crops_true']) > 0 else torch.zeros_like(pred_loss)
        crop_loss = torch.zeros_like(pred_loss)

        # ACT loss
        vae_loss_map['saccades_scalar'] = output_map['saccades_scalar']
        act_loss = torch.zeros_like(pred_loss)
        # act_loss = F.binary_cross_entropy_with_logits(
        #     input=output_map['act'],
        #     target=torch.ones_like(output_map['act']),
        #     reduction='none'
        # )

        # TODO: try multi-task loss, ie:
        # vae_loss_map['loss'] = vae_loss_map['loss'] + (pred_loss + crop_loss + act_loss)
        vae_loss_map['loss'] = vae_loss_map['loss'] * (pred_loss + crop_loss + act_loss)

        # compute the means for visualizations of bp
        vae_loss_map['act_loss_mean'] = torch.mean(act_loss)
        vae_loss_map['pred_loss_mean'] = torch.mean(pred_loss)
        vae_loss_map['crop_loss_mean'] = torch.mean(crop_loss)
        vae_loss_map['loss_mean'] = torch.mean(vae_loss_map['loss'])
        return vae_loss_map

    def _z_to_image_transformer(self, z, imgs):
        imgs = imgs.type(z.dtype)
        z_proj = self.posterior_to_st(z)
        crops_pred = self.spatial_transformer(z_proj, imgs)
        nan_check_and_break(crops_pred, "predicted_crops_t")
        return {
            'crops_pred': crops_pred
        }

    def _clamp_sample(self, z):
        ''' clamps to [0, 1] '''
        clamp_map = {
            'beta': lambda z: z,                          # no change
            'isotropic_gaussian': lambda z: (z + 3) / 6,  # re-normalize to [0, 1]
            'mixture': lambda z: (                        # re-normalize to [0, 1]
                z[:, 0:self.vae.reparameterizer.num_continuous_input] + 3) / 6
        }
        return torch.clamp(clamp_map[self.config['reparam_type']](z), 0.0, 1.0)

    def _invert_clamp_sample(self, z):
        ''' inverts clamp to [-3, 3] '''
        clamp_map = {
            'beta': lambda z: (z * 6) - 3,               # re-normalize to [-3, 3]
            'isotropic_gaussian': lambda z: (z * 6) - 3, # re-normalize to [-3, 3]
            'mixture': lambda z: (                       # re-normalize to [-3, 3]
                z[:, 0:self.vae.reparameterizer.num_continuous_input] * 6) - 3
        }
        return clamp_map[self.config['reparam_type']](z)


    def _z_to_image_lambda(self, z, imgs):
        ''' run the z's through the threadpool, returning inlays and true crops'''
        clamped_z = self._clamp_sample(z[:, 0:3])  # clamp to [0, 1] range
        crops = self._pool_to_imgs(clamped_z, imgs)
        predicted_crops = self.numgrad(z, crops)

        nan_check_and_break(predicted_crops, "predicted_crops_t")

        return {
            'crops_pred': predicted_crops
        }

    def _pool_to_imgs(self, z, imgs, override=False):
        # tabulate the crops using the threadpool and re-stitch together
        z_np = z.clone().detach().cpu().numpy() if not isinstance(z, np.ndarray) else z
        pool_tabulated = self.pool(imgs, z_np, override=override)
        crops = torch.cat(pool_tabulated, 0)

        # memory cleanups
        del pool_tabulated

        # push to cuda if necessary
        is_cuda = z.is_cuda if isinstance(z, torch.Tensor) else False
        return crops.cuda() if is_cuda else crops

    def _z_to_image(self, z, imgs):
        ''' accepts images (or lambdas to crop) and z (gauss or disc)
            and returns an [N, C, H_trunc, W_trunc] array '''
        assert len(imgs) == z.shape[0], "batch sizes for crop preds vs imgs dont match"
        crop_fn_map = {
            'transformer': self._z_to_image_transformer,
            'lambda': self._z_to_image_lambda
        }
        crop_type = 'transformer' if isinstance(imgs, torch.Tensor) else 'lambda'
        return crop_fn_map[crop_type](z, imgs)

    def generate(self, batch_size):
        self.eval()
        samples = []
        with torch.no_grad():
            for i in range(self.config['max_time_steps']):
                samples.append(
                    self.vae.generate_synthetic_samples(batch_size, reset_state=i==0)
                )

        return samples

    def forward(self, x, x_related):
        ''' encode with VAE, then use posterior sample to
            gather image from original space (eg: 4k image) '''
        batch_size, chans = x_related.size(0), x_related.size(1)

        def _forward_internal(x_related, inference_only=False):
            params, crops, crops_true, inlays, decodes = [], [], [], [], []

            # reset the state, output and the truncate window
            self.vae.memory.init_state(batch_size, cuda=x_related.is_cuda)
            self.vae.memory.init_output(batch_size, seqlen=1, cuda=x_related.is_cuda)

            # accumulator for predictions and ACT
            x_preds = zeros((batch_size, self.config['latent_size']),
                            cuda=x_related.is_cuda,
                            dtype=get_dtype(x_related)).requires_grad_()
            act = zeros((batch_size, 1), cuda=x_related.is_cuda,
                        dtype=get_dtype(x_related)).squeeze().requires_grad_()

            for i in range(self.config['max_time_steps']):
                # get posterior and params, expand 0'th dim for seqlen
                x_related_inference = add_noise_to_imgs(x_related) \
                    if self.config['add_img_noise'] else x_related
                z_t, params_t = self.vae.posterior(x_related_inference)

                nan_check_and_break(x_related_inference, "x_related_inference")
                nan_check_and_break(z_t['prior'], "prior")
                nan_check_and_break(z_t['posterior'], "posterior")
                nan_check_and_break(z_t['x_features'], "x_features")

                # extract the required crop from original image
                x_trunc_t = self._z_to_image(z_t['posterior'], x)

                # do preds and sum
                state = torch.mean(self.vae.memory.get_state()[0], 0)
                crops_pred_perturbed = add_noise_to_imgs(x_trunc_t['crops_pred']) \
                    if self.config['add_img_noise'] else x_trunc_t['crops_pred']
                state_proj = self.latent_projector(crops_pred_perturbed, state)
                x_preds = x_preds + state_proj[:, 0:-1] # last bit is for ACT

                # decode the posterior
                decoded_t = self.vae.decode(z_t, produce_output=True)
                nan_check_and_break(decoded_t, "decoded_t")

                # cache for loss function & visualization
                params.append(params_t)
                crops.append(x_trunc_t['crops_pred'])
                decodes.append(decoded_t)

                # only add these if we are in the lambda setting:
                if 'crops_true' in x_trunc_t and 'inlay' in x_trunc_t:
                    crops_true.append(x_trunc_t['crops_true'])
                    inlays.append(x_trunc_t['inlay'])

                # conditionally break away based on ACT
                # act = act + torch.sigmoid(state_proj[:, -1])
                # if torch.max(act / max(i, 1)) >= 0.9998:
                #     break

            preds = self.latent_projector.get_output(
                x_preds / self.config['max_time_steps']
            )
            return {
                'act': act / max(i, 1),
                'saccades_scalar': i,
                'decoded': decodes,
                'params': params,
                'preds': preds,
                'inlays': inlays,
                'crops': crops,
                'crops_true': crops_true
            }

        # forward pass with decoding [normal]
        standard_forward_pass = _forward_internal(x_related, inference_only=False)

        #  re-eval posterior for mut-info using decoded from above
        if (self.config['continuous_mut_info'] > 0
            or self.config['discrete_mut_info'] > 0):
            ''' append the posterior of re-forward passing'''
            q_z_given_x_hat = _forward_internal(standard_forward_pass['decoded'],
                                                inference_only=True)
            for param, q_z_given_xhat_param in zip(standard_forward_pass['params'],
                                                   q_z_given_x_hat['params']):
                param['posterior'] = {
                    **param['posterior'],
                    'q_z_given_xhat': q_z_given_xhat_param['posterior']
                }

        # return both the standard forward pass and the mut-info one
        # after clearing the cached memory
        self.vae.memory.clear()
        return standard_forward_pass
