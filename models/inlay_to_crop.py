import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from helpers.layers import ResnetBlock, get_decoder, str_to_activ_module


class InlayToCropProjector(nn.Module):
    ''' takes an inlay and a latent vector(z) and returns a crop prediction '''
    def __init__(self, config):
        super(InlayToCropProjector, self).__init__()
        self.config = config
        self.z_to_img_net, self.joint_net = self._build_model()

    def parallel(self):
        self.z_to_img_net = nn.DataParallel(self.z_to_img_net)
        self.joint_net = nn.DataParallel(self.joint_net)

    def _build_model(self):
        assert self.config['differentiable_image_size'] == 128, "only supports 128x128 imgs"
        assert self.config['window_size'] == 32, "can only operate over window size of 32"

        z_to_img = nn.Sequential(
            get_decoder(self.config, reupsample=True, name='z_to_img_decoder')(
                input_size=3, num_layers=6, # num_layers=6 takes output to [#chans, 71, 71], then upsample
                output_shape=[self.config['img_shp'][0], # channels derived from original
                              self.config['differentiable_image_size'],
                              self.config['differentiable_image_size']],
                activation_fn=str_to_activ_module(self.config['activation'])
            ),
            str_to_activ_module(self.config['activation'])()
        )

        # accepts z-projected to an image and the inlay
        # and projects that to the window size
        chans = self.config['img_shp'][0] * 2

        joint_conv = nn.Sequential(
            ResnetBlock(chans, 64, stride=2, downsample=True, activation_str=self.config['activation']),
            ResnetBlock(64, 32, stride=2, downsample=True, activation_str=self.config['activation']),
            nn.Conv2d(32, int(chans / 2), 1, bias=False) # non-activated output
        )

        # joint_conv = nn.Sequential(
        #     ResnetBlock(chans, 32, stride=1, downsample=False, activation_str=self.config['activation']),
        #     ResnetBlock(32, 64, stride=2, downsample=True, activation_str=self.config['activation']),
        #     ResnetBlock(64, 32, stride=2, downsample=True, activation_str=self.config['activation']),
        #     nn.Conv2d(32, int(chans / 2), 1, bias=False) # non-activated output
        # )

        # joint_conv = nn.Sequential(
        #     ResnetBlock(chans, 32, stride=1, downsample=False, activation_str=self.config['activation']),
        #     #ResnetBlock(32, 32, stride=1, downsample=False, activation_str=self.config['activation']),

        #     ResnetBlock(32, 64, stride=2, downsample=True, activation_str=self.config['activation']),
        #     #ResnetBlock(64, 64, stride=1, downsample=False, activation_str=self.config['activation']),

        #     ResnetBlock(64, 128, stride=2, downsample=True, activation_str=self.config['activation']),
        #     #ResnetBlock(128, 128, stride=1, downsample=False, activation_str=self.config['activation']),

        #     ResnetBlock(128, 64, stride=1, downsample=False, activation_str=self.config['activation']),
        #     #ResnetBlock(64, 64, stride=1, downsample=False, activation_str=self.config['activation']),

        #     ResnetBlock(64, 32, stride=1, downsample=False, activation_str=self.config['activation']),
        #     #ResnetBlock(32, chans, stride=1, downsample=False, activation_str=self.config['activation'])
        #     nn.Conv2d(32, int(chans / 2), 1, bias=False) # non-activated output
        # )

        return z_to_img, joint_conv

    def loss_function(self, crop_pred, crop):
        crop_pred = torch.cat([cp.unsqueeze(1) for cp in crop_pred], 1)
        crop = torch.cat([cp.unsqueeze(1) for cp in crop], 1)
        return F.mse_loss(input=crop_pred, target=crop, size_average=True)

    def forward(self, inlay, z):
        '''concats the image produced by the conv-decoder
           and the inlay and projects '''
        z_img = self.z_to_img_net(z)
        jc =  self.joint_net(torch.cat([z_img, inlay], 1))
        return jc
