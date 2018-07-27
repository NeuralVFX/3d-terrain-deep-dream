from scipy.ndimage.filters import gaussian_filter
from osgeo import gdal
from torch.utils.data import *
import torch
import numpy as np
import cv2
import torch.nn as nn
import neural_renderer as nr


############################################################################
# Re-usable blocks
############################################################################


class Vert2Tri(nn.Module):
    # converts vertex colors into tri-strips for Neural Renderer
    def __init__(self):
        super(Vert2Tri, self).__init__()
        self.uv_conv = nn.Conv2d(3, 48, 2, stride=[1], padding=[0], bias=False, dilation=[1])
        state = torch.load('vert2strip.json')
        self.uv_conv.load_state_dict(state)

        for a in self.uv_conv.parameters():
            a.requires_grad = False

    def forward(self, x): return self.uv_conv(x)


class ConvTrans(nn.Module):
    # One Block to be used as conv and transpose throughout the model
    def __init__(self, ic=4, oc=4, kernel_size=4, block_type='res', padding=1, stride=2, drop=.01, use_bn=True,
                 relu=True):
        super(ConvTrans, self).__init__()
        self.block_type = block_type

        operations = []

        if relu:
            operations += [nn.LeakyReLU(.2)]

        if self.block_type == 'down':
            operations += [nn.Conv2d(in_channels=ic, out_channels=oc, padding=padding, kernel_size=kernel_size,
                                     stride=stride, bias=False)]

        else:
            operations += [nn.ConvTranspose2d(in_channels=ic, out_channels=oc, padding=padding, kernel_size=kernel_size,
                                              stride=stride, bias=False)]

        if use_bn:
            operations += [nn.InstanceNorm2d(oc)]

        operations += [nn.Dropout(drop)]

        self.block = nn.Sequential(*operations)

    def forward(self, x):
        return self.block(x)


############################################################################
# DEM Loader,   Renderer,   Discriminator
############################################################################


class Model(nn.Module):
    # Loaded for DEM file, and storage for out textures
    def __init__(self, geo, dem, res):
        super(Model, self).__init__()
        vertices, faces = nr.load_obj(geo)
        vertices = vertices[None, :, :]
        vertices = vertices.transpose(0, 2).view(3, res, res)
        self.faces = faces[None, :, :]

        # Load_dem
        geo = gdal.Open(dem)
        geo_arr = geo.ReadAsArray()
        scaled_down = cv2.resize(geo_arr, (res, res), interpolation=cv2.INTER_AREA)
        scaled_down = ((scaled_down - scaled_down.mean()) / scaled_down.std()) * .2

        # Replace grid y axis with height from DEM
        cat_list = [vertices[:1, :, :], scaled_down.reshape([1, res, res]), vertices[2:, :, :]]
        vertices = np.concatenate(cat_list, axis=0)
        vertices = torch.FloatTensor(vertices).cuda()
        self.vertices = vertices

        # Initialize random textures
        blurry_noise = gaussian_filter(np.random.normal(0, .5, (3, res, res)), sigma=3)
        textures = torch.FloatTensor(blurry_noise).cuda()
        self.textures = nn.Parameter(textures)

    def forward(self):
        return self.textures, self.vertices, self.faces


class Render(nn.Module):
    # Neural renderer, ready to receive new lighting direction and color during forward pass
    def __init__(self, res=128):
        super(Render, self).__init__()
        renderer = nr.Renderer(camera_mode='look_at',
                               image_size=res,
                               orig_size=256)

        renderer.light_intensity_directional = .9
        renderer.light_intensity_ambient = .75
        self.renderer = renderer

    def forward(self,
                vertices,
                faces,
                textures,
                eye,
                light_dir=[.5, .5, .5],
                light_color_directional=[.8, 1, .7],
                light_color_ambient=[1, 1.2, 1.2]):

        self.renderer.light_color_directional = light_color_directional
        self.renderer.light_direction = light_dir
        self.renderer.light_color_ambient = light_color_ambient
        self.renderer.eye = eye
        return self.renderer(vertices, faces, textures)


class Discriminator(nn.Module):
    def __init__(self, channels=3, filts=512, kernel_size=4, layers=5):
        super(Discriminator, self).__init__()

        operations = []
        out_operations = [nn.Conv2d(in_channels=filts, out_channels=1, padding=1, kernel_size=kernel_size, stride=1)]

        # Build up discriminator backwards based on final filter count
        for a in range(layers):
            if a == layers - 1:
                operations += [ConvTrans(ic=int(filts // 2), oc=filts, kernel_size=kernel_size, block_type='down')]
            else:
                operations += [ConvTrans(ic=int(filts // 2), oc=filts, kernel_size=kernel_size, block_type='down')]
            filts = int(filts // 2)

        operations.reverse()
        in_operations = [nn.ReflectionPad2d(3),
                         nn.Conv2d(in_channels=channels, out_channels=filts, kernel_size=7, stride=1)]

        operations = in_operations + operations + out_operations
        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        return self.operations(x)
