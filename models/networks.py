import math
import torch
from torch.utils.data import *
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter
from osgeo import gdal
import numpy as np
import cv2
import neural_renderer as nr


############################################################################
# Re-usable blocks
############################################################################


class Vert2Tri(nn.Module):
    # converts vertex colors into tri-strips for Neural Renderer, or back
    def __init__(self, conv=True):
        super(Vert2Tri, self).__init__()
        state = torch.load('models/vert2strip.json')

        if conv:
            self.uv_conv = nn.Conv2d(3, 48, 2, stride=[1], padding=[0], bias=False, dilation=[1])
        else:
            self.uv_conv = nn.ConvTranspose2d(48, 3, 2, stride=[1], padding=[0], bias=False, dilation=[1])
            state['weight'] = state['weight'].transpose(2,3) / 16

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


def make_xyz_grid(res):
    # build vertex and face data from scratch
    vert_id_weights = [[1, 0], [0, 0], [0, 1], [0, 0], [0, 0], [0, 1], [1, 0], [0, 0], [0, 0], [0, 1], [0, 0], [1, 0]]
    weights = np.asarray(vert_id_weights, dtype='float32').reshape(6, 1, 2, 2)
    vert_id_weights = torch.FloatTensor(weights).cuda()
    faces_conv = nn.Conv2d(1, 6, 2, stride=[1], padding=[0], bias=False, dilation=[1]).cuda()
    faces_conv.weight = nn.Parameter(vert_id_weights)

    # use convolution to convert vert_ids to face centric vert id map
    x = torch.FloatTensor(np.arange(res * res).reshape(res, res)).cuda()
    vert_ids = faces_conv(x.view(1, 1, res, res))
    vert_ids = vert_ids.permute(0, 2, 3, 1).contiguous().view(1, (res - 1) * (res - 1) * 2, 3).int()

    # build XYZ vertex coordinate grid
    coords = np.arange(-1, 1 + 2.0 / (res - 1), 2.0 / (res - 1))
    grid = np.concatenate([np.meshgrid(coords, coords)[0].reshape(1, res, res),
                           np.zeros([1, res, res]),
                           np.moveaxis(-np.meshgrid(coords, coords)[0].reshape(1, res, res), (1, 2), (2, 1))],
                          axis=0)
    vert_grid = torch.FloatTensor(grid).cuda()
    return vert_ids, vert_grid


class Model(nn.Module):
    # Loaded for DEM file, and storage for out textures
    def __init__(self, res,dem):
        super(Model, self).__init__()
        print(f'Building Grid: {res}x{res}')
        faces, vertices  = make_xyz_grid(res)
        self.faces = faces
        self.res = res

        # Load_dem
        print(f'Loading DEM: {dem}')
        geo = gdal.Open(dem)
        geo_arr = geo.ReadAsArray()
        scaled_down = cv2.resize(geo_arr, (self.res, self.res), interpolation=cv2.INTER_AREA)
        scaled_down = ((scaled_down - scaled_down.mean()) / scaled_down.std()) * .2

        # Replace grid y axis with height from DEM
        cat_list = [vertices[:1, :, :], scaled_down.reshape([1, self.res, self.res]), vertices[2:, :, :]]
        vertices = np.concatenate(cat_list, axis=0)
        vertices = torch.FloatTensor(vertices).cuda()
        self.vertices = nn.Parameter(vertices)

        # Initialize random textures
        v2t = Vert2Tri()
        v2t.cuda()
        blurry_noise = gaussian_filter(np.random.normal(0, 1, (3, self.res, self.res)), sigma=3)
        textures = torch.FloatTensor(blurry_noise).cuda()
        textures = v2t(textures.unsqueeze(0))

        self.textures = nn.Parameter(textures.permute(0, 2, 3, 1).contiguous().view(1, self.faces.shape[1], 2, 2, 2, 3))

    def forward(self):
        return self.textures, self.vertices, self.faces


class Render(nn.Module):
    # Neural renderer, ready to receive new lighting direction and color during forward pass
    def __init__(self, res=128):
        super(Render, self).__init__()
        renderer = nr.Renderer(camera_mode='look_at',
                               image_size=res,
                               orig_size=res*2,
                               viewing_angle=20)

        renderer.light_intensity_directional = .9
        renderer.light_intensity_ambient = .75
        self.renderer = renderer
        self.rotation_array = self.generate_rotation_array(1)

    def generate_rotation_array(self, batch_size):
        rot_list = []
        for a in range(batch_size):
            rotation_matrix = cv2.getRotationMatrix2D((0, 0), (a / batch_size) * 360, 1)
            rotation_matrix = np.array([[rotation_matrix[0][0], 0, rotation_matrix[0][1]],
                                        [0, 1, 0],
                                        [rotation_matrix[1][0], 0, rotation_matrix[1][1]]])

            rot_list.append(np.copy(rotation_matrix))

        batch_rot = torch.FloatTensor(np.array(rot_list)).cuda()
        batch_rot.requires_grad = False
        return batch_rot

    def forward(self,
                vertices,
                faces,
                textures,
                eye,
                light_dir=[.5, .5, .5],
                light_color_directional=[.8, 1, .7],
                light_color_ambient=[1, 1.2, 1.2],
                batch_size=1):

        if batch_size != self.rotation_array.shape[0]:
            self.rotation_array = self.generate_rotation_array(batch_size)

        self.renderer.light_color_directional = light_color_directional
        self.renderer.light_direction = light_dir
        self.renderer.light_color_ambient = light_color_ambient
        self.renderer.eye = eye

        transformerd_verts = torch.matmul(vertices.expand(batch_size, -1, -1), self.rotation_array)
        return self.renderer(transformerd_verts,
                             faces.expand(batch_size, -1, -1),
                             textures.expand(batch_size, -1, -1, -1, -1, -1))


class Discriminator(nn.Module):
    def __init__(self, channels=3, filts=512, kernel_size=4, layers=5):
        super(Discriminator, self).__init__()

        operations = []
        out_operations = [nn.Conv2d(in_channels=filts, out_channels=1, padding=1, kernel_size=kernel_size, stride=1)]

        # Build up discriminator backwards based on final filter count
        for a in range(layers):
            operations += [ConvTrans(ic=int(filts // 2), oc=filts, kernel_size=kernel_size, block_type='down')]
            filts = int(filts // 2)

        operations.reverse()
        in_operations = [nn.ReflectionPad2d(3),
                         nn.Conv2d(in_channels=channels, out_channels=filts, kernel_size=7, stride=1)]

        operations = in_operations + operations + out_operations
        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        return self.operations(x)
