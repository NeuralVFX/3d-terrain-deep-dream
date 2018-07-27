#!/usr/bin/env python
import argparse
from terrain_dream import *


parser = argparse.ArgumentParser()

parser.add_argument("cmd", help=argparse.SUPPRESS, nargs="*")
parser.add_argument('--obj_file', nargs='?', default='/geo/grid_256.obj', type=str)
parser.add_argument('--dem_file', nargs='?',
                    default='/data/USGS/USGS_NED_one_meter_x34y441_CO_Central_Western_2016_IMG_2018.img', type=str)
parser.add_argument('--disc_filters', nargs='?', default=512, type=int)
parser.add_argument('--disc_layers', nargs='?', default=4, type=int)
parser.add_argument('--lr_disc', nargs='?', default=.001, type=float)
parser.add_argument('--lr_mesh', nargs='?', default=.001, type=float)
parser.add_argument('--render_res', nargs='?', default=256, type=int)
parser.add_argument('--train_epoch', nargs='?', default=6, type=int)
parser.add_argument('--save_every', nargs='?', default=1, type=int)
parser.add_argument('--loader_workers', nargs='?', default=4, type=int)
parser.add_argument('--data_perc', nargs='?', default=1, type=float)
parser.add_argument('--save_root', nargs='?', default='austria', type=str)
parser.add_argument('--load_state', nargs='?', type=str)

params = vars(parser.parse_args())

# if load_state arg is not used, then train model from scratch
if __name__ == '__main__':
    p2p = TerrainDream(params)
    if params['load_state']:
        p2p.load_state(params['load_state'])
    else:
        print('Starting From Scratch')
    p2p.train()
