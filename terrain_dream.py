import torch.optim as optim
import torch
import time
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import *
import neural_renderer as nr
import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from util import helpers as helper
from util import loaders as load
from models import networks as n


def random_eye_and_light():
    light_dir = [(random.random() - .5) * 2,
                  random.random(),
                  (random.random() - .5) * 2]

    light_color_directional = [(random.random() * .5) + 1.3,
                                (random.random() * .5) + 1.3,
                                (random.random() * .5) + 1]

    eye = nr.get_points_from_angles((random.random() * .5) + 1.5,
                                     (random.random() * 10) + 20,
                                     random.random() * 360)

    return light_dir, light_color_directional, eye


############################################################################
# Train
############################################################################

class TerrainDream:
    """
    Example usage if not using command line:


    """

    def __init__(self, params):
        self.params = params
        self.model_dict = {}
        self.opt_dict = {}
        self.current_epoch = 0
        self.current_iter = 0
        self.loop_iter = 0

        self.transform = load.NormDenorm([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        self.train_loader, self.data_len = load.data_load(self.transform,
                                                          1,
                                                          shuffle=True,
                                                          output_res=params["render_res"],
                                                          perc=params['test_perc'])

        print(f'Data Loader Initialized: {self.data_len} Images')

        self.model_dict["M"] = n.Model(params["obj_file"],
                                       params["dem_file"])

        self.model_dict["D"] = n.Discriminator(channels=3,
                                               filts=params["disc_filters"],
                                               kernel_size=4,
                                               layers=params["disc_layers"])
        self.v2t = n.Vert2Tri()
        self.render = n.Render(res=params["render_res"])

        self.v2t.cuda()
        self.render.cuda()
        self.model_dict["D"].apply(helper.weights_init_normal)

        for i in self.model_dict.keys():
            self.model_dict[i].cuda()
            self.model_dict[i].train()
        print('Networks Initialized')

        self.l1_loss = nn.L1Loss()

        self.light_dir, self.light_color_directional, self.eye = random_eye_and_light()

        # setup optimizers #
        self.opt_dict["M"] = optim.RMSprop(self.model_dict["M"].parameters(), lr=params['lr_mesh'])
        self.opt_dict["D"] = optim.RMSprop(self.model_dict["D"].parameters(), lr=params['lr_disc'])
        print('Optimizers Initialized')

        # setup history storage #
        self.losses = ['M_Loss', 'D_Loss']
        self.loss_batch_dict = {}
        self.loss_epoch_dict = {}
        self.train_hist_dict = {}

        for loss in self.losses:
            self.train_hist_dict[loss] = []
            self.loss_epoch_dict[loss] = []
            self.loss_batch_dict[loss] = []

    def load_state(self, filepath):
        # Load previously saved sate from disk, including models, optimizers and history
        state = torch.load(filepath)
        self.current_iter = state['iter'] + 1
        self.current_epoch = state['epoch'] + 1
        for i in self.model_dict.keys():
            self.model_dict[i].load_state_dict(state['models'][i])
        for i in self.opt_dict.keys():
            self.opt_dict[i].load_state_dict(state['optimizers'][i])
        self.train_hist_dict = state['train_hist']

    def save_state(self, filepath):
        # Save current state of all models, optimizers and history to disk
        out_model_dict = {}
        out_opt_dict = {}
        for i in self.model_dict.keys():
            out_model_dict[i] = self.model_dict[i].state_dict()
        for i in self.opt_dict.keys():
            out_opt_dict[i] = self.opt_dict[i].state_dict()
        model_state = {'iter': self.current_iter,
                       'epoch': self.current_epoch,
                       'models': out_model_dict,
                       'optimizers': out_opt_dict,
                       'train_hist': self.train_hist_dict}
        torch.save(model_state, filepath)
        return f'Saving State at Iter:{self.current_iter}'

    def display_history(self):
        # Draw history of losses, called at end of training
        fig = plt.figure()
        for key in self.losses:
            x = range(len(self.train_hist_dict[key]))
            if len(x) > 0:
                plt.plot(x, self.train_hist_dict[key], label=key)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend(loc=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'output/{self.params["save_root"]}_loss.jpg')
        plt.show()
        plt.close(fig)

    def set_grad_req(self, d=True, g=True):
        # Easily enable and disable gradient storage per model
        for par in self.model_dict["D"].parameters():
            par.requires_grad = d
        for par in self.model_dict["M"].parameters():
            par.requires_grad = g

    def get_eye_and_light(self):
        # create random lighting and camera for render, stochastic for 600 epochs, then only changes every 300
        iter_n = self.loop_iter
        if iter_n % 300 == 0 or (self.current_iter < 600 and self.current_epoch == 0):
            self.light_dir, self.light_color_directional, self.eye = random_eye_and_light()
        return self.light_dir, self.light_color_directional, self.eye

    def train_disc(self, fake, real):
        # train discriminator on fake and real image
        self.opt_dict["D"].zero_grad()
        disc_result_fake = self.model_dict["D"](fake.detach())
        disc_result_real = self.model_dict["D"](real)
        disc_loss_fake = self.l1_loss(disc_result_fake, torch.zeros_like(disc_result_fake)) ** 2
        disc_loss_real = self.l1_loss(disc_result_real, torch.ones_like(disc_result_fake)) ** 2
        self.loss_batch_dict['D_Loss'] = (disc_loss_fake + disc_loss_real) * .5
        self.loss_batch_dict['D_Loss'].backward()
        self.opt_dict["D"].step()

    def dream_on_mesh(self):
        # create render, and use discriminator to dream, backprop to the mesh#
        self.opt_dict["M"].zero_grad()

        tex, vert, face = self.model_dict["M"]()
        tex_a = self.v2t(tex.unsqueeze(0))
        tex_prep = (F.tanh(tex_a).permute(0, 2, 3, 1).contiguous().view(1, face.shape[1], 2, 2, 2, 3) * .5) + .5
        vert_prep = vert.view(3, self.model_dict["M"].res,
                              self.model_dict["M"].res).transpose(0, 2).contiguous().view(1, -1, 3)

        light_dir, light_color_directional, eye = self.get_eye_and_light()

        fake_data = self.render(vert_prep,
                             face,
                             tex_prep,
                             eye,
                             light_dir=light_dir,
                             light_color_directional=light_color_directional)

        fake = self.transform.norm(fake_data, tensor=True)

        # add some noise
        fake = (fake * .9) + (.1 * torch.FloatTensor(fake.shape).normal_(-1, 1).cuda())

        disc_result_fake = self.model_dict["D"](fake)
        self.loss_batch_dict['M_Loss'] = (self.l1_loss(disc_result_fake, torch.ones_like(disc_result_fake)) ** 2) * .5
        self.loss_batch_dict['M_Loss'].backward()
        self.opt_dict["M"].step()
        return fake

    def train(self):
        # Train
        params = self.params
        for epoch in range(params["train_epoch"]):

            # clear last epopchs losses
            for loss in self.losses:
                self.loss_epoch_dict[loss] = []

            print(f"Sched Iter:{self.current_iter}, Sched Epoch:{self.current_epoch}")
            [print(f"Learning Rate({opt}): {self.opt_dict[opt].param_groups[0]['lr']}") for opt in
             self.opt_dict.keys()]

            self.loop_iter = 0
            epoch_start_time = time.time()

            for (real_data) in tqdm(self.train_loader):
                real = real_data.cuda()
                # add some noise
                real = (real * .9) + (.1 * torch.FloatTensor(real.shape).normal_(-1, 1).cuda())

                # DREAM #
                self.set_grad_req(d=False, g=True)
                fake = self.dream_on_mesh()

                # TRAIN DISC #
                self.set_grad_req(d=True, g=False)
                self.train_disc(fake, real)

                # append all losses in loss dict #
                [self.loss_epoch_dict[loss].append(self.loss_batch_dict[loss].data[0]) for loss in self.losses]
                self.loop_iter += 1
                self.current_iter += 1

            self.current_epoch += 1

            if self.current_epoch % params['save_every'] == 0:
                helper.show_test(real,
                                 fake,
                                 self.model_dict['M'].textures.unsqueeze(0),
                                 self.transform,
                                 save=f'output/{params["save_root"]}_{self.current_epoch}.jpg')

                save_str = self.save_state(f'output/{params["save_root"]}_{self.current_epoch}.json')
                print(save_str)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            [self.train_hist_dict[loss].append(helper.mft(self.loss_epoch_dict[loss])) for loss in self.losses]
            print(f'Epoch:{self.current_epoch}, Epoch Time:{per_epoch_ptime}')
            [print(f'Train {loss}: {helper.mft(self.loss_epoch_dict[loss])}') for loss in self.losses]

        self.display_history()
        print('Hit End of Learning Schedule!')
