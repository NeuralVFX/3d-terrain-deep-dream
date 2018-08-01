import random
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import neural_renderer as nr
import torch.nn.functional as F


plt.switch_backend('agg')

############################################################################
# Helper Utilities
############################################################################


def weights_init_normal(m):
    # Set initial state of weights
    classname = m.__class__.__name__
    if 'ConvTrans' == classname:
        pass
    elif 'Conv2d' in classname or 'Linear' in classname or 'ConvTrans' in classname:
        nn.init.normal_(m.weight.data, 0, .02)


def mft(tensor):
    # Return mean float tensor #
    return torch.mean(torch.FloatTensor(tensor))


def random_eye_and_light():
    light_dir = [(random.random() - .5) * 2,
                 random.random(),
                 (random.random() - .5) * 2]

    light_color_directional = [(random.random() * .5) + 1,
                               (random.random() * .5) + 1,
                               (random.random() * .5) + .75]

    eye = nr.get_points_from_angles((random.random() * .8) + 1.7,
                                    (random.random() * 15) + 20,
                                    random.random() * 360)

    return light_dir, light_color_directional, eye


############################################################################
# Display Images
############################################################################

import pdb
def show_test(real, fake, art_mesh, art_test, transform, save=False):
    # Show and save
    #pdb.set_trace()
    batch_size = fake.shape[0]
    fig, ax = plt.subplots(batch_size, 4, figsize=(11, 4*batch_size))

    for i in range(batch_size):
        r = transform.denorm(real.detach()[i], cpu=True, variable=False)
        f = transform.denorm(fake.detach()[i], cpu=True, variable=False)
        m = transform.denorm(F.tanh(art_test.detach()[0]), cpu=True, variable=False)
        m_test = transform.denorm(F.tanh(art_mesh.detach()[0]), cpu=True, variable=False)

        ax[i,0].cla()
        ax[i,0].imshow(r)
        ax[i,1].cla()
        ax[i,1].imshow(f)
        ax[i,2].cla()
        ax[i,2].imshow(m)
        ax[i,3].cla()
        ax[i,3].imshow(m_test)
    if save:
        plt.savefig(save)
    plt.show()
    plt.close(fig)
