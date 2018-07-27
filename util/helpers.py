import torch.nn as nn
import torch
import matplotlib.pyplot as plt


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


############################################################################
# Display Images
############################################################################


def show_test(real, fake, art_mesh, transform, save=False):
    # Show and save
    fig, ax = plt.subplots(1, 3, figsize=(11, 7))

    r = transform.denorm(real.detach()[0], cpu=True, variable=False)
    f = transform.denorm(fake.detach()[0], cpu=True, variable=False)
    m = transform.denorm(art_mesh.detach()[0], cpu=True, variable=False)

    ax[0].cla()
    ax[0].imshow(r)
    ax[1].cla()
    ax[1].imshow(f)
    ax[2].cla()
    ax[2].imshow(m)
    if save:
        plt.savefig(save)
    plt.show()
    plt.close(fig)
