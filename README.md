# 3d-terrain-deep-dream
Pytorch implementation of deep-dreaming on a 3d mesh based on [Neural 3D Mesh Renderer](https://arxiv.org/pdf/1711.07566.pdf)

The basic idea with this is that if you have a DEM file with no assosciated textures, that you can use photographs of similar terrain to automatically create fitting textures. This uses a GAN style approach of asking a discriminator to look at real images and rendered images, training the discriminator to get better while the deep dream makes the textures better.

I just started this, so `it's not ready to clone yet....`

I've tested this using the [GeoPose3k Dataset](http://cphoto.fit.vutbr.cz/geoPose3K/).

# Code usage
Usage instructions found here: [user manual page](USAGE.md).

## Example Output
### GeoPose3k Dataset
#### 1) Refernce Image  2) Neural Render  3) Texture in UV Space
![](output/austria_01.jpg)
