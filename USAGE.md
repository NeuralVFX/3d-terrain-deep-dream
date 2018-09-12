
## Getting Started
- Check your python version, this is built on `python 3.6`
- Install `pytorch 0.4.0` and dependencies from https://pytorch.org/
- Install packages 'tqdm', 'matplotlib', 'unzip', 'tar', 'scipy', 'osgeo', 'cv2'

- Clone and install [Neural Renderer](https://github.com/daniilidis-group/neural_renderer)

- Clone this repo:
```bash
git clone https://github.com/NeuralVFX/3d-terrain-deep-dream.git
cd 3d-terrain-deep-dream
```
- Download the image dataset [GeoPose3K](http://cphoto.fit.vutbr.cz/geoPose3K/):
```bash
bash data/get_test_dataset.sh
```
- Download a DEM file (e.g.[USGS Central-Colorado](https://www.sciencebase.gov/catalog/item/5acecd0ee4b0e2c2dd1a6acf))
```bash
bash dem/get_test_dem.sh
```
## Train The Model
```bash
python train.py --render_res 256 --train_epoch 50 --save_root austria
```

## Continue Training Existing Saved State
```bash
python train.py --load_state output/austria_3.json --render_res 256 --train_epoch 50 --save_root austria
```

## Command Line Arguments

```
--dem_file, default='x34y441_CO.img', type=str          # Train folder name
--grid_res, default=256, type=int                       # Resolution of mesh grid
--disc_layers, default=7, type=int                      # Count of conv layers in discriminator
--disc_filters, default=512, type=int                   # Filter count for discriminators
--lr_disc, default=.05, type=float                      # Learning rate for discriminator
--lr_tex, default=.01, type=float                       # Learning rate for texture on mesh
--lr_mesh, default=.0001, type=float                    # Learning rate for verts on mesh
--render_res, default=256, type=int                     # Image size for both Neural Render and Dataloader
--use_generic_dataset, default=False, type=bool         # If generic, "dataset" is simply used as a searchpath for images, otherwise geoPose3k is assumed
--dataset, default='geoPose3K_final_publish', type=str  # Dataset folder
--opt_mesh default=False, type=bool                     # Allow grad on mesh
--opt_tex, default=True, type=bool                      # Allow grad on texture
--batch_size, default=8, type=int                       # Batchsize for dataloader
--camera_pausing default=False, type=bool               # Stall camera for "cam_pause_len" iterations during dreaming
--cam_pause_len, default=30, type=int                   # Number of iterations to pause camera before each reset
--train_epoch', default=6, type=int                     # Number loops trough training set
--save_every', default=1, type=int                      # How many epochs to train between every save
--save_img_every', default=1, type=int                  # How many epochs to train between every image save
--load_workers', default=1, type=int                    # How many workers the data loader should use
--data_perc', default=1, type=float                     # Percentage of dataset to use, selected with fixed random seed
--save_root', default='lsun_test', type=str             # Prefix for files created by the model under the /output directory
--load_state', type=str                                 # Optional: filename of state to load and resume training from
```

## Data Folder Structure

- This is the folder layout that the data is expected to be in:

`data/<data set>/`

- The DEM file should be placed in the dem directory:

`dem/<dem file>`

## Output Folder Structure

- `weights`, `test images`, and `loss graph`, are all output to this directory: `output/<save_root>_*.*`

- Loss Graph Example: `output/austria_loss.jpg`
![](output/austria_loss.jpg)

- Test Image Example (output every loop through dataset): `output/austria_05.jpg`
![](output/austria_05.jpg)

## Other Notes

- Currently the data loader works very specifically with the `GeoPose3k` dataset

- Only square grids should be used for this, the math of the texture preparation absolutely requires this



