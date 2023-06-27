import argparse

parser = argparse.ArgumentParser(description='Cross-Spectral Image Reconstruction')

# data specifications 
parser.add_argument('--flist_train', type=str, default='datasets/train.flist', help='image dataset directory')

# model specifications
parser.add_argument('--luma-bins', type=int, default=32, help='Number of luma bins in 3D linear regression cubes')
parser.add_argument('--num-down-layers', type=int, default=4, help='Number of downsampling layers')
parser.add_argument('--feature-layers', type=int, default=7, help='Number of same conv layers')
parser.add_argument('--batch-norm', action='store_false', help='If set use batch norm')
parser.add_argument('--net-input-size', type=int, default=512, help='Size of input')

# hardware specifications 
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers used in data loader')

# adam optimization specifications
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for generator')
parser.add_argument('--lrepochs', default="20,32,40,48,56:2",type=str, help='the epochs to decay lr: the downscale rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 in optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 in optimier')

# training specifications
parser.add_argument('--epochs', type=int, default=64, help='the number of epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='batch size in each mini-batch')

# log specifications
parser.add_argument('--save_every', type=int, default=1e4, help='frequency for saving models')
parser.add_argument('--save_dir', type=str, default='experiments/', help='directory for saving models and logs')

# ----------------------------------
args = parser.parse_args()
