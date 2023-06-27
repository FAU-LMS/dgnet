import torch
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt

from model.model import DGNet
from utils.option import args

def load_model(path='checkpoint/pretrained.pt', gpu=True):
    model = DGNet(args)
    print(f'[**] Loading generator network from {path}')
    if torch.cuda.is_available() and gpu:
        torch.backends.cudnn.benchmark = True
        model.load_state_dict(torch.load(path, map_location='cuda'))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    return model


def predict(model, ref_image, masked_image, mask):
    norm_ref_image = ref_image
    norm_masked_image = masked_image
    norm_masked_image = (norm_masked_image * (1 - mask)) + mask

    dividable = 2 ** args.num_down_layers
    pad_y, pad_x = 0, 0
    if norm_ref_image.shape[2] % dividable > 0:
        pad_y = dividable - norm_ref_image.shape[2] % dividable
    if norm_ref_image.shape[3] % dividable > 0:
        pad_x = dividable - norm_ref_image.shape[3] % dividable

    norm_ref_image = F.pad(norm_ref_image, (0, pad_x, 0, pad_y), mode='replicate')
    norm_masked_image = F.pad(norm_masked_image, (0, pad_x, 0, pad_y), mode='replicate')
    norm_mask = F.pad(mask, (0, pad_x, 0, pad_y), mode='replicate')

    model.eval()
    with torch.no_grad():
        out = model(norm_ref_image, norm_masked_image, norm_mask)
    out = (out * norm_mask) + (norm_masked_image * (1 - norm_mask))

    if pad_y > 0:
        out = out[:, :, :-pad_y, :]
    if pad_x > 0:
        out = out[:, :, :, :-pad_x]
    out[out > 1] = 1
    out[out < 0] = 0
    return out


def reconstruct(model, ms_image_masked, ms_mask):
    ms_image_masked = torch.transpose(ms_image_masked, 0, 1)
    ms_mask = torch.transpose(ms_mask, 0, 1)
    channels = ms_image_masked.shape[0]
    ref_image = ms_image_masked[channels // 2]
    ms_image = predict(model, torch.tile(ref_image[None, :], dims=(channels, 1, 1, 1)), ms_image_masked, 1 - ms_mask)
    return torch.transpose(ms_image, 0, 1)


def run(ms_channels=9, gpu=True):
    model = load_model(gpu=gpu)
    ms_image_masked = None
    ms_mask = None
    for i in range(ms_channels):
        img = imageio.v3.imread(f'test_data/masked_image_{i}.png')/255
        mask = imageio.v3.imread(f'test_data/mask_{i}.png')/255
        if ms_mask is None:
            ms_mask = torch.zeros(1, ms_channels, mask.shape[0], mask.shape[1])
            ms_image_masked = torch.zeros(1, ms_channels, img.shape[0], img.shape[1])
        ms_mask[:, i] = torch.from_numpy(mask)
        ms_image_masked[:, i] = torch.from_numpy(img)

    if gpu:
        ms_image_masked = ms_image_masked.cuda()
        ms_mask = ms_mask.cuda()

    ms_image = reconstruct(model, ms_image_masked, ms_mask)

    real_rgb = torch.concat([ms_image[:, 3], ms_image[:, 4], ms_image[:, 5]], dim=0).permute(1, 2, 0).cpu().detach()
    fake_rgb = torch.concat([ms_image[:, 0], ms_image[:, 1], ms_image[:, 2]], dim=0).permute(1, 2, 0).cpu().detach()

    plt.subplot(121)
    plt.imshow(real_rgb)
    plt.subplot(122)
    plt.imshow(fake_rgb)
    plt.show()


if __name__ == '__main__':
    run()
