import os
from tqdm import tqdm
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import Dataset
from utils import PSNR
from model.model import DGNet


class Trainer():
    def __init__(self, args):
        self.args = args 
        self.iteration = 0

        self.train_dataset = Dataset(args.flist_train, args.net_input_size)

        self.psnr = PSNR.PSNR(1.0).to(0)
        self.curr_psnr = -1

        self.net = DGNet(args).cuda()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    def save(self, ):
        if self.args.global_rank == 0:
            print(f'\nsaving {self.iteration} model to {self.args.save_dir} ...')
            torch.save(self.net.state_dict(),
                os.path.join(self.args.save_dir, f'M{str(self.iteration).zfill(7)}.pt'))
            torch.save(self.optim.state_dict(),
                os.path.join(self.args.save_dir, f'O{str(self.iteration).zfill(7)}.pt'))

    def train(self):
        data_loader = DataLoader(
            self.train_dataset, batch_size=self.args.batch_size,
            shuffle=True, num_workers=self.args.num_workers, pin_memory=True)

        pbar = tqdm(range(len(self.train_dataset)), initial=0, dynamic_ncols=True, smoothing=0.9)
        for epoch in range(1, self.args.epochs + 1):
            print('\n\nTraining epoch: %d' % epoch)
            self.adjust_learning_rate(self.optim, epoch, self.args.lr, self.args.lrepochs)

            pbar.reset()
            for items in data_loader:
                self.iteration += 1
                masked_images, ref_images, masks = self.cuda(*items)
                masked_inputs = (masked_images * (1 - masks).float()) + masks

                pred_img = self.net(ref_images, masked_inputs, masks)

                # reconstruction loss
                total_loss = self.calc_masked_loss(pred_img, masked_images, masks)

                # backward
                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

                psnr_loss = self.psnr(self.postprocess(masked_images), self.postprocess(pred_img))

                self.curr_psnr = self.ema(psnr_loss.item(), self.curr_psnr)

                # logs
                description = f'Loss:{total_loss.item():.3f}\t'
                description += f'PSNR:{self.curr_psnr:.3f}'
                pbar.set_description(description)
                pbar.update(len(masked_images))

                if (self.iteration % self.args.save_every) == 0:
                    self.save()

    def cuda(self, *args):
        return (item.to(0) for item in args)

    def postprocess(self, img):
        img[img < 0] = 0
        img[img > 1] = 1
        return img

    def calc_masked_loss(self, input, target, mask, alpha=10):
        masked_loss = F.smooth_l1_loss(input[mask == 1], target[mask == 1])
        nonmasked_loss = F.smooth_l1_loss(input[mask == 0], target[mask == 0])
        return alpha * masked_loss + nonmasked_loss

    def ema(self, new_value, old_value, l=0.99):
        if old_value == -1:
            old_value = new_value
        return (1 - l) * new_value + l * old_value

    def adjust_learning_rate(self, optimizer, epoch, base_lr, lrepochs):
        splits = lrepochs.split(':')
        assert len(splits) == 2

        downscale_epochs = [int(eid_str) for eid_str in splits[0].split(',')]
        downscale_rate = float(splits[1])
        print("downscale epochs: {}, downscale rate: {}".format(downscale_epochs, downscale_rate))

        lr = base_lr
        for eid in downscale_epochs:
            if epoch >= eid:
                lr /= downscale_rate
            else:
                break
        print("setting learning rate to {}".format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
