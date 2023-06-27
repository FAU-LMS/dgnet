import os
import torch
from datetime import datetime

from utils.option import args
from trainer.trainer import Trainer

def main_worker(id, args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    args.local_rank = args.global_rank = id
    args.save_dir = os.path.join(
        args.save_dir, f'deep-guided_{dt_string}')
        
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'config.txt'), 'a') as f:
        for key, val in vars(args).items():
            f.write(f'{key}: {val}\n')
    print(f'[**] create folder {args.save_dir}')

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # setup distributed parallel training environments
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        print("No GPU found!")
        quit()

    main_worker(0, args)
