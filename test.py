import os

# Set environment variable for CUDA launch blocking
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from shutil import rmtree

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch import nn

torch.backends.cudnn.benchmark = True

from model.randlanet import RandLANet
from utils.Open3D import Open3Dataset
from utils.metrics import Metrics
from utils.argparser import init_params
from utils.utils import adjust_learning_rate
from utils.logger import init_logger


# Define step classes for continual learning
step_classes = {0: [9, 10, 11, 12, 15, 17],
                1: [13, 14, 16, 18, 19],
                2: [1, 2, 3, 4, 5, 6, 7, 8]}

##### Validation Routine
def validate(model_id, writer, logger, vset, vloader, epoch, model, device, args):
    # Get current step classes
    current_step_classes = np.hstack([s for t, s in step_classes.items() if t <= args.CLstep])
    current_step_classes = {c: i + 1 for i, c in enumerate(current_step_classes)}

    # Get all step classes and remap them
    all_step_classes = np.hstack(list(step_classes.values()))
    all_step_classes = np.insert(all_step_classes, 0, 0)
    remap_classes = {c: i for i, c in enumerate(all_step_classes)}

    # Sort class names according to remap_classes
    class_names = [vset.pointcloud_dataset.cnames[i] for i in all_step_classes]
    class_names = sorted(class_names, key=lambda x: remap_classes[vset.pointcloud_dataset.cnames.index(x)])

    # Initialize metrics
    metric = Metrics(class_names, device=device, mask_unlabeled=True,
                     mask=list(np.arange(1, len(current_step_classes) + 1)), step=args.CLstep)
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(
                tqdm(vloader, "Validating Epoch %d" % (epoch + 1), total=int(len(vset) / vloader.batch_size))):
            o, _, feat, _, = model(data)
            y = [remap_classes[d] for d in data["labels"].numpy().flatten()]
            y = torch.tensor(np.array(y)).unsqueeze(0).cuda()

            metric.add_sample(o.argmax(dim=2).flatten(), y.flatten())

    # Calculate metrics
    miou = metric.percent_mIoU()
    acc = metric.percent_acc()
    prec = metric.percent_prec()
    writer.add_scalar('mIoU', miou, epoch)
    writer.add_scalar('PP', prec, epoch)
    writer.add_scalar('PA', acc, epoch)
    writer.add_scalars('IoU', {n: 100 * v for n, v in zip(metric.name_classes, metric.IoU()) if not torch.isnan(v)},
                       epoch)
    print(metric)
    logger.cprint(str(metric))
    model.train()
    return miou, o.swapaxes(1, 2), y


def main(args):
    # Set device to CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = args.dataset
    args.c2f = False
    args.CL = True

    vset = Open3Dataset(pointcloud_dataset=dataset(augment=False, split='val',
                                                   CL=False,
                                                   c2f=args.c2f,
                                                   step=args.CLstep,
                                                   setup="Sequential"))
    vloader = DataLoader(vset,
                         batch_size=args.val_batch_size,
                         shuffle=False,
                         num_workers=args.num_workers)

    current_step_classes = np.hstack([s for t, s in step_classes.items() if t <= args.CLstep])
    current_step_classes = np.insert(current_step_classes, 0, 0)

    # Initialize model
    model = RandLANet(step=args.CLstep, num_neighbors=16, device='cuda', num_classes=len(current_step_classes))

    # Set up logging directory
    logdir = os.path.join(args.logdir, "test_" + args.test_name)
    rmtree(logdir, ignore_errors=True)
    writer = SummaryWriter(logdir, flush_secs=.5)

    # init logger
    logger = init_logger(logdir, args)

    # Load pretrained model if specified
    model_save = torch.load(args.ckpt_file)
    model.load_state_dict(model_save)
    model.to('cuda')
    with torch.no_grad():
        miou, o, y = validate('model_1', writer, logger, vset, vloader, 0, model, device, args)

    writer.close()


if __name__ == '__main__':
    args = init_params('train_envelope', verbose=True)
    main(args)
