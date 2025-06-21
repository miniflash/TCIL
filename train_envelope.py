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

# miniflash
from model.kd_module import Incremental_base_loss

# Define step classes for continual learning
step_classes = {0: [9, 10, 11, 12, 15, 17],
                1: [13, 14, 16, 18, 19],
                2: [1, 2, 3, 4, 5, 6, 7, 8]}

##### Validation Routine
def validate(writer, logger, vset, vloader, epoch, model, device, args):
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
            o, _, _, _ = model(data)
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

    # Print configuration
    print(f"Training SemanticKITTI using RandLA-Net step {args.CLstep}")
    print("CIL configuration")
    if args.CLstrategy == "tcil":
        print("using tcil constraint")
    else:
        print("using knowledge distillation")

    # Get datasets with continual learning configurations
    dset = Open3Dataset(pointcloud_dataset=dataset(split='train',
                                                   CL=args.CL,
                                                   c2f=args.c2f,
                                                   step=args.CLstep,
                                                   setup="Sequential"))
    dloader = DataLoader(dset,
                         batch_size=args.batch_size,
                         shuffle=True,
                         num_workers=args.num_workers,
                         drop_last=True)

    vset = Open3Dataset(pointcloud_dataset=dataset(augment=False, split='val',
                                                   CL=False,
                                                   c2f=args.c2f,
                                                   step=args.CLstep,
                                                   setup="Sequential"))
    vloader = DataLoader(vset,
                         batch_size=args.val_batch_size,
                         shuffle=False,
                         num_workers=args.num_workers)

    # Get current step classes and remap them
    previous_step_classes = np.hstack([s for t, s in step_classes.items() if t < args.CLstep])
    previous_step_classes = np.insert(previous_step_classes, 0, 0)

    current_step_classes = np.hstack([s for t, s in step_classes.items() if t <= args.CLstep])
    current_step_classes = np.insert(current_step_classes, 0, 0)
    remap_classes = {c: i for i, c in enumerate(current_step_classes)}

    # Sort class names according to remap_classes
    class_names = [vset.pointcloud_dataset.cnames[i] for i in current_step_classes]
    class_names = sorted(class_names, key=lambda x: remap_classes[vset.pointcloud_dataset.cnames.index(x)])

    # Initialize model
    model = RandLANet(step=args.CLstep, num_neighbors=16, device='cuda', num_classes=len(current_step_classes))

    # Set up logging directory
    logdir = os.path.join(args.logdir, "train_" + args.test_name)
    rmtree(logdir, ignore_errors=True)
    writer = SummaryWriter(logdir, flush_secs=.5)

    # init logger
    logger = init_logger(logdir, args)

    # Load pretrained model if specified
    if args.pretrained_model:
        new = model.state_dict()
        old = torch.load(args.ckpt_file)

        # Load weights but exclude the certain layers
        if args.CL:
            for k in new:
                if args.CLstep == 1 and 'Multimodal_interaction' in k:
                    continue
                if "binary_head.3" in k:
                    continue  # Don't load auxiliary seg loss
                if "fc1.3" not in k or new[k].shape == old[k].shape:
                    new[k] = old[k]

        model.load_state_dict(new)

        # load previous model for knowledge distillation or inpainting
        if args.CLstep != 0:
            old_model = RandLANet(step=args.CLstep - 1, num_neighbors=16, device='cuda',
                                  num_classes=len(previous_step_classes))
            old_model.load_state_dict(old, strict=False)
            old_model.to('cuda')
            old_model.eval()
    model.to('cuda')

    # freeze old model
    for param in old_model.parameters():
        param.requires_grad = False

    # Training parameters
    steps_per_epoch = len(dset) // args.batch_size
    lr0 = args.lr  # Initial learning rate
    lr_decays = {i: args.poly_power for i in range(0, args.decay_over_iterations)}
    optim = Adam(model.parameters(), weight_decay=args.weight_decay)

    best_miou = 0

    for e in range(args.epochs):
        torch.cuda.empty_cache()

        # Validation routine
        if e % args.eval_every_n_epochs == 0 and e > 0:
            miou, o, y = validate(writer, logger, vset, vloader, e, model, device, args)
            if miou > best_miou:
                best_miou = miou
                logger.cprint('******************* Best Model Saved *******************')
                torch.save(model.state_dict(), logdir + "/val_best.pth")

        metrics = Metrics(class_names, device=device, mask_unlabeled=True, step=args.CLstep)

        pbar = tqdm(dloader, total=steps_per_epoch,
                    desc="Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress" % (e + 1, args.epochs, 0., 0.))

        # logger record
        logger.cprint("=====Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress=====" % (e + 1, args.epochs, 0., 0.))

        # Training routine
        for i, data in enumerate(pbar):
            step = i + steps_per_epoch * e

            loss = nn.CrossEntropyLoss(ignore_index=0)

            # loss by prototypes
            loss_incremental_proto = Incremental_base_loss(step=args.CLstep, mode='MSELoss')

            # loss by binary
            loss_seg_aux = nn.CrossEntropyLoss(ignore_index=0)

            lr = lr0  # Initial learning rate
            optim.param_groups[0]['lr'] = lr
            lr = adjust_learning_rate(optim, lr_decays, e)

            optim.zero_grad()

            # get old labels and features
            if args.CLstep != 0:
                o_s, f_s, fd_s, o_binary_s = old_model(data)

            o, f, fd, o_binary = model(data)
            y = [remap_classes[d] for d in data["labels"].numpy().flatten()]

            y = torch.tensor(np.array(y)).unsqueeze(0).cuda()

            l = loss(o.swapaxes(1, 2).contiguous(), y.view(args.batch_size, -1).contiguous())
            if args.CLstep != 0:
                loss_pd = loss_incremental_proto(fd, fd_s, y)  # proto
                l += args.lambda_1 * loss_pd

                y_binary = y.clone()
                mask_class = [remap_classes[cls] for cls in step_classes[args.CLstep]]
                mask = torch.isin(y_binary, torch.tensor(mask_class).to(y_binary.device))
                y_binary[mask] = 2
                y_binary[~mask & (y_binary != 0)] = 1
                loss_oc = loss_seg_aux(o_binary.swapaxes(1, 2).contiguous(),
                                       y_binary.view(args.batch_size, -1).contiguous())
                l += args.lambda_2 * loss_oc

            l.backward()

            pred = o.argmax(dim=2).flatten()

            # print(np.unique(pred.cpu().numpy()))
            metrics.add_sample(pred.flatten(), y.flatten())

            optim.step()
            miou = metrics.percent_mIoU()
            pbar.set_description(
                "Epoch %d/%d, Loss: %.2f, Loss_oc: %.2f, Loss_pd: %.2f, mIoU: %.2f, Progress" % (
                    e + 1, args.epochs, l.item(), loss_oc.item(), loss_pd.item(), miou))
            # logger record
            logger.cprint(
                "=====Epoch %d/%d, Loss: %.2f, Loss_oc: %.2f, Loss_pd: %.2f, mIoU: %.2f, Progress=====" % (
                    e + 1, args.epochs, l.item(), loss_oc.item(), loss_pd.item(), miou))

            writer.add_scalar('lr', lr, step)
            writer.add_scalar('loss', l.item(), step)
            writer.add_scalar('step_mIoU', miou, step)

        torch.save(model.state_dict(), logdir + "/latest.pth")

    writer.close()

    # Final validation
    miou, o, y = validate(writer, logger, vset, vloader, e, model, device, args)
    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), logdir + "/val_best.pth")
        logger.cprint('******************* Best Model Saved *******************')


if __name__ == '__main__':
    args = init_params('train_envelope', verbose=True)
    main(args)
