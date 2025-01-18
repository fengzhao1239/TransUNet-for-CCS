import os
import datetime
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Model.Zeeshan_unet import UNET
from Model.derivatives import *
from TensorProcessing.Zeeshan_2d import ZeeshanDatasetOneStep
import torch
import torch.nn as nn
import time
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

ts = 1  # one-step prediction

# finite difference filters, for calculating 1st & 2nd derivatives
# 1 st derivative of x and y axes
dxx = [[[[0, 0, 0],
         [-1 / 2, 0, 1 / 2],
         [0, 0, 0]]]]

dyy = [[[[0, -1 / 2, 0],
         [0, 0, 0],
         [0, 1 / 2, 0]]]]

# 2 nd derivative of x and y axes
dxx_xx = [[[[0, 0, 0],
            [1, -2, 1],
            [0, 0, 0]]]]

dyy_yy = [[[[0, 1, 0],
            [0, -2, 0],
            [0, 1, 0]]]]


def set_default():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.manual_seed(3407)
    torch.set_default_dtype(torch.float32)


def cal_loss(pred, y, loss_fn):
    """
    [b, c=1, 50, 50]
    """
    original_loss = loss_fn(pred, y)  # [b, 1, 50, 50]
    pred = pred.clone()
    pred[y == 0.001] = pred[y == 0.001] * 0
    penalty_loss = loss_fn(pred, y)  # [b, 1, 50, 50]
    sum_loss = torch.mean(original_loss + penalty_loss * 0.0, dim=1)
    batch_avg = torch.mean(sum_loss, dim=0)
    return torch.mean(batch_avg)


def grad_loss(pred, label, loss_fn):
    """
    [b, t=1, 50, 50]
    """
    model_dx = Dx(dxx, ts, ts).float().to(device)
    model_dy = Dy(dyy, ts, ts).float().to(device)
    model_dxx = Dxx(dxx_xx, ts, ts).float().to(device)
    model_dyy = Dyy(dyy_yy, ts, ts).float().to(device)

    pred_dx, y_dx = model_dx(pred), model_dx(label)
    pred_dy, y_dy = model_dy(pred), model_dy(label)
    pred_dxx, y_dxx = model_dxx(pred), model_dxx(label)
    pred_dyy, y_dyy = model_dyy(pred), model_dyy(label)

    loss_tensor = loss_fn(pred_dx, y_dx) + loss_fn(pred_dy, y_dy) + (loss_fn(pred_dxx, y_dxx) + loss_fn(pred_dyy, y_dyy)) * 0
    sum_time = torch.mean(loss_tensor, dim=1)
    mean_batch = torch.mean(sum_time, dim=0)
    return torch.mean(mean_batch)


def train_loop(dataset_loader, gpu, net, optimizer, loss_criterion):
    net.train()
    size = len(dataset_loader.dataset)
    num_batch = len(dataset_loader)
    total_loss_l2 = 0

    print(f'num of batch = {num_batch}')

    for batch_idx, (X, Y) in enumerate(dataset_loader):
        X = X.to(gpu)
        Y = Y.to(gpu)

        pred = net(X)
        loss = cal_loss(pred, Y, loss_criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_l2 += loss.item()

        if batch_idx % 100 == 0:
            loss_value_l2, current_batch = loss.item(), batch_idx * len(Y)
            print(f'Loss L2: {loss_value_l2:>15f} | [{current_batch:>5d}/{size:>5d}]')

    return total_loss_l2 / num_batch


def val_loop(dataloader, gpu, net, loss_criterion):
    net.eval()
    num_batches = len(dataloader)
    val_loss_l2 = 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X = X.to(gpu)
            Y = Y.to(gpu)

            pred = net(X)
            loss = cal_loss(pred, Y, loss_criterion)

            val_loss_l2 += loss.item()

    val_loss_l2 /= num_batches
    print(f'&& Validation Error: avg loss L2 = {val_loss_l2:.5e}')

    return val_loss_l2


if __name__ == '__main__':

    now = datetime.datetime.now()
    set_default()
    print(f'Beginning time is: {now}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, GPU name: {torch.cuda.get_device_name()}')

    state_variable = 'saturation'
    whole_dataset = ZeeshanDatasetOneStep(label=state_variable, test_phase=False)
    dataset_size = len(whole_dataset)
    train_size = int(0.89 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(whole_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(666))
    print(f'len of trainset: {len(train_dataset)}, len of valset: {len(val_dataset)}')

    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                            pin_memory=True)

    print(f'Data loading finished...')

    model = UNET(in_channels=9, out_channels=1).to(device)
    print(f'UNET Model trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    learning_rate = 1e-3
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,)
    criterion = nn.MSELoss(reduction='none')
    scheduler = ReduceLROnPlateau(adam_optimizer, mode='min', factor=0.2, patience=20, threshold=0.0001,
                                  threshold_mode='rel', cooldown=0, min_lr=1.e-8, eps=1e-08)

    epochs = 200

    train_loss_list = []
    val_loss_list = []

    begins = time.time()
    for tt in range(epochs):
        print(f'Epoch {tt + 1}\n===========================================')
        begin1 = time.time()
        train_l2_loss_epoch = train_loop(train_loader, device, model, adam_optimizer, criterion)
        val_l2_loss_epoch = val_loop(val_loader, device, model, criterion)

        scheduler.step(val_l2_loss_epoch)
        train_loss_list.append(train_l2_loss_epoch)
        val_loss_list.append(val_l2_loss_epoch)

        end1 = time.time()
        print(f'current learning rate: {adam_optimizer.param_groups[0]["lr"]}')
        print(f'******* This epoch takes {(end1 - begin1) / 60:.2f} min. *******')
        print(f'******* All epochs takes {(end1 - begins) / 60:.2f} min. *******\n')

    torch.save(model.state_dict(),
               f'G:\\optim_code\\checkpoints\\zeeshan_UNET_weights_{state_variable}_{now.strftime("%m.%d")}.pth')
    np.savetxt(f'G:\\optim_code\\losses\\zeeshan_UNET_l2_training_loss_{state_variable}_{now.strftime("%m.%d")}.txt',
               train_loss_list)
    np.savetxt(f'G:\\optim_code\\losses\\zeeshan_UNET_l2_validation_loss_{state_variable}_{now.strftime("%m.%d")}.txt',
               val_loss_list)

    plt.semilogy(np.arange(len(train_loss_list)), train_loss_list, c='r', label='training loss')
    plt.semilogy(np.arange(len(val_loss_list)), val_loss_list, c='b', label='validating loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'G:\\optim_code\\plots\\zeeshan UNET Loss Values_{state_variable}_{now.strftime("%m.%d")}.png', dpi=300)
    plt.close()

    print('=========Done===========')
