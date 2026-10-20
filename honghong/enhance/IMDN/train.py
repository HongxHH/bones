import argparse, os
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import architecture
from data import DIV2K, overfit
import utils
import random
from collections import OrderedDict

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Training settings
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="IMDN_low", help="选择具体的 IMDN 变体")
    parser.add_argument("--dataset", type=str, default="bbb", help="[DIV2K , bbb]")
    parser.add_argument("--batch_size", type=int, default=16, help="训练阶段的 batch size")
    parser.add_argument("--testBatchSize", type=int, default=1, help="测试阶段的 batch size")
    parser.add_argument("--epochs", type=int, dest="nEpochs", default=500, help="总训练 epoch 数")
    parser.add_argument("--lr", type=float, default=2e-4, help="初始学习率")
    parser.add_argument("--step_size", type=int, default=200, help="学习率衰减间隔（epoch）")
    parser.add_argument("--gamma", type=float, default=0.5, help="学习率衰减倍率")
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--resume", default="", type=str, help="断点恢复路径")
    parser.add_argument("--start-epoch", default=1, type=int,
                        help="手动指定起始 epoch（通常用于断点恢复）")
    parser.add_argument("--threads", type=int, default=8,
                        help="DataLoader 的 worker 数量")
    parser.add_argument("--n_train", type=int, default=0,
                        help="可选：仅当需要限制训练样本数量时设置，0 表示使用全部样本")
    parser.add_argument("--n_val", type=int, default=1,
                        help="验证集大小（某些数据集专用）")
    parser.add_argument("--test_every", type=int, default=0,
                        help="兼容占位参数，当前流程不再依赖该值")
    parser.add_argument("--scale", type=int, default=2,
                        help="超分辨缩放比例")
    parser.add_argument("--patch_size", type=int, default=192,
                        help="训练裁剪的 HR patch 大小")
    parser.add_argument("--rgb_range", type=int, default=1,
                        help="归一化后 RGB 最大值")
    parser.add_argument("--n_colors", type=int, default=3,
                        help="训练使用的颜色通道数")
    parser.add_argument("--pretrained", default="", type=str,
                        help="预训练模型路径")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--isY", action="store_true", default=True)
    parser.add_argument("--ext", type=str, default='.png', help="图像扩展名")
    parser.add_argument("--phase", type=str, default='train', help="训练 / 测试阶段标记")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                        help="每隔多少个 epoch 保存一次权重")
    return parser


def train(epoch, model, training_data_loader, optimizer, l1_criterion, args, device):
    model.train()
    utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
    print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])
    for iteration, (lr_tensor, hr_tensor) in enumerate(training_data_loader, 1):

        if args.cuda:
            lr_tensor = lr_tensor.to(device)  # ranges from [0, 1]
            hr_tensor = hr_tensor.to(device)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        loss_l1 = l1_criterion(sr_tensor, hr_tensor)
        loss_sr = loss_l1

        loss_sr.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss_l1: {:.5f}".format(epoch, iteration, len(training_data_loader),
                                                                  loss_l1.item()))


def save_checkpoint(model, epoch, args):
    model_folder = "{}_{}_x{}/".format(args.model, args.dataset, args.scale)
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset == "DIV2K":
        args.root = ""
    elif args.dataset == "bbb":
        args.root = r"E:\dev\Code\Python\bones\enhance\IMDN\dataset\bbb"
        args.dir_hr = os.path.join(args.root, '240p_x2_hr')
        args.dir_lr = os.path.join(args.root, '240p_lr')
        args.pretrained = r"E:\dev\Code\Python\bones\enhance\IMDN\checkpoint\IMDN_low_DIV2K_x2.pth"
        args.ext = '.png'
    else:
        raise NotImplementedError

    print(args)

    torch.backends.cudnn.benchmark = True
    # random seed
    seed = args.seed
    if seed is None:
        seed = random.randint(1, 10000)
    print("Ramdom Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    cuda = args.cuda
    device = torch.device('cuda' if cuda else 'cpu')

    print("===> Loading datasets")

    if args.dataset == 'DIV2K':
        trainset = DIV2K.div2k(args)
    elif args.dataset == 'bbb':
        trainset = overfit.OverfitDataset(args)
    else:
        raise NotImplementedError
    training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    print("===> Building models")
    args.is_train = True

    if args.model == "IMDN_ultra":
        model = architecture.IMDN(upscale=args.scale)
    elif args.model == "IMDN_high":
        model = architecture.IMDN(upscale=args.scale, nf=32)
    elif args.model == "IMDN_medium":
        model = architecture.IMDN_RTC(upscale=args.scale)
    elif args.model == "IMDN_low":
        model = architecture.IMDN_RTC(upscale=args.scale, num_modules=3, nf=6)
    else:
        raise NotImplementedError

    l1_criterion = nn.L1Loss()

    print("===> Setting GPU")
    if cuda:
        model = model.to(device)
        l1_criterion = l1_criterion.to(device)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("===> loading models '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            new_state_dcit = OrderedDict()
            for k, v in checkpoint.items():
                if 'module' in k:
                    name = k[7:]
                else:
                    name = k
                new_state_dcit[name] = v
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dcit.items() if k in model_dict}

            for k, v in model_dict.items():
                if k not in pretrained_dict:
                    print(k)
            model.load_state_dict(pretrained_dict, strict=True)

        else:
            print("===> no models found at '{}'".format(args.pretrained))

    print("===> Setting Optimizer")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("===> Training")
    print_network(model)
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        train(epoch, model, training_data_loader, optimizer, l1_criterion, args, device)
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_checkpoint(model, epoch, args)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

