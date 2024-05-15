# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from augmentation import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

args = parser.parse_args()

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_ds = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)
test_ds  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='vgg':
    net = VGG('VGG19')
    
elif args.net=='res18':
    net = ResNet18()
    
elif args.net=='res34':
    net = ResNet34()
    
elif args.net=='res50':
    net = ResNet50()
    
elif args.net=='res101':
    net = ResNet101()
    
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
    
elif args.net=="Conv_tr":
    from models.Conv_tr import CvT
    
    embed_size  = 64   # 192 : CvT-W24 
    num_class   = 10
    
    net = CvT(embed_size, num_class)
    
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = 10
    )
    
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
    )
    
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
    )
    
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
    )
    
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
    )
    
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
    
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,                # depth of transformer for patch to patch attention only
    cls_depth=2,              # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
    )
    
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
    )
    
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=10,
                downscaling_factors=(2,2,2,1))

elif args.net=="t2t":
    from models.t2t import T2TViT
    net = T2TViT(
        dim = 512,
        image_size = 224,
        depth = 5,
        heads = 8,
        mlp_dim = 512,
        num_classes = 10,
        t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
    )

elif args.net=="cct":
    from models.cct import CCT
    net = CCT(
        img_size = (224, 224),
        embedding_dim = 384,
        n_conv_layers = 2,
        kernel_size = 7,
        stride = 2,
        padding = 3,
        pooling_kernel_size = 3,
        pooling_stride = 2,
        pooling_padding = 1,
        num_layers = 14,
        num_heads = 6,
        mlp_ratio = 3.,
        num_classes = 10,
        positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
    )
    
elif args.net=="cct_2":
    from models.cct import cct_2
    net = cct_2(
        img_size = 224,
        n_conv_layers = 1,
        kernel_size = 7,
        stride = 2,
        padding = 3,
        pooling_kernel_size = 3,
        pooling_stride = 2,
        pooling_padding = 1,
        num_classes = 10,
        positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
    )

elif args.net=="cross_vit":
    from models.cross_vit import CrossViT
    net = CrossViT(
        image_size = 256,
        num_classes = 10,
        depth = 4,               # number of multi-scale encoding blocks
        sm_dim = 192,            # high res dimension
        sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
        sm_enc_depth = 2,        # high res depth
        sm_enc_heads = 8,        # high res heads
        sm_enc_mlp_dim = 2048,   # high res feedforward dimension
        lg_dim = 384,            # low res dimension
        lg_patch_size = 64,      # low res patch size
        lg_enc_depth = 3,        # low res depth
        lg_enc_heads = 8,        # low res heads
        lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
        cross_attn_depth = 2,    # cross attention rounds
        cross_attn_heads = 8,    # cross attention heads
        dropout = 0.1,
        emb_dropout = 0.1
    )

elif args.net=="crossformer":
    from models.crossformer import CrossFormer
    net = CrossFormer(
        num_classes = 10,                  # number of output classes
        dim = (64, 128, 256, 512),         # dimension at each stage
        depth = (2, 2, 8, 2),              # depth of transformer at each stage
        global_window_size = (8, 4, 2, 1), # global window sizes at each stage
        local_window_size = 7,             # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
    )

elif args.net=="cvt":
    from models.cvt import CvT
    net = CvT(
        num_classes = 10,
        s1_emb_dim = 64,        # stage 1 - dimension
        s1_emb_kernel = 7,      # stage 1 - conv kernel
        s1_emb_stride = 4,      # stage 1 - conv stride
        s1_proj_kernel = 3,     # stage 1 - attention ds-conv kernel size
        s1_kv_proj_stride = 2,  # stage 1 - attention key / value projection stride
        s1_heads = 1,           # stage 1 - heads
        s1_depth = 1,           # stage 1 - depth
        s1_mlp_mult = 4,        # stage 1 - feedforward expansion factor
        s2_emb_dim = 192,       # stage 2 - (same as above)
        s2_emb_kernel = 3,
        s2_emb_stride = 2,
        s2_proj_kernel = 3,
        s2_kv_proj_stride = 2,
        s2_heads = 3,
        s2_depth = 2,
        s2_mlp_mult = 4,
        s3_emb_dim = 384,       # stage 3 - (same as above)
        s3_emb_kernel = 3,
        s3_emb_stride = 2,
        s3_proj_kernel = 3,
        s3_kv_proj_stride = 2,
        s3_heads = 4,
        s3_depth = 10,
        s3_mlp_mult = 4,
        dropout = 0.
    )

elif args.net=="deepvit":
    from models.deepvit import DeepViT
    net = DeepViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 10,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

# Cannot test at A-100 (inssufficient GPU)
elif args.net=="efficient_att_vit":
    from models.efficient import Nystromformer, ViT
    """
    efficient_transformer = Nystromformer(
        dim = 512,
        depth = 12,
        heads = 8,
        num_landmarks = 256
    )
    """
    
    efficient_transformer = Nystromformer(
        dim = 512,
        depth = 6,
        heads = 8,
        num_landmarks = 256
    )
    
    net = ViT(
        dim = 512,
        image_size = 256,
        patch_size = 32,
        num_classes = 10,
        transformer = efficient_transformer
    )
    
    """        
    net = ViT(
        dim = 512,
        image_size = 2048,
        patch_size = 32,
        num_classes = 10,
        transformer = efficient_transformer
    )
    """

elif args.net=="levit":
    from models.levit import LeViT
    net = LeViT(
        image_size = 224,
        num_classes = 10,
        stages = 3,             # number of stages
        dim = (256, 384, 512),  # dimensions at each stage
        depth = 4,              # transformer of depth 4 at each stage
        heads = (4, 6, 8),      # heads at each stage
        mlp_mult = 2,
        dropout = 0.1
    )

elif args.net=="max_vit":
    from models.max_vit import MaxViT
    net = MaxViT(
        num_classes = 10,
        dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
        dim = 96,                         # dimension of first layer, doubles every layer
        dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
        depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
        window_size = 7,                  # window size for block and grids
        mbconv_expansion_rate = 4,        # expansion rate of MBConv
        mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
        dropout = 0.1                     # dropout
    )


















# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

from tqdm import tqdm, tqdm_notebook, trange

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    # print('\nEpoch : ', epoch+1)
    net.train()
    with tqdm_notebook(total=len(train_ds), desc=f"Train Epoch {epoch+1}") as pbar:    
        train_losses = []
        train_accuracies = []
        train_loss = 0
        correct = 0
        total_num = 0
        for batch_idx, (inputs, targets) in enumerate(train_ds):
            inputs, targets = inputs.to(device), targets.to(device)
            # Train with amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_num += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total_num
            
            train_losses.append(loss.item())
            train_accuracies.append(acc)
            
            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss.item():.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")
            
            # progress_bar(batch_idx, len(train_ds), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    with tqdm_notebook(total=len(test_ds), desc=f"Test_ Epoch {epoch+1}") as pbar:    
        test_losses = []
        test_accuracies = []
        test_loss = 0
        correct = 0
        total_num = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_ds):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total_num += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(test_ds), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #     % (test_loss/(batch_idx+1), 100.*correct/total_num, correct, total_num))
                acc = 100.*correct/total_num
                test_losses.append(loss.item())
                test_accuracies.append(acc)

                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {loss.item():.4f} ({np.mean(test_losses):.4f}) Acc: {acc:.3f} ({np.mean(test_accuracies):.3f})")
        
    # Save checkpoint.
    acc = 100.*correct/total_num
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = 'Current time : ' + time.ctime() + ' ' + f'Epoch {epoch+1}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

net.cuda()

for epoch in range(args.n_epochs):
    start = time.time()
    trainloss     = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    print("Epoch :", epoch+1, "train_loss :{:.3f} , val_loss :{:.3f}".format(trainloss, val_loss),
          "val_acc :", acc, "lr :{:.7f}".format(optimizer.param_groups[0]["lr"]),
          "epoch_time :", int(time.time()-start) ,"sec")

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    # print(list_loss)

    
