# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

# base: cosine decay learning rate schedular
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# adding the triangular2 learning rate schedular
def adjust_learning_rate_triangular2(optimizer, epoch, args):
    """Decay the learning rate with triangular2"""
    num_cycles = 5
    cycle_length = args.epochs / num_cycles
    cycle = math.floor(1 + epoch / (2 * cycle_length))
    x = abs(epoch / cycle_length - 2 * cycle + 1)
    lr = args.min_lr + (args.lr - args.min_lr) * max(0, (1 - x)) / (2 ** (cycle - 1))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# adding the step decay learning rate schedular
def adjust_learning_rate_step_decay(optimizer, epoch, args):
    """Decay the learning rate with step decay after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        decay_factor = 0.65
        step_size = 2
        lr = args.lr * (decay_factor ** ((epoch - args.warmup_epochs) // step_size))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

# adding the exponential decay learning rate schedular
def adjust_learning_rate_exp_decay(optimizer, epoch, args):
    """Natural exponential decay learning rate with warm-up"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        min_lr = 1e-5
        decay_factor = math.exp(math.log(min_lr / args.lr) / (args.epochs - args.warmup_epochs))
        lr = args.lr * decay_factor ** (epoch - args.warmup_epochs)
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
