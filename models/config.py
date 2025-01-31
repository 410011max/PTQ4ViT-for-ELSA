# --------------------------------------------------------
# TinyViT Config
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Adapted for TinyViT
# --------------------------------------------------------

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Dataset mean/std type
_C.DATA.MEAN_AND_STD_TYPE = "default"
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
# Data image filename format
_C.DATA.FNAME_FORMAT = '{}.jpeg'
# Data debug, when debug is True, only use few images
_C.DATA.DEBUG = False
# Percentage of validation data (to speed up the evaluation process)
_C.DATA.VAL_PERCENTAGE = 1.0

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'tiny_vit'
# Model name
_C.MODEL.NAME = 'tiny_vit'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# TinyViT Model
_C.MODEL.TINY_VIT = CN()
_C.MODEL.TINY_VIT.IN_CHANS = 3
_C.MODEL.TINY_VIT.DEPTHS = [2, 2, 6, 2]
_C.MODEL.TINY_VIT.NUM_HEADS = [3, 6, 12, 18]
_C.MODEL.TINY_VIT.WINDOW_SIZES = [7, 7, 14, 7]
_C.MODEL.TINY_VIT.EMBED_DIMS = [96, 192, 384, 576]
_C.MODEL.TINY_VIT.MLP_RATIO = 4.
_C.MODEL.TINY_VIT.MBCONV_EXPAND_RATIO = 4.0
_C.MODEL.TINY_VIT.LOCAL_CONV_SIZE = 3

# Deit Model
_C.MODEL.DEIT = CN()
_C.MODEL.DEIT.PATCH_SIZE = 16
_C.MODEL.DEIT.IN_CHANS = 3
_C.MODEL.DEIT.EMBED_DIM = 768
_C.MODEL.DEIT.DEPTH = 12
_C.MODEL.DEIT.NUM_HEADS = 12
_C.MODEL.DEIT.MLP_RATIO = 4.
_C.MODEL.DEIT.QKV_BIAS = True

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.SVD_CONFIG = []

# DISTILL
_C.DISTILL = CN()
_C.DISTILL.ENABLED = False
_C.DISTILL.TEACHER_LOGITS_PATH = ''
_C.DISTILL.SAVE_TEACHER_LOGITS = False
_C.DISTILL.LOGITS_TOPK = 100

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# train learning rate decay
_C.TRAIN.LAYER_LR_DECAY = 1.0
# batch norm is in evaluation mode when training
_C.TRAIN.EVAL_BN_WHEN_TRAINING = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# NAS settings
# -----------------------------------------------------------------------------
_C.NAS = CN()
_C.NAS.NUM_CHOICES_BLOCKS = 48
_C.NAS.SEARCH_SPACE = []
_C.NAS.PER_CAND_AFFINE = False
_C.NAS.SAMPLE_POLICY = CN()
_C.NAS.SAMPLE_POLICY.ENABLE_EPS = True
_C.NAS.SAMPLE_POLICY.EPS1 = CN()
_C.NAS.SAMPLE_POLICY.EPS1.MIN = 0.0 # do not sample from greedy pool if zero
_C.NAS.SAMPLE_POLICY.EPS1.MAX = 0.0
_C.NAS.SAMPLE_POLICY.EPS1.PATIENT_EPOCHS = 0
_C.NAS.SAMPLE_POLICY.EPS1.FIX_EPOCHS = 0
_C.NAS.SAMPLE_POLICY.EPS2 = CN()
_C.NAS.SAMPLE_POLICY.EPS2.MIN = 0.5 
_C.NAS.SAMPLE_POLICY.EPS2.MAX = 0.5
_C.NAS.SAMPLE_POLICY.EPS2.PATIENT_EPOCHS = 0
_C.NAS.SAMPLE_POLICY.EPS2.FIX_EPOCHS = 0
_C.NAS.SAMPLE_POLICY.NUM_SAMPLE_SUBNETS = 1
_C.NAS.SAMPLE_POLICY.NUM_KEPT_SUBNETS = 1
_C.NAS.SAMPLE_POLICY.FILTERED_POLICY = None
_C.NAS.SAMPLE_POLICY.CAND_POOL_SIZE = 1000
_C.NAS.SAMPLE_POLICY.ENABLE_CAND_POOL = False
_C.NAS.TARGET_COMP_RATIO = 0.565
_C.NAS.TEST_CONFIG = CN()
_C.NAS.TEST_CONFIG.UNIFORM_SUBNETS = []
_C.NAS.TEST_CONFIG.TEST_SUBNET = []
_C.NAS.INIT_CONFIG = None
_C.NAS.PROXY = CN()
_C.NAS.PROXY.ENABLE = False
_C.NAS.PROXY.IMG_PER_CLASSES = 1
_C.NAS.SEPARATE_CONFIG = []



# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration (swin)
_C.FUSED_WINDOW_PROCESS = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, cfg):
    _update_config_from_file(config, cfg)

    config.defrost()

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(cfg):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, cfg)

    return config
