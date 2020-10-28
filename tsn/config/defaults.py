from yacs.config import CfgNode as CN

_C = CN()

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# ---------------------------------------------------------------------------- #
# Distributed options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_NODES = 1

# The index of the current machine.
_C.RANK_ID = 0

# Distributed backend.
_C.DIST_BACKEND = "nccl"

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.MAX_ITER = 30000
_C.TRAIN.LOG_STEP = 10
_C.TRAIN.SAVE_STEP = 1000
_C.TRAIN.EVAL_STEP = 1000
_C.TRAIN.RESUME = False
_C.TRAIN.USE_TENSORBOARD = True

# ---------------------------------------------------------------------------- #
# DataSets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TYPE = 'RawFrame'
_C.DATASETS.MODALITY = 'RGB'
_C.DATASETS.SAMPLE_STRATEGY = 'SegSample'
_C.DATASETS.CLIP_LEN = 1
_C.DATASETS.FRAME_INTERVAL = 1
_C.DATASETS.NUM_CLIPS = 3
# for densesample test
_C.DATASETS.NUM_SAMPLE_POSITIONS = 10
# for vidoe decode
# Enable multi thread decoding.
_C.DATASETS.ENABLE_MULTI_THREAD_DECODE = False
# Decoding backend, options include `pyav` or `torchvision`
_C.DATASETS.DECODING_BACKEND = "pyav"
# train
_C.DATASETS.TRAIN = CN()
_C.DATASETS.TRAIN.NAME = 'HMDB51'
_C.DATASETS.TRAIN.DATA_DIR = 'data/hmdb51/rawframes'
_C.DATASETS.TRAIN.ANNOTATION_DIR = 'data/hmdb51'
# for hmdb51 and ucf101
_C.DATASETS.TRAIN.SPLIT = 1
# test
_C.DATASETS.TEST = CN()
_C.DATASETS.TEST.NAME = 'HMDB51'
_C.DATASETS.TEST.DATA_DIR = 'data/hmdb51/rawframes'
_C.DATASETS.TEST.ANNOTATION_DIR = 'data/hmdb51'
# for hmdb51 and ucf101
_C.DATASETS.TEST.SPLIT = 1

# ---------------------------------------------------------------------------- #
# Transform
# ---------------------------------------------------------------------------- #
_C.TRANSFORM = CN()
_C.TRANSFORM.MEAN = (0.485, 0.456, 0.406)  # (0.5, 0.5, 0.5)
_C.TRANSFORM.STD = (0.229, 0.224, 0.225)  # (0.5, 0.5, 0.5)

_C.TRANSFORM.TRAIN = CN()
_C.TRANSFORM.TRAIN.SCALE_JITTER = (256, 320)
_C.TRANSFORM.TRAIN.RANDOM_HORIZONTAL_FLIP = True
# (brightness, contrast, saturation, hue)
_C.TRANSFORM.TRAIN.COLOR_JITTER = (0.1, 0.1, 0.1, 0.1)
_C.TRANSFORM.TRAIN.RANDOM_ROTATION = 10
_C.TRANSFORM.TRAIN.RANDOM_CROP = True
_C.TRANSFORM.TRAIN.CENTER_CROP = False
_C.TRANSFORM.TRAIN.TRAIN_CROP_SIZE = 224
_C.TRANSFORM.TRAIN.RANDOM_ERASING = True

_C.TRANSFORM.TEST = CN()
_C.TRANSFORM.TEST.SHORTER_SIDE = 256
_C.TRANSFORM.TEST.CENTER_CROP = True
_C.TRANSFORM.TEST.THREE_CROP = False
_C.TRANSFORM.TEST.TEST_CROP_SIZE = 256

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH_SIZE = 16
_C.DATALOADER.TEST_BATCH_SIZE = 16
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = "TSN"
_C.MODEL.PRETRAINED = ""
_C.MODEL.SYNC_BN = False
_C.MODEL.INPUT_SIZE = (224, 224, 3)

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet50'
_C.MODEL.BACKBONE.PARTIAL_BN = False
_C.MODEL.BACKBONE.TORCHVISION_PRETRAINED = False
_C.MODEL.BACKBONE.ZERO_INIT_RESIDUAL = False

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = 'TSNHead'
_C.MODEL.HEAD.FEATURE_DIMS = 2048
_C.MODEL.HEAD.DROPOUT = 0.0
_C.MODEL.HEAD.NUM_CLASSES = 51

_C.MODEL.RECOGNIZER = CN()
_C.MODEL.RECOGNIZER.NAME = 'TSNRecognizer'

_C.MODEL.CONSENSU = CN()
_C.MODEL.CONSENSU.NAME = 'AvgConsensus'

_C.MODEL.CRITERION = CN()
_C.MODEL.CRITERION.NAME = 'CrossEntropyLoss'

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'sgd'
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 1e-4
# for sgd
_C.OPTIMIZER.SGD = CN()
_C.OPTIMIZER.SGD.MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# LR_Scheduler
# ---------------------------------------------------------------------------- #
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.NAME = 'multistep_lr'
_C.LR_SCHEDULER.IS_WARMUP = False
_C.LR_SCHEDULER.GAMMA = 0.1

# for SteLR
_C.LR_SCHEDULER.STEP_LR = CN()
_C.LR_SCHEDULER.STEP_LR.STEP_SIZE = 10000
# for MultiStepLR
_C.LR_SCHEDULER.MULTISTEP_LR = CN()
_C.LR_SCHEDULER.MULTISTEP_LR.MILESTONES = [15000, 25000]
# for CosineAnnealingLR
_C.LR_SCHEDULER.COSINE_ANNEALING_LR = CN()
_C.LR_SCHEDULER.COSINE_ANNEALING_LR.MINIMAL_LR = 3e-4
# for Warmup
_C.LR_SCHEDULER.WARMUP = CN()
_C.LR_SCHEDULER.WARMUP.ITERATION = 400
_C.LR_SCHEDULER.WARMUP.MULTIPLIER = 1.0
