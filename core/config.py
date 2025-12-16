from detectron2.detectron2.config import CfgNode as CN
def add_config(cfg,args):
    cfg.MODEL.NUM_CLASSES = 81
    cfg.MODEL.NUM_PROPOSALS = 300

    # batch size
    # cfg.SOLVER.IMS_PER_BATCH = 2



    # RCNN Head.
    cfg.MODEL.NHEADS = 8
    cfg.MODEL.DROPOUT = 0.0
    cfg.MODEL.DIM_FEEDFORWARD = 2048
    cfg.MODEL.ACTIVATION = 'relu'
    cfg.MODEL.HIDDEN_DIM = 256
    cfg.MODEL.NUM_CLS = 1
    cfg.MODEL.NUM_REG = 3
    cfg.MODEL.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.NUM_DYNAMIC = 2
    cfg.MODEL.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.CLASS_WEIGHT = 2.0
    cfg.MODEL.NC = True
    cfg.MODEL.NC_WEIGHT = 0.1
    cfg.MODEL.GIOU_WEIGHT = 2.0
    cfg.MODEL.L1_WEIGHT = 5.0
    cfg.MODEL.DEEP_SUPERVISION = True
    cfg.MODEL.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.ALPHA = 0.25
    cfg.MODEL.GAMMA = 2.0
    cfg.MODEL.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.OTA_K = 5
    cfg.MODEL.FORWARD_K = 10

    # WARM_UP
    cfg.MODEL.CHANGE_START = 0

    # Diffusion
    cfg.MODEL.SNR_SCALE = 2.0
    cfg.MODEL.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.USE_NMS = True
    cfg.MODEL.M_STEP = 20
    cfg.MODEL.SAMPLING_METHOD = 'Random_'

    # Disentanglement
    cfg.MODEL.DISENTANGLED = 2  # 0: RandBox, 1: separate head, 2: feature orthogonality
    cfg.MODEL.DECORR_WEIGHT = 1.  # weight for prediction decorrelation loss

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # OW EVALUATION
    cfg.TEST.PREV_INTRODUCED_CLS = 0
    cfg.TEST.CUR_INTRODUCED_CLS = 20
    cfg.TEST.PREV_CLASSES = ()  # previously seen classes
    cfg.TEST.MASK = 1  # 0: no mask, 1: mask unseen classes, 2: mask prev and unseen classes
    cfg.TEST.SCORE_THRESH = 0.15  # follow RandBox

    cfg.OPENSET = CN()
    cfg.OPENSET.ENABLE_REW = False

    cfg.OPENSET.REW_INFERENCE = args.rew_inference


    cfg.OPENSET.REW = CN()
    cfg.OPENSET.REW.AE_INTER = [64, 32, 16, 8, 4]
    cfg.OPENSET.REW.FITER_PERCENT = 0.2
    cfg.OPENSET.REW.HARD_THR = False
    cfg.OPENSET.REW.ENABLE_SL = True
    cfg.OPENSET.REW.GAMMA = 2.0
    cfg.OPENSET.REW.ALPHA = 1.0
    cfg.OPENSET.REW.SAMPLING_ITERS = 2500
    cfg.OPENSET.REW.NUM_SAMPLES = 160000
    cfg.OPENSET.REW.UPDATE_WEIBULL = False
    cfg.OPENSET.REW.UPDATE_FREQ = 1000



    cfg.OPENSET.ENABLE_OLN = False
    cfg.OPENSET.OLN_INFERENCE = False
    cfg.OPENSET.INFERENCE_SELT_TRAIN = False
    cfg.OPENSET.OLN = CN()
    cfg.OPENSET.OLN.IOU_LABELS = [-1, 1]
    cfg.OPENSET.OLN.IOU_THRESHOLDS = [0.3]
    cfg.OPENSET.OLN.NMS_THRESH = 0.7
    cfg.OPENSET.OLN.POSITIVE_FRACTION = 1.0
    cfg.OPENSET.OLN.POST_NMS_TOPK_TEST = 10000
    cfg.OPENSET.OLN.POST_NMS_TOPK_TRAIN = 10000
    cfg.OPENSET.OLN.PRE_NMS_TOPK_TEST = 2000
    cfg.OPENSET.OLN.PRE_NMS_TOPK_TRAIN = 2000
    cfg.OPENSET.OLN.BATCH_SIZE_PER_IMAGE = 256

    cfg.OPENSET.NUM_KNOWN_CLASSES = 20
    cfg.OPENSET.NUM_PREV_KNOWN_CLASSES = 0
    cfg.OPENSET.EVAL_UNKNOWN = False
    cfg.OPENSET.OUTPUT_PATH_REW = "./"
    cfg.OPENSET.CALIBRATE = True
    cfg.OPENSET.CALIBRATE_WEIGHT = 5.0
    cfg.OPENSET.FILTER_THRESH = 0.5
