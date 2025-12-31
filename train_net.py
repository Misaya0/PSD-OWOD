import os
import itertools
import time
import weakref
from contextlib import ExitStack
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict
import torch.nn as nn
import torch
from fvcore.nn.precise_bn import get_bn_modules
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import detectron2.detectron2.utils.comm as comm
from detectron2.detectron2.utils.logger import setup_logger
from detectron2.detectron2.checkpoint import DetectionCheckpointer
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.data import build_detection_train_loader
from detectron2.detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results, inference_context
from detectron2.detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.detectron2.modeling import build_model

from core import DatasetMapper, add_config
from core.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from core.pascal_voc import register_pascal_voc
from core.pascal_voc_evaluation import PascalVOCDetectionEvaluator


class Register:
    def __init__(self, dataset_root, split, cfg):
        self.dataset_root = dataset_root
        self.super_split = split.split('/')[0]
        self.cfg = cfg

        self.PREDEFINED_SPLITS_DATASET = {
            "my_train": split,
            "my_val": os.path.join(self.super_split, 'test')
        }

    def register_dataset(self):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        """
        for name, split in self.PREDEFINED_SPLITS_DATASET.items():
            register_pascal_voc(name, self.dataset_root, self.super_split, split, self.cfg)


class ProcessedDataset(Dataset):
    def __init__(self, data_loader, model):
        self.data_loader = data_loader  # 原始的DataLoader
        self.model = model  # 需要处理数据的模型

        # 处理后的数据
        self.processed_data = self._process_data()

    def _process_data(self):
        processed_data = []

        for idx, inputs in enumerate(self.data_loader):
            gt_instances = self.model(inputs)  # 获取模型的输出
            for i in range(len(inputs)):  # 遍历每个样本
                data_item = inputs[i]  # 获取当前样本的数据
                data_item['instances'] = gt_instances[i]['instances']  # 将模型的输出保存到'instances'字段
                processed_data.append(data_item)  # 将处理后的样本添加到列表中
                print("Processed {} instances".format(len(processed_data)))

        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, index):
        return self.processed_data[index]
class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted """

    def __init__(self, cfg, args, rew_cfg=None):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        self.rew_model = None
        if rew_cfg != None:
            self.rew_model = self.build_model(rew_cfg)
            self.rew_model = create_ddp_model(self.rew_model, broadcast_buffers=False)
            DetectionCheckpointer(self.rew_model, save_dir=cfg.WEIGHTS_DIR).resume_or_load(
                rew_cfg.MODEL.WEIGHTS, resume=args.resume
            )

        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        # self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
        #     model, data_loader, optimizer, rew_model=self.rew_model
        # )
        if cfg.SOLVER.AMP.ENABLED:
            self._trainer = AMPTrainer(model, data_loader, optimizer)
        else:
            self._trainer = SimpleTrainer(model, data_loader, optimizer)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = cfg.SOLVER.START_ITER
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return PascalVOCDetectionEvaluator(dataset_name, cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def inference_rew(cls, cfg, rew_model, output_path=None):
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_train_loader(cfg)
            processed_data = []
            for batch in data_loader:
                for data in batch:
                    # 将图像传入 rew_model，得到预测结果
                    with torch.no_grad():
                        gt_instances = rew_model(data)
                    for i in range(len(data)):  # 遍历每个样本
                        data_item = data[i]  # 获取当前样本的数据
                        data_item['instances'] = gt_instances[i]['instances']  # 将模型的输出保存到'instances'字段
                        processed_data.append(data_item)  # 将处理后的样本添加到列表中
                        print("Processed {} instances".format(len(processed_data)))




            # processed_dataset = ProcessedDataset(data_loader, model)
            # processed_data_loader = DataLoader(processed_dataset, batch_size=12, shuffle=False, num_workers=0,pin_memory=True)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))


        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


def setup(args):
    """
    Create configs and perform basic setups.
    """
    # CUDA_VISIBLE_DEVICES = 0,1,2,3
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.set_new_allowed(True)
    add_config(cfg,args)
    file_name = os.path.basename(args.config_file)
    cfg.CONFIG_NAME = os.path.splitext(file_name)[0]
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.MODEL.WEIGHTS = "output/M-OWODB/model_0019999.pth"
    # cfg.SOLVER.MAX_ITER = 100
    cfg.freeze()

    default_setup(cfg, args)

    # cfg.SOLVER.IMS_PER_BATCH = 2
    return cfg
def rew_setup(args):
    """
    Create configs and perform basic setups.
    """
    # CUDA_VISIBLE_DEVICES = 0,1,2,3
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.set_new_allowed(True)
    add_config(cfg,args)
    file_name = os.path.basename(args.rew_config_file)
    cfg.CONFIG_NAME = os.path.splitext(file_name)[0]
    cfg.WEIGHTS_DIR = ""
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.rew_config_file)
    cfg.merge_from_list(args.opts)
    # cfg.SOLVER.IMS_PER_BATCH = 4
    # cfg.MODEL.WEIGHTS = "output/M-OWODB/model_0019999.pth"
    # cfg.SOLVER.MAX_ITER = 100
    cfg.freeze()

    default_setup(cfg, args)

    # cfg.SOLVER.IMS_PER_BATCH = 2
    return cfg

def main(args):
    rew_cfg = None
    cfg = setup(args)
    if args.rew_config_file != "":
        rew_cfg = rew_setup(args)
    data_register = Register('./datasets/', args.task, cfg)
    data_register.register_dataset()
    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                              resume=args.resume)
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                                           resume=args.resume)
        res = Trainer.ema_test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    # if rew_cfg != None:
    #     rew_model = Trainer.build_model(rew_cfg)
    #     DetectionCheckpointer(rew_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #         rew_cfg.MODEL.WEIGHTS, resume=args.resume
    #     )



    trainer = Trainer(cfg, args)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    parser0 = default_argument_parser()
    parser0.add_argument("--task", default="")
    parser0.add_argument("--rew-config-file", default="")
    parser0.add_argument("--rew-inference", action="store_true")
    # parser0.add_argument("--resume", action='store_true')
    # parser0.add_argument("MODEL_WEIGHTS", type=str, help="Path to the model weights file")
    args = parser0.parse_args()
    # args.resume = True
    # args.MODEL_WEIGHTS = "/2T/gzj/OrthogonalDet-main/output/M-OWODB/t2/model_0019999.pth"

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
