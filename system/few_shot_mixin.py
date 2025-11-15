from utils import utils_system
import torch
import torch.nn as nn
import numpy as np
import torch
import numpy as np
from torchmetrics.functional import accuracy, f1_score


def is_ddp_training(trainer):
    """检查是否使用DDP训练，兼容新旧版本PyTorch Lightning"""
    if hasattr(trainer, 'use_ddp'):
        return trainer.use_ddp
    if hasattr(trainer, 'strategy'):
        strategy_name = str(type(trainer.strategy).__name__).lower()
        return 'ddp' in strategy_name
    return False
try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except ImportError:
    try:
        from pytorch_lightning.utilities.distributed import rank_zero_only
    except ImportError:
        def rank_zero_only(fn):
            def wrapper(*args, **kwargs):
                from utils.utils_system import get_rank
                if get_rank() == 0:
                    return fn(*args, **kwargs)
            return wrapper
from system import utils_finetune as ftune
from typing import MutableSequence


class FewShotMixin():
    def _init_meter(self) -> None:
        if self.trainer.testing:
            datasets = self.hparams.data.test_dataset
        else:
            datasets = self.hparams.data.val_dataset

        self.val_names = datasets if isinstance(
            datasets, MutableSequence) else [datasets]

        self.val_len = 1 if not isinstance(datasets,
                                           MutableSequence) else len(datasets)

        self.acc_meter = [
            utils_system.AverageMeter(use_ddp=is_ddp_training(self.trainer))
            for _ in range(self.val_len)
        ]
        
        # 添加 macro F1 score 计量器
        self.f1_meter = [
            utils_system.AverageMeter(use_ddp=is_ddp_training(self.trainer))
            for _ in range(self.val_len)
        ]

    def few_shot_validation_step(self, batch, batch_idx, dataset_idx=0):
        if not hasattr(self, 'acc_meter'):
            self._init_meter()

        out = self.few_shot_finetune(batch, dataset_idx)
        if self.hparams.print_val:
            self.loguru_log(
                f"val :: ({self.val_names[dataset_idx]}) : ({batch_idx}) : {out['acc']:.4f}, , acc: {out['acc']:.4f}, acc_mean: {out['acc_mean']:.4f}",
                level="DEBUG")

        if hasattr(self.logger, "log_csv"):
            self.logger.log_csv(
                {
                    "dataset": self.val_names[dataset_idx],
                    "accuracy": out['acc'],
                    "mean": out['acc_mean']
                },
                step=batch_idx)

        return out

    def few_shot_val_end(self, outputs):
        mean_val = []
        mean_std = []
        mean_f1 = []
        f1_std = []
        # max_val = []
        for dataset_idx in range(self.val_len):
            acc_mean = self.acc_meter[dataset_idx].mean
            acc_std = self.acc_meter[dataset_idx].std
            
            # 获取 macro F1 score 统计
            f1_mean = self.f1_meter[dataset_idx].mean
            f1_std_val = self.f1_meter[dataset_idx].std

            if self.logger is not None:
                self.loguru_log(
                    f"Test Acc ({self.val_names[dataset_idx]}) = {acc_mean:4.4f} +- {acc_std:4.4f}, Macro F1 = {f1_mean:4.4f} +- {f1_std_val:4.4f}",
                    level="DEBUG")
            self.acc_meter[dataset_idx].reset()
            self.f1_meter[dataset_idx].reset()
            mean_val.append(acc_mean)
            mean_std.append(acc_std)
            mean_f1.append(f1_mean)
            f1_std.append(f1_std_val)

            if dataset_idx == 0:
                tqdm_dict = {
                    "acc_mean": acc_mean, 
                    "acc_std": acc_std,
                    "macro_f1_mean": f1_mean,
                    "macro_f1_std": f1_std_val
                }
                self.log_dict(tqdm_dict, prog_bar=False)

        acc_mean = torch.mean(torch.stack(mean_val))
        acc_std = torch.mean(torch.stack(mean_std))
        f1_mean_full = torch.mean(torch.stack(mean_f1))
        f1_std_full = torch.mean(torch.stack(f1_std))
        
        tqdm_dict = {
            "acc_mean_full": acc_mean,
            "macro_f1_full": f1_mean_full
        }
        self.log_dict(tqdm_dict, prog_bar=True)
        
        # 打印汇总统计
        print(f"\n{'='*60}")
        print(f"验证结果汇总:")
        print(f"  准确率 (Accuracy): {acc_mean:.4f} ± {acc_std:.4f}")
        print(f"  Macro F1 Score: {f1_mean_full:.4f} ± {f1_std_full:.4f}")
        print(f"{'='*60}\n")
        
        return tqdm_dict

    @rank_zero_only
    def loguru_log(self, msg, level="INFO"):
        try:
            self.logger.log(msg)
        except AttributeError:
            self.logger.experiment.log(msg)

    def get_fewshot_batch(self, batch):
        x, y = batch
        self.n_way = self.hparams.n_way
        self.n_shot = self.hparams.n_shot

        self.n_query = x.size(1) - self.n_shot
        labels_dict_map = dict(
            zip(y[:, 0].data.cpu().numpy(), range(self.n_way)))

        # support set
        y_support = torch.from_numpy(np.repeat(range(
            self.n_way), self.n_shot)).to(self.device,
                                          non_blocking=True)  # (25,)

        x_support = x[:, :self.n_shot, :, :, :].contiguous().view(
            self.n_way * self.n_shot,
            *x.size()[2:])  # (25, 3, 224, 224)

        # query set
        x_query = x[:, self.n_shot:, :, :, :].contiguous().view(
            self.n_way * self.n_query,
            *x.size()[2:])

        y_query = torch.arange(self.n_way, device=self.device,
                               dtype=x.dtype).repeat_interleave(self.n_query)

        return x_support, y_support, x_query, y_query, labels_dict_map, y

    def few_shot_finetune(self, batch, dataset_idx=0):
        x_support, y_support, x_query, y_query, *_ = self.get_fewshot_batch(
            batch)

        encoder = self.get_feature_extractor()

        topk_ind, _ = ftune.LR(encoder,
                               x_support,
                               y_support,
                               x_query,
                               norm=self.hparams.eval.fine_tune.use_norm)

        # 兼容新版本 torchmetrics，添加 task 参数
        _val = accuracy(topk_ind, y_query.long(), task='multiclass', num_classes=self.n_way)
        self.acc_meter[dataset_idx].add(_val, n=1)
        
        # 计算 macro F1 score
        _f1 = f1_score(topk_ind, y_query.long(), task='multiclass', num_classes=self.n_way, average='macro')
        self.f1_meter[dataset_idx].add(_f1, n=1)

        return {
            "acc": self.acc_meter[dataset_idx].value,
            "acc_mean": self.acc_meter[dataset_idx].mean,
            "macro_f1": self.f1_meter[dataset_idx].value,
            "macro_f1_mean": self.f1_meter[dataset_idx].mean
        }


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim, n_way)
        # self.fc = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(),
        #                         nn.Linear(dim // 2, n_way))

    def forward(self, x):
        x = self.fc(x)
        return x
