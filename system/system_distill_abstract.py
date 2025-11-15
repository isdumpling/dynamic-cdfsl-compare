import copy
from typing import MutableSequence
from torch import nn
from torch.functional import Tensor
from system.multiloader_mixin import MultiTrainLoaderMixin
from system import system_abstract
import torch
import abc
import torchmetrics
from torch.nn import functional as F
import math
import time


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(MultiTrainLoaderMixin, system_abstract.LightningSystem):
    """ Abstract class """
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)
        if isinstance(self.hparams.data.dataset, MutableSequence) and len(
                self.hparams.data.dataset) == 1:
            self.hparams.data.dataset = self.hparams.data.dataset[0]
        self._create_model(self.num_classes)

        if self.hparams.ckpt_preload is not None:
            ckpt = torch.load(
                self.hparams.ckpt_preload,
                map_location=lambda storage, loc: storage)['state_dict']
            self.load_state_dict(ckpt, strict=False)

        self.create_student()
        self.create_teacher()

        # self.distill_loss = DistillLoss(self.hparams.teacher_temp)
        self.ce_loss = nn.CrossEntropyLoss()
        # torchmetrics v0.11+ requires 'task' parameter
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        
        # 初始化训练时间记录
        self.train_start_time = None
        self.train_end_time = None

    def create_student(self):
        student_modules = []
        student_modules.append(self.feature)
        st_header = copy.deepcopy(self.get_header())
        if self.hparams.reset_student_head:
            st_header.reset_parameters()
        student_modules.append(st_header)
        self.student = nn.Sequential(*student_modules)
        self.student.requires_grad_(True)

    @abc.abstractmethod
    def create_teacher(self):
        pass

    def forward(self, x):
        return self.student(x)

    def set_forward(self, *x_list):
        dims = [x.shape[0] for x in x_list]
        scores_all = self(torch.cat(x_list, dim=0))
        return scores_all.split(dims)

    # --------------------------------- training --------------------------------- #

    @abc.abstractmethod
    def _forward_loss(self, batch, batch_u):
        pass

    def training_step(self, batch, batch_idx=0):
        x, y = batch

        batch_train = (x, y)
        batch_u = self.get_unlabel_batch()

        loss_ce, loss_pseudo, top1 = self._forward_loss(batch_train, batch_u)

        l2 = 1
        if self.hparams.cosine_weight:
            l2 = self.get_cosine_weight()

        loss = self.hparams.lm_ce * loss_ce + l2 * self.hparams.lm_u * loss_pseudo

        tqdm_dict = {
            "loss_train": loss,
            "l_ce": loss_ce,
            "l_u": loss_pseudo,
            "top1": top1
        }

        self.log_dict(tqdm_dict, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # --------------------------------- utilities -------------------------------- #\
    def cross_entropy(self, logits, y_gt) -> Tensor:
        return cross_entropy(logits, y_gt)

    def get_header(self) -> nn.Module:
        header = self.classifier
        return header

    def distill_loss(self, student_out, teacher_out):
        teacher_out /= self.hparams.teacher_temp
        loss = self.cross_entropy(student_out, teacher_out.softmax(dim=-1))
        return loss

    def get_cosine_weight(self):
        total_steps = self.trainer.max_epochs * self.trainer.num_training_batches
        current_step = self.trainer.global_step
        multiplier = min(
            1, 0.75 * (1 - math.cos(math.pi * current_step / total_steps)))
        return multiplier
    
    def on_train_start(self):
        """ 记录训练开始时间 """
        self.train_start_time = time.time()
        self.log("train_start_timestamp", self.train_start_time)
    
    def on_train_end(self):
        """ 记录训练结束时间和总训练时长 """
        self.train_end_time = time.time()
        if self.train_start_time is not None:
            train_duration = self.train_end_time - self.train_start_time
            train_duration_hours = train_duration / 3600
            train_duration_minutes = train_duration / 60
            
            print(f"\n{'='*60}")
            print(f"Stage 2 训练完成统计 (目标域适应):")
            print(f"  开始时间: {time.ctime(self.train_start_time)}")
            print(f"  结束时间: {time.ctime(self.train_end_time)}")
            print(f"  总训练时长: {train_duration_hours:.2f} 小时 ({train_duration_minutes:.2f} 分钟, {train_duration:.2f} 秒)")
            print(f"{'='*60}\n")


def cross_entropy(logits, y_gt) -> Tensor:
    if len(y_gt.shape) < len(logits.shape):
        return F.cross_entropy(logits, y_gt, reduction='mean')
    else:
        return (-y_gt * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
