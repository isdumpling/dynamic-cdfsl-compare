import torch
try:
    from torchmetrics.functional import accuracy
    # 检查是否是新版本（需要task参数）
    import inspect
    _accuracy_sig = inspect.signature(accuracy)
    _needs_task_param = 'task' in _accuracy_sig.parameters
except ImportError:
    from pytorch_lightning.metrics.functional import accuracy
    _needs_task_param = False
from torch.nn import functional as F
import time

from system import system_abstract
from typing import MutableSequence


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(system_abstract.LightningSystem):
    def __init__(self, hparams, datamodule=None):
        super().__init__(hparams, datamodule)

        if isinstance(self.hparams.data.dataset, MutableSequence) and len(
                self.hparams.data.dataset) == 1:
            self.hparams.data.dataset = self.hparams.data.dataset[0]
        self._create_model(self.num_classes)
        
        # 初始化训练时间记录
        self.train_start_time = None
        self.train_end_time = None

    def forward(self, x):
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores, out

    # --------------------------------- training --------------------------------- #
    def training_step(self, batch, batch_idx):
        x, y = batch
        if isinstance(x, MutableSequence) or isinstance(x, tuple):
            # MoCo augmentation
            x = torch.cat(x, dim=0)
            y = torch.cat((y, y), dim=0)

        scores, _ = self(x)
        _, predicted = torch.max(scores.data, 1)
        loss = torch.nn.functional.cross_entropy(scores, y)

        with torch.no_grad():
            # 兼容新旧版本的accuracy函数
            if _needs_task_param:
                # 新版本需要task参数
                num_classes = scores.size(1)
                task = 'binary' if num_classes == 2 else 'multiclass'
                accur = accuracy(predicted.detach(), y, task=task, num_classes=num_classes)
            else:
                accur = accuracy(predicted.detach(), y)

        tqdm_dict = {"loss_train": loss, "top1": accur}
        self.log_dict(tqdm_dict,
                      prog_bar=True,
                      on_step=True,
                      on_epoch=True,
                      logger=True)
        return loss

    def get_feature_extractor(self):
        """ return feature extractor """
        return self.feature
    
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
            print(f"Stage 1 训练完成统计 (源域预训练):")
            print(f"  开始时间: {time.ctime(self.train_start_time)}")
            print(f"  结束时间: {time.ctime(self.train_end_time)}")
            print(f"  总训练时长: {train_duration_hours:.2f} 小时 ({train_duration_minutes:.2f} 分钟, {train_duration:.2f} 秒)")
            print(f"{'='*60}\n")
