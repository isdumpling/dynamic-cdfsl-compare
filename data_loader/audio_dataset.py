"""
音频数据集类
用于加载梅尔频谱图（PNG格式）
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path


class AudioMelSpectrogramDataset(Dataset):
    """
    加载梅尔频谱图的数据集类
    兼容原始CDFSL框架的ImageFolder结构
    """
    
    def __init__(self, root, transform=None, mode=None):
        """
        Args:
            root: 数据集根目录
            transform: 数据增强和转换
            mode: 'train' 或 'test' (目前不使用，保持兼容性)
        """
        self.root = root
        self.transform = transform
        self.mode = mode
        
        # 查找所有类别文件夹
        self.classes = self._find_classes(root)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 加载所有样本
        self.samples = self._make_dataset()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"找不到任何样本在: {root}")
    
    def _find_classes(self, root):
        """查找所有类别文件夹"""
        classes = []
        root_path = Path(root)
        
        if not root_path.exists():
            raise RuntimeError(f"数据集路径不存在: {root}")
        
        # 查找所有包含图像的子文件夹
        for item in sorted(root_path.iterdir()):
            if item.is_dir():
                # 检查是否包含PNG文件
                if any(item.glob('*.png')):
                    classes.append(item.name)
        
        return classes
    
    def _make_dataset(self):
        """创建样本列表 (path, class_index)"""
        samples = []
        root_path = Path(self.root)
        
        for class_name in self.classes:
            class_path = root_path / class_name
            class_idx = self.class_to_idx[class_name]
            
            # 查找所有PNG文件
            for img_path in sorted(class_path.glob('*.png')):
                samples.append((str(img_path), class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        返回:
            image: 转换后的图像张量 (C, H, W)
            label: 类别索引
        """
        img_path, label = self.samples[idx]
        
        # 加载梅尔频谱图（PNG格式）
        # 转换为RGB格式（复制灰度通道3次）以兼容现有的图像处理框架
        image = Image.open(img_path).convert('RGB')
        
        # 应用转换
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def __repr__(self):
        return (f"AudioMelSpectrogramDataset(root={self.root}, "
                f"num_classes={len(self.classes)}, "
                f"num_samples={len(self.samples)})")


# 为了与原始数据加载器兼容，创建包装函数
def ColdZone(data_path, mode=None):
    """
    cold_zone数据集加载器（源域）
    
    Args:
        data_path: 数据集根目录
        mode: 'train' 或 'test'（暂不使用，整个数据集作为训练集）
    """
    return AudioMelSpectrogramDataset(data_path, mode=mode)


def Hot1331(data_path, mode=None):
    """
    hot_13.31数据集加载器（目标域）
    
    Args:
        data_path: 数据集根目录
        mode: 'train' 或 'test'（暂不使用，整个数据集作为训练集）
    """
    return AudioMelSpectrogramDataset(data_path, mode=mode)


# 为了便于使用，也可以创建一个通用的加载器
def load_audio_dataset(dataset_name, data_path, mode=None):
    """
    通用音频数据集加载器
    
    Args:
        dataset_name: 数据集名称 ('cold_zone' 或 'hot_13.31')
        data_path: 数据集根目录
        mode: 'train' 或 'test'
    """
    if dataset_name.lower() in ['cold_zone', 'coldzone']:
        return ColdZone(data_path, mode=mode)
    elif dataset_name.lower() in ['hot_13.31', 'hot1331', 'hot_1331']:
        return Hot1331(data_path, mode=mode)
    else:
        raise ValueError(f"未知的数据集: {dataset_name}")

