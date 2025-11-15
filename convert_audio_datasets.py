"""
批量转换音频数据集为梅尔频谱图
处理cold_zone（源域）和hot_13.31（目标域）数据集
"""
import os
from utils.audio_preprocessing import convert_dataset_to_melspectrograms


def main():
    # 基础路径
    base_dir = "data"
    
    # 源域：cold_zone
    source_input_dir = os.path.join(base_dir, "cold_zone")
    source_output_dir = os.path.join(base_dir, "cold_zone_mel")
    
    # 目标域：hot_13.31
    target_input_dir = os.path.join(base_dir, "hot_13.31")
    target_output_dir = os.path.join(base_dir, "hot_13.31_mel")
    
    # 梅尔频谱图参数
    params = {
        'sr': 16000,           # 采样率
        'n_mels': 64,          # 梅尔滤波器数量（适合VGGish）
        'n_fft': 400,          # FFT窗口大小
        'hop_length': 160,     # 帧移
        'duration': 1.0,       # 音频持续时间（秒）
        'save_as_image': True, # 保存为PNG图像
        'num_workers': 1       # 并行处理进程数（临时改为单进程测试）
    }
    
    print("="*80)
    print("开始转换音频数据集为梅尔频谱图")
    print("="*80)
    
    # 转换源域数据集（cold_zone）
    print(f"\n[1/2] 处理源域数据集: {source_input_dir}")
    print("-"*80)
    if os.path.exists(source_input_dir):
        convert_dataset_to_melspectrograms(
            input_dir=source_input_dir,
            output_dir=source_output_dir,
            **params
        )
        print(f"✓ 源域数据集处理完成，保存到: {source_output_dir}")
    else:
        print(f"✗ 警告：源域目录不存在: {source_input_dir}")
    
    # 转换目标域数据集（hot_13.31）
    print(f"\n[2/2] 处理目标域数据集: {target_input_dir}")
    print("-"*80)
    if os.path.exists(target_input_dir):
        convert_dataset_to_melspectrograms(
            input_dir=target_input_dir,
            output_dir=target_output_dir,
            **params
        )
        print(f"✓ 目标域数据集处理完成，保存到: {target_output_dir}")
    else:
        print(f"✗ 警告：目标域目录不存在: {target_input_dir}")
    
    print("\n" + "="*80)
    print("所有数据集转换完成！")
    print("="*80)
    print(f"\n数据集结构：")
    print(f"  源域（Stage 1）: {source_output_dir}/")
    print(f"    ├── cough/")
    print(f"    └── non-cough/")
    print(f"\n  目标域（Stage 2）: {target_output_dir}/")
    print(f"    ├── cough/")
    print(f"    └── non-cough/")
    print("\n现在可以使用这些梅尔频谱图进行训练了！")


if __name__ == '__main__':
    main()

