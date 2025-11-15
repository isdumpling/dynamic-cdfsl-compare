"""
音频预处理工具
将wav文件转换为梅尔频谱图
"""
import os
import numpy as np
import librosa
import librosa.display
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path


def wav_to_melspectrogram(wav_path, output_path=None, sr=16000, n_mels=64, n_fft=400, hop_length=160, 
                         duration=1.0, save_as_image=True):
    """
    将wav文件转换为梅尔频谱图
    
    Args:
        wav_path: wav文件路径
        output_path: 输出路径，如果为None则不保存
        sr: 采样率
        n_mels: 梅尔滤波器数量
        n_fft: FFT窗口大小
        hop_length: 帧移
        duration: 音频持续时间（秒）
        save_as_image: 是否保存为图像格式
    
    Returns:
        mel_spectrogram: 梅尔频谱图 (n_mels, time_frames)
    """
    try:
        # 加载音频文件
        y, _ = librosa.load(wav_path, sr=sr, duration=duration)
        
        # 如果音频太短，进行填充
        target_length = int(sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # 计算梅尔频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels,
            fmax=sr/2
        )
        
        # 转换为对数刻度（dB）
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # 归一化到 [0, 255]
        mel_spec_normalized = ((mel_spec_db - mel_spec_db.min()) / 
                              (mel_spec_db.max() - mel_spec_db.min()) * 255).astype(np.uint8)
        
        # 保存
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if save_as_image:
                # 保存为图像
                img = Image.fromarray(mel_spec_normalized)
                img.save(output_path)
            else:
                # 保存为numpy数组
                np.save(output_path, mel_spec_db)
        
        return mel_spec_db
        
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None


def process_single_file(args):
    """处理单个文件（用于多进程）"""
    wav_path, output_path, params = args
    return wav_to_melspectrogram(wav_path, output_path, **params)


def convert_dataset_to_melspectrograms(input_dir, output_dir, sr=16000, n_mels=64, n_fft=400, 
                                      hop_length=160, duration=1.0, save_as_image=True, 
                                      num_workers=4):
    """
    将整个数据集的wav文件转换为梅尔频谱图
    
    Args:
        input_dir: 输入目录（包含cough和non-cough子文件夹）
        output_dir: 输出目录
        sr: 采样率
        n_mels: 梅尔滤波器数量
        n_fft: FFT窗口大小
        hop_length: 帧移
        duration: 音频持续时间（秒）
        save_as_image: 是否保存为PNG图像格式
        num_workers: 并行处理的进程数
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 准备参数
    params = {
        'sr': sr,
        'n_mels': n_mels,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'duration': duration,
        'save_as_image': save_as_image
    }
    
    # 查找所有wav文件
    wav_files = list(input_path.rglob('*.wav'))
    print(f"找到 {len(wav_files)} 个wav文件在 {input_dir}")
    
    if len(wav_files) == 0:
        print(f"警告：在 {input_dir} 中没有找到wav文件")
        return
    
    # 准备任务列表
    tasks = []
    for wav_file in wav_files:
        # 保持目录结构
        relative_path = wav_file.relative_to(input_path)
        if save_as_image:
            output_file = output_path / relative_path.with_suffix('.png')
        else:
            output_file = output_path / relative_path.with_suffix('.npy')
        
        tasks.append((str(wav_file), str(output_file), params))
    
    # 多进程处理
    if num_workers > 1:
        print(f"使用 {num_workers} 个进程进行并行处理...")
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_file, tasks), total=len(tasks)))
    else:
        print("单进程处理...")
        results = [process_single_file(task) for task in tqdm(tasks)]
    
    # 统计处理结果
    success_count = sum(1 for r in results if r is not None)
    print(f"成功处理 {success_count}/{len(wav_files)} 个文件")
    print(f"梅尔频谱图已保存到: {output_dir}")


def main():
    """主函数：转换cold_zone和hot_13.31数据集"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将wav文件转换为梅尔频谱图')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    parser.add_argument('--sr', type=int, default=16000, help='采样率')
    parser.add_argument('--n_mels', type=int, default=64, help='梅尔滤波器数量')
    parser.add_argument('--n_fft', type=int, default=400, help='FFT窗口大小')
    parser.add_argument('--hop_length', type=int, default=160, help='帧移')
    parser.add_argument('--duration', type=float, default=1.0, help='音频持续时间（秒）')
    parser.add_argument('--save_as_image', action='store_true', default=True, 
                       help='保存为PNG图像格式')
    parser.add_argument('--num_workers', type=int, default=4, help='并行处理的进程数')
    
    args = parser.parse_args()
    
    convert_dataset_to_melspectrograms(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sr=args.sr,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        duration=args.duration,
        save_as_image=args.save_as_image,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()

