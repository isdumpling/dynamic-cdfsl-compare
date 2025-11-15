#!/bin/bash

# 音频数据集训练脚本
# 用于ColdZone（源域）到HotZone（目标域）的跨域小样本学习
# 数据已通过 data/convert.py 转换为图像格式

echo "=========================================="
echo "音频数据集训练流程 (VGGish)"
echo "源域: ColdZone | 目标域: HotZone"
echo "=========================================="

# 检查数据是否已转换
if [ ! -d "data/ColdZone" ] || [ ! -d "data/HotZone" ]; then
    echo ""
    echo "❌ 错误：找不到数据目录！"
    echo "请确保已运行 data/convert.py 生成图像数据"
    echo "预期目录结构："
    echo "  - data/ColdZone/cough/*.png"
    echo "  - data/ColdZone/non-cough/*.png"
    echo "  - data/HotZone/cough/*.png"
    echo "  - data/HotZone/non-cough/*.png"
    exit 1
fi

echo ""
echo "[步骤1/3] 数据检查"
echo "----------------------------------------"
echo "✓ 数据目录存在"
echo "  源域: data/ColdZone"
echo "  目标域: data/HotZone"

# Stage 1: 源域预训练
echo ""
echo "[步骤2/3] Stage 1: 源域预训练（ColdZone）"
echo "----------------------------------------"
echo "使用VGGish backbone在ColdZone数据集上进行预训练..."

if [ ! -f "ckpt/ce_audio_ColdZone_vggish/last.ckpt" ]; then
    python main.py system=audio_ce \
      ++backbone=vggish \
      ++pretrained=true \
      ++data.dataset=ColdZone \
      ++data.val_dataset=ColdZone \
      ++data.n_way=2 \
      ++n_way=2 \
      ++model_name=ce_audio_ColdZone_vggish \
      ++trainer.gpus=1 \
      ++trainer.max_epochs=100
    
    if [ $? -ne 0 ]; then
        echo "❌ Stage 1训练失败！"
        exit 1
    fi
    echo "✓ Stage 1训练完成"
else
    echo "⚠ Stage 1模型已存在，跳过训练"
    echo "如需重新训练，请删除: ckpt/ce_audio_ColdZone_vggish/"
fi

# Stage 2: 目标域适应
echo ""
echo "[步骤3/3] Stage 2: 目标域适应（HotZone）"
echo "----------------------------------------"
echo "使用动态蒸馏在HotZone数据集上进行域适应..."

python main.py system=audio_distill \
  ++backbone=vggish \
  ++data.dataset=ColdZone \
  ++data.val_dataset=HotZone \
  ++data.n_way=2 \
  ++n_way=2 \
  ++unlabel_params.dataset=HotZone \
  ++data.num_episodes=600 \
  ++ckpt_preload=ckpt/ce_audio_ColdZone_vggish/last.ckpt \
  ++model_name=dynamic_cdfsl_audio_HotZone_vggish \
  ++trainer.gpus=1 \
  ++trainer.max_epochs=60

if [ $? -ne 0 ]; then
    echo "❌ Stage 2训练失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 训练流程完成！"
echo "=========================================="
echo ""
echo "模型保存位置："
echo "  - Stage 1: ckpt/ce_audio_ColdZone_vggish/last.ckpt"
echo "  - Stage 2: ckpt/dynamic_cdfsl_audio_HotZone_vggish/last.ckpt"
echo ""
echo "进行Few-shot评估："
echo "  python main.py system=few_shot \\"
echo "    ++backbone=vggish \\"
echo "    ++data.test_dataset=HotZone \\"
echo "    ++data.n_way=2 \\"
echo "    ++data.n_shot=5 \\"
echo "    ++data.n_query=15 \\"
echo "    ++ckpt=ckpt/dynamic_cdfsl_audio_HotZone_vggish/last.ckpt \\"
echo "    ++test=true"
echo ""

