#! /user/bin/env python
# -*- coding: utf-8 -*-
"""
LongCat-Image-Edit-Turbo在昇腾NPU上的多卡并行推理脚本
采用Pipeline并行策略：将不同组件分配到不同NPU卡上，避免CPU offloading的搬运开销
设备分配：Text Encoder + VAE -> npu:1 | Transformer -> npu:2 | 0号卡不使用
"""

import os
# 禁用flash attention以解决NPU兼容性问题
os.environ['LONGCAT_DISABLE_FLASH_ATTN'] = '1'

import torch
import argparse
from PIL import Image
from pathlib import Path
from diffusers import LongCatImageEditPipeline
from transformers import AutoProcessor
from diffusers.models import LongCatImageTransformer2DModel

def setup_environment():
    """设置运行环境 - 多卡并行模式"""
    # 设置各组件的目标设备，不使用0号卡
    # npu:1 承载 Text Encoder + VAE（体积较小，共用一张卡）
    # npu:2 承载 Transformer（最大组件，独占一张卡）
    device_te = torch.device('npu:1')
    device_transformer = torch.device('npu:2')

    print(f"Text Encoder / VAE 设备：{device_te}")
    print(f"Transformer 设备：{device_transformer}")

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    if torch.npu.is_available():
        torch.npu.manual_seed(42)

    return device_te, device_transformer

def load_model(checkpoint_dir, device_transformer):
    """加载模型和处理器"""
    print("开始加载模型和处理器...")

    # 加载文本处理器
    text_processor = AutoProcessor.from_pretrained(
        checkpoint_dir, 
        subfolder='tokenizer', 
    )
    print("文本处理器加载完成")

    # 加载transformer模型，明确禁用flash attention
    transformer = LongCatImageTransformer2DModel.from_pretrained(
        checkpoint_dir, 
        subfolder='transformer', 
        torch_dtype=torch.bfloat16, 
        use_flash_attention=False  # 关键！关闭flash attention 以解决NPU兼容问题
    )
    # 将transformer移动到npu:2设备
    transformer = transformer.to(device_transformer)
    print(f"Transformer模型加载完成，设备：{device_transformer}")

    return text_processor, transformer

def create_pipeline(checkpoint_dir, transformer, text_processor, device_te, device_transformer):
    """创建编辑管道 - 多卡并行模式"""
    print("创建图像编辑管道...")

    pipe = LongCatImageEditPipeline.from_pretrained(
        checkpoint_dir, 
        transformer=transformer, 
        text_processor=text_processor, 
        torch_dtype=torch.bfloat16,
    )

    # === 多卡并行：手动将各组件分配到不同设备 ===
    # 不再使用 pipe.to(device) 统一移动，而是分别放到各自的目标卡上

    # Text Encoder 放到 npu:1
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        pipe.text_encoder = pipe.text_encoder.to(device_te)
        print(f"Text Encoder 已放置到 {device_te}")

    # VAE 放到 npu:1（与Text Encoder共用）
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        pipe.vae = pipe.vae.to(device_te)
        print(f"VAE 已放置到 {device_te}")

    # Transformer 确保在 npu:2 上（load_model时已移动，此处再次确认）
    if hasattr(pipe, 'transformer') and pipe.transformer is not None:
        pipe.transformer = pipe.transformer.to(device_transformer)
        print(f"Transformer 已确认在 {device_transformer}")

    # 注意：不再使用 enable_model_cpu_offload()
    # 多卡部署已将组件分散到不同卡上，解决了显存问题
    # 同时避免了CPU offloading中CPU<->NPU的搬运开销，推理速度更快

    print("图像编辑管道创建完成")
    return pipe

def perform_editing(pipe, input_image_path, prompt, output_path):
    """执行图像编辑"""
    print(f"开始图像编辑任务...")
    print(f"输入图像：{input_image_path}")
    print(f"编辑指令：{prompt}")

    # 加载输入图像
    try:
        img = Image.open(input_image_path).convert('RGB')
        print(f"图像加载成功，尺寸{img.size}")
    except Exception as e:
        print(f"图像加载失败：{e}")
        return False

    # 创建随机数生成器（保持在CPU上，pipeline会自动处理张量的设备转移）
    generator = torch.Generator("cpu").manual_seed(43)

    # 执行编辑
    try:
        print("开始推理...")
        result = pipe(
            img, 
            prompt, 
            negative_prompt="",   # 负面提示词，可用于排除某些内容
            guidance_scale=1.0,   # Turbo模型guidance_scale必须约等于1
            num_inference_steps=8,   # Turbo模型使用8步推理
            num_images_per_prompt=1,   # 每个提示生成的图像数量
            generator=generator
        )

        # 获取图像结果
        edited_image = result.images[0]

        # 保存结果
        edited_image.save(output_path)
        print(f"编辑完成！结果已保存到：{output_path}")

        return True

    except Exception as e:
        print(f"编辑过程出错：{e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LongCat-Image-Edit-Turbo推理脚本")
    parser.add_argument('--input', type=str, 
                        default='home/ma-user/work/d50059042/LC-IET-test/images/input/test_image0.jpg', 
                        help="输入图像路径")
    parser.add_argument('--prompt', type=str, 
                            default='在图片右上角加入从天而降的大陨石，冒着火花向地面袭来', 
                            help="编辑指令提示词")
    parser.add_argument('--output', type=str, 
                            default='home/ma-user/work/d50059042/LC-IET-test/images/output/test0_edited.png', 
                            help="输出图像路径")
    parser.add_argument('--checkpoint', type=str, 
                            default='home/ma-user/work/d50059042/image_model/LongCat-Image-Edit-Turbo', 
                            help="模型权重路径")

    args = parser.parse_args()

    print("=" * 60)
    print("LongCat-Image-Edit-Turbo昇腾NPU多卡并行推理")
    print("=" * 60)

    # 设置环境 - 获取多设备配置
    device_te, device_transformer = setup_environment()

    # 检查模型权重是否存在
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误：模型权重路径不存在：{args.checkpoint}")
        print("请先下载模型权重或检查路径是否正确")
        return

    # 检查输入图像是否存在
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：输入图像路径不存在：{args.input}")
        print("检查输入图像路径是否正确")
        return

    # 确保输出目录存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载模型组件（transformer加载到npu:2）
    text_processor, transformer = load_model(args.checkpoint, device_transformer)

    # 创建管道（多卡并行：分别传递各组件的目标设备）
    pipe = create_pipeline(args.checkpoint, transformer, text_processor, 
                           device_te, device_transformer)

    # 执行编辑（不再需要device参数）
    success = perform_editing(pipe, args.input, args.prompt, args.output)

    if success:
        print("\n推理完成！")
        print(f"输入提示：{args.prompt}")
        print(f"结果保存至：{args.output}")
    else:
        print("\n推理失败，请检查上述错误信息")

    print("=" * 60)

if __name__ == "__main__":
    main()
