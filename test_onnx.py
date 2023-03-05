import os
import cv2
import time
import argparse
import onnxruntime
import numpy as np
import albumentations

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='onnx推理')
parser.add_argument('--model_path', default='best.onnx', type=str, help='|onnx模型位置|')
parser.add_argument('--image_path', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=640, type=int, help='|模型输入图片大小，要与导出的模型对应|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量，要与导出的模型对应|')
parser.add_argument('--device', default='cuda', type=str, help='|用CPU/GPU推理|')
parser.add_argument('--float16', default=True, type=bool, help='|推理数据类型，要与导出的模型对应，False时为float32|')
args = parser.parse_args()
args.model_path = args.model_path.split('.')[0] + '.onnx'
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'没有找到模型{args.model_path}'
assert os.path.exists(args.image_path), f'没有找到图片文件夹{args.image_path}'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def test_onnx():
    # 加载模型
    provider = 'CUDAExecutionProvider' if args.device.lower() in ['gpu', 'cuda'] else 'CPUExecutionProvider'
    session = onnxruntime.InferenceSession(args.model_path, providers=[provider])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f'| 模型加载成功:{args.model_path} |')
    # 加载数据
    transform = albumentations.Compose([
        albumentations.LongestMaxSize(args.input_size),
        albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                   border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))])
    start_time = time.time()
    image_dir = sorted(os.listdir(args.image_path))
    image_all = np.zeros((len(image_dir), args.input_size, args.input_size, 3)).astype(
        np.float16 if args.float16 else np.float32)
    for i in range(len(image_dir)):
        image = cv2.imread(args.image_path + '/' + image_dir[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = transform(image=image)['image']  # 缩放和填充图片(归一化、减均值、除以方差、调维度等在模型中完成)
        image_all[i] = image
    end_time = time.time()
    print('| 数据加载成功:{} 每张耗时:{:.4f} |'.format(len(image_all), (end_time - start_time) / len(image_all)))
    # 推理
    start_time = time.time()
    result = []
    n = len(image_dir) // args.batch
    if n > 0:  # 如果图片数量>=批量
        for i in range(n):
            batch = image_all[i * args.batch:(i + 1) * args.batch]
            pred = session.run([output_name], {input_name: batch})
            result.extend(pred)
        if len(image_dir) % args.batch > 0:  # 如果图片数量没有刚好满足批量
            batch = image_all[(i + 1) * args.batch:]
            pred = session.run([output_name], {input_name: batch})
            result.extend(pred)
    else:  # 如果图片数量<批量
        batch = image_all
        pred = session.run([output_name], {input_name: batch})
        result.extend(pred)
    for i in range(len(result)):
        result[i] = [round(_.item(), 4) for _ in result[i]]
        print(f'| {image_dir[i]}:{result[i]} |')
    end_time = time.time()
    print('| 数据:{} 批量:{} 每张耗时:{:.4f} |'.format(len(image_all), args.batch, (end_time - start_time) / len(image_all)))


if __name__ == '__main__':
    test_onnx()
