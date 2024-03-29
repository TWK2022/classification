import os
import cv2
import time
import argparse
import onnxruntime
import numpy as np
import albumentations

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|onnx模型推理|')
parser.add_argument('--model_path', default='best.onnx', type=str, help='|onnx模型位置|')
parser.add_argument('--data_path', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=320, type=int, help='|模型输入图片大小，要与导出的模型对应|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量，要与导出的模型对应|')
parser.add_argument('--device', default='cuda', type=str, help='|推理设备|')
parser.add_argument('--float16', default=True, type=bool, help='|推理数据类型，要与导出的模型对应，False时为float32|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
assert os.path.exists(args.model_path), f'! model_path不存在:{args.model_path} !'
assert os.path.exists(args.data_path), f'! data_path不存在:{args.data_path} !'


# -------------------------------------------------------------------------------------------------------------------- #
def predict_onnx(args):
    # 加载模型
    provider = 'CUDAExecutionProvider' if args.device.lower() in ['gpu', 'cuda'] else 'CPUExecutionProvider'
    model = onnxruntime.InferenceSession(args.model_path, providers=[provider])  # 加载模型和框架
    input_name = model.get_inputs()[0].name  # 获取输入名称
    output_name = model.get_outputs()[0].name  # 获取输出名称
    print(f'| 模型加载成功:{args.model_path} |')
    # 加载数据
    start_time = time.time()
    transform = albumentations.Compose([
        albumentations.LongestMaxSize(args.input_size),
        albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                   border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128))])
    image_dir = sorted(os.listdir(args.data_path))
    image_all = np.zeros((len(image_dir), args.input_size, args.input_size, 3)).astype(
        np.float16 if args.float16 else np.float32)
    for i in range(len(image_dir)):
        image = cv2.imread(args.data_path + '/' + image_dir[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = transform(image=image)['image']  # 缩放和填充图片(归一化、减均值、除以方差、调维度等在模型中完成)
        image_all[i] = image
    end_time = time.time()
    print('| 数据加载成功:{} 每张耗时:{:.4f} |'.format(len(image_all), (end_time - start_time) / len(image_all)))
    # 推理
    start_time = time.time()
    result = []
    n = len(image_all) // args.batch
    if n > 0:  # 如果图片数量>=批量(分批预测)
        for i in range(n):
            batch = image_all[i * args.batch:(i + 1) * args.batch]
            pred_batch = model.run([output_name], {input_name: batch})
            result.extend(pred_batch[0].tolist())
        if len(image_all) % args.batch > 0:  # 如果图片数量没有刚好满足批量
            batch = image_all[(i + 1) * args.batch:]
            pred_batch = model.run([output_name], {input_name: batch})
            result.extend(pred_batch[0].tolist())
    else:  # 如果图片数量<批量(直接预测)
        batch = image_all
        pred_batch = model.run([output_name], {input_name: batch})
        result.extend(pred_batch[0].tolist())
    for i in range(len(result)):
        result[i] = [round(result[i][_], 2) for _ in range(len(result[i]))]
        print(f'| {image_dir[i]}:{result[i]} |')
    end_time = time.time()
    print('| 数据:{} 批量:{} 每张耗时:{:.4f} |'.format(len(image_all), args.batch, (end_time - start_time) / len(image_all)))


if __name__ == '__main__':
    predict_onnx(args)
