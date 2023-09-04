import os
import cv2
import time
import argparse
import tensorrt
import numpy as np
import albumentations
import pycuda.autoinit
import pycuda.driver as cuda

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='|tensorrt模型推理|')
parser.add_argument('--model_path', default='best.trt', type=str, help='|trt模型位置|')
parser.add_argument('--data_path', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=320, type=int, help='|输入图片大小，要与导出的模型对应|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量，要与导出的模型对应，一般为1|')
parser.add_argument('--float16', default=True, type=bool, help='|推理数据类型，要与导出的模型对应，False时为float32|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'没有找到模型{args.model_path}'
assert os.path.exists(args.data_path), f'没有找到图片文件夹{args.data_path}'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def test_tensorrt(args):
    # 加载模型
    logger = tensorrt.Logger(tensorrt.Logger.WARNING)  # 创建日志记录信息
    with tensorrt.Runtime(logger) as runtime, open(args.model_path, "rb") as f:
        model = runtime.deserialize_cuda_engine(f.read())  # 读取模型并构建一个对象
    np_type = tensorrt.nptype(model.get_tensor_dtype('input'))  # 获取接口的数据类型并转为np的字符串格式
    h_input = np.zeros(tensorrt.volume(model.get_tensor_shape('input')), dtype=np_type)  # 获取输入的形状(一维)
    h_output = np.zeros(tensorrt.volume(model.get_tensor_shape('output')), dtype=np_type)  # 获取输出的形状(一维)
    d_input = cuda.mem_alloc(h_input.nbytes)  # 分配显存空间
    d_output = cuda.mem_alloc(h_output.nbytes)  # 分配显存空间
    bindings = [int(d_input), int(d_output)]  # 绑定显存输入输出
    stream = cuda.Stream()  # 创建cuda流
    model_context = model.create_execution_context()  # 创建模型推理器
    print(f'| 加载模型成功:{args.model_path} |')
    # 加载数据
    start_time = time.time()
    transform = albumentations.Compose([
        albumentations.LongestMaxSize(args.input_size),
        albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                   border_mode=cv2.BORDER_CONSTANT, value=(127, 127, 127))])
    image_dir = sorted(os.listdir(args.data_path))
    image_list = [0 for _ in range(len(image_dir))]
    for i in range(len(image_dir)):
        image = cv2.imread(args.data_path + '/' + image_dir[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = transform(image=image)['image'].reshape(-1).astype(
            np.float16 if args.float16 else np.float32)  # 缩放和填充图片(归一化、减均值、除以方差、调维度等在模型中完成)
        image_list[i] = image
    end_time = time.time()
    print('| 数据加载成功:{} 每张耗时:{:.4f} |'.format(len(image_list), (end_time - start_time) / len(image_list)))
    # 推理
    start_time = time.time()
    result = [0 for _ in range(len(image_list))]
    for i in range(len(image_list)):
        cuda.memcpy_htod_async(d_input, image_list[i], stream)  # 将输入数据从CPU锁存复制到GPU显存
        model_context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)  # 执行推理
        cuda.memcpy_dtoh_async(h_output, d_output, stream)  # 将输出数据从GPU显存复制到CPU锁存
        stream.synchronize()  # 同步线程
        result[i] = [round(_, 2) for _ in h_output.tolist()]
        print(f'| {image_dir[i]}:{result[i]} |')
    end_time = time.time()
    print('| 数据:{} 批量:{} 每张耗时:{:.4f} |'.format(len(image_list), args.batch, (end_time - start_time) / len(image_list)))


if __name__ == '__main__':
    test_tensorrt(args)
