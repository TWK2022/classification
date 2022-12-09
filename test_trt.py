import os
import cv2
import time
import torch
import argparse
import tensorrt as trt
import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_path', default='best.trt', type=str, help='|trt模型位置|')
parser.add_argument('--image_path', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=160, type=int, help='|输入图片大小，trt模型构建时确定的|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量，trt模型构建时确定的，一般为1|')
parser.add_argument('--bgr_mean', default=(0.485, 0.456, 0.406), type=tuple, help='|图片预处理时BGR通道减去的均值|')
parser.add_argument('--float16', default=False, type=bool, help='|推理数据类型，要与模型相对应，False时为float32|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'没有找到模型{args.model_path}'
assert os.path.exists(args.image_path), f'没有找到图片文件夹{args.image_path}'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序

def trt_version():
    return trt.__version__


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self.engine = engine
        if self.engine is not None:
            # engine创建执行context
            self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names

def forward(self, *inputs):
    batch_size = inputs[0].shape[0]
    bindings = [None] * (len(self.input_names) + len(self.output_names))
    # 创建输出tensor，并分配内存
    outputs = [None] * len(self.output_names)
    for i, output_name in enumerate(self.output_names):
        idx = self.engine.get_binding_index(output_name)  # 通过binding_name找到对应的input_id
        dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))  # 找到对应的数据类型
        shape = (batch_size,) + tuple(self.engine.get_binding_shape(idx))  # 找到对应的形状大小
        device = torch_device_from_trt(self.engine.get_location(idx))
        output = torch.empty(size=shape, dtype=dtype, device=device)
        outputs[i] = output
        bindings[idx] = output.data_ptr()  # 绑定输出数据指针

    for i, input_name in enumerate(self.input_names):
        idx = self.engine.get_binding_index(input_name)
        bindings[idx] = inputs[0].contiguous().data_ptr()  # 应当为inputs[i]，对应3个输入。但由于我们使用的是单张图片，所以将3个输入全设置为相同的图片。

    self.context.execute_async(batch_size, bindings, torch.cuda.current_stream().cuda_stream)  # 执行推理

    outputs = tuple(outputs)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def resize(image):
    w0 = image.shape[1]
    h0 = image.shape[0]
    if w0 == h0:
        image = cv2.resize(image, (args.input_size, args.input_size))
    elif w0 > h0:  # 宽大于高
        w = args.input_size
        h = int(w / w0 * h0)
        image = cv2.resize(image, (w, h))
        add_y = (w - h) // 2
        image = cv2.copyMakeBorder(image, add_y, w - h - add_y, 0, 0, cv2.BORDER_CONSTANT, value=(126, 126, 126))
    else:  # 宽小于高
        h = args.input_size
        w = int(h / h0 * w0)
        image = cv2.resize(image, (w, h))
        add_x = (h - w) // 2
        image = cv2.copyMakeBorder(image, 0, 0, add_x, h - w - add_x, cv2.BORDER_CONSTANT, value=(126, 126, 126))
    return image


def processing(image):
    image = (image / 255).transpose(2, 0, 1).astype(np.float16 if args.float16 else np.float32)
    image[0] = image[0] - args.bgr_mean[0]
    image[1] = image[1] - args.bgr_mean[1]
    image[2] = image[2] - args.bgr_mean[2]
    return image


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        image = cv2.imread(args.image_path + '/' + image_dir[index])
        image = resize(image)
        image = processing(image)
        image = torch.tensor(image, dtype=torch.float32 if args.float16 else torch.float16).to('cuda')
        return image


logger = trt.Logger(trt.Logger.INFO)  # 忽略INFO信息
with open(args.model_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())  # 读取模型并构建一个对象
for idx in range(engine.num_bindings):  # 查看输入输出的序号，名称，形状，类型
    is_input = engine.binding_is_input(idx)
    name = engine.get_binding_name(idx)
    shape = engine.get_binding_shape(idx)
    op_type = engine.get_binding_dtype(idx)
    print('input id:', idx, ' is input: ', is_input, ' binding name:', name, ' shape:', shape, 'type: ', op_type)

trt_model = TRTModule(engine, ["input"], ["output"])

start_time = time.time()
image_dir = sorted(os.listdir(args.image_path))
dataloader = torch.utils.data.DataLoader(torch_dataset(image_dir), batch_size=args.batch,
                                         shuffle=False, drop_last=False, pin_memory=False)

for item, image in enumerate(dataloader):
    start_inference = time.time()
    a = image
    result_trt = trt_model(image)
    end_inference = time.time()
1
