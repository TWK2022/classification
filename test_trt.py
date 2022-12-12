import os
import cv2
import time
import torch
import argparse
import tensorrt
import albumentations

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='tensorrt推理')
parser.add_argument('--model_path', default='best.trt', type=str, help='|trt模型位置|')
parser.add_argument('--image_path', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=160, type=int, help='|输入图片大小，trt模型构建时确定的|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量，trt模型构建时确定的，一般为1|')
parser.add_argument('--rgb_mean', default=(0.406, 0.456, 0.485), type=tuple, help='|图片预处理时RGB通道减去的均值|')
parser.add_argument('--rgb_std', default=(0.225, 0.224, 0.229), type=tuple, help='|图片预处理时RGB通道除以的方差|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.model_path), f'没有找到模型{args.model_path}'
assert os.path.exists(args.image_path), f'没有找到图片文件夹{args.image_path}'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序

def trt_version():
    return tensorrt.__version__


def torch_device_from_trt(device):
    if device == tensorrt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == tensorrt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)


def torch_dtype_from_trt(dtype):
    if dtype == tensorrt.int8:
        return torch.int8
    elif tensorrt_version() >= '7.0' and dtype == tensorrt.bool:
        return torch.bool
    elif dtype == tensorrt.int32:
        return torch.int32
    elif dtype == tensorrt.float16:
        return torch.float16
    elif dtype == tensorrt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None):
        super(TRTModule, self).__init__()
        self.engine = engine
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


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.Normalize(max_pixel_value=255, mean=args.rgb_mean, std=args.rgb_std),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0))])

    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, index):
        image = cv2.imread(args.image_path + '/' + self.image_dir[index])  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = self.transform(image=image)['image']  # 归一化、减均值、除以方差
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # 转换为tensor
        return image


logger = tensorrt.Logger(tensorrt.Logger.INFO)  # 忽略INFO信息
with tensorrt.Runtime(logger) as runtime, open(args.model_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())  # 读取模型并构建一个对象
context = engine.create_execution_context()

for binding in engine:
    is_input = engine.binding_is_input(binding)
    shape = engine.get_binding_shape(binding)
    op_type = tensorrt.nptype(engine.get_binding_dtype(binding))
    1

# trt_model = TRTModule(engine, ["input"], ["output"])
start_time = time.time()
image_dir = sorted(os.listdir(args.image_path))
dataloader = torch.utils.data.DataLoader(torch_dataset(image_dir), batch_size=args.batch,
                                         shuffle=False, drop_last=False, pin_memory=False)
for item, image in enumerate(dataloader):
    start_inference = time.time()
    pred = torch.zeros(1, 3, args.input_size, args.input_size)
    context.execute_async(batch_size=1, bindings=pred)
    end_inference = time.time()
1
