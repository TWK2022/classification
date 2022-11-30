import torch
import argparse
from models import inference

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='')
parser.add_argument('--weight', default='efficientnetv2_s.pt', type=str, help='|模型位置|')
parser.add_argument('--input_size', default=320, type=int, help='|输入图片大小|')
parser.add_argument('--batch', default=0, type=int, help='|输入图片批量，0为动态|')
parser.add_argument('--dtype', default='float32', type=str, help='|推理数据类型|')
parser.add_argument('--device', default='cpu', type=str, help='|设备:cpu/cuda|')
parser.add_argument('--sim', default=True, type=bool, help='|使用onnxsim压缩简化模型|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
if args.dtype == 'float16':
    assert args.device not in ['gpu', 'cuda'], 'cpu上无法用float16只能用float32'
if args.device in ['gpu', 'cuda']:
    assert torch.cuda.is_available(), '选择了cuda设备但cuda不可用'


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def export():
    model_dict = torch.load(args.weight, map_location='cpu')
    model = model_dict['model']
    model.half().eval() if args.dtype == 'float16' else model.float().eval()
    model.to('cpu' if args.device == 'cpu' else 'cuda')
    input_shape = torch.randn(1, 3, args.input_size, args.input_size,
                              dtype=torch.float16 if args.dtype == 'float16' else torch.float32)
    torch.onnx.export(model, input_shape, args.weight.split('.')[0] + '.onnx',
                      opset_version=12, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {args.batch: 'batch_size'}, 'output': {args.batch: 'batch_size'}})
    print('| 转为onnx模型成功:{} |'.format(args.weight.split('.')[0] + '.onnx'))
    if args.sim:
        import onnx
        import onnxsim

        model = onnx.load(args.weight.split('.')[0] + '.onnx')
        model_simplify, check = onnxsim.simplify(model)
        onnx.save(model_simplify, args.weight.split('.')[0] + '.onnx')
        print('| 使用onnxsim简化模型成功 |')


if __name__ == '__main__':
    export()
