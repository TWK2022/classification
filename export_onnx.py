import onnx
import torch
import onnxsim
import argparse
from model.layer import deploy

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|模型转为onnx|')
parser.add_argument('--weight', default='best.pt', type=str, help='|模型位置|')
parser.add_argument('--input_size', default=320, type=int, help='|输入图片大小|')
parser.add_argument('--batch', default=1, type=int, help='|输入图片批量，0为动态|')
parser.add_argument('--sim', default=True, type=bool, help='|使用onnxsim压缩简化模型|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--float16', default=True, type=bool, help='|数据类型|')
args = parser.parse_args()
args.weight = args.weight.split('.')[0] + '.pt'
args.save_path = args.weight.split('.')[0] + '.onnx'
args.device = 'cpu' if not torch.cuda.is_available() else args.device


# -------------------------------------------------------------------------------------------------------------------- #
def export_onnx(args=args):
    model_dict = torch.load(args.weight, map_location='cpu', weights_only=False)
    model = deploy(model_dict['model'])
    model = model.eval().half().to(args.device) if args.float16 else model.eval().float().to(args.device)
    input_one = torch.rand(1, 3, args.input_size, args.input_size,
                           dtype=torch.float16 if args.float16 else torch.float32).to(args.device)
    torch.onnx.export(model, input_one, args.save_path, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {args.batch: 'batch_size'}, 'output': {args.batch: 'batch_size'}})
    print(f'| onnx模型转换成功:{args.save_path} |')
    if args.sim:
        model_onnx = onnx.load(args.save_path)
        model_simplify, check = onnxsim.simplify(model_onnx)
        onnx.save(model_simplify, args.save_path)
        print(f'| onnxsim简化模型成功:{args.save_path} |')


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    export_onnx()
