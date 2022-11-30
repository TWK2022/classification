import argparse
import onnxruntime

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default='efficientnetv2_s.onnx', type=str, help='|onnx模型位置|')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
session = onnxruntime.InferenceSession(args.model)
input_name = session.get_inputs()
output_name = session.get_outputs()

1
