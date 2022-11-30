import cv2
import argparse
import onnxruntime

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_path', default='efficientnetv2_s.onnx', type=str, help='|onnx模型位置位置|')
parser.add_argument('--input_size', default=320, type=int, help='|onnx模型位置位置|')
parser.add_argument('--image_path', default='001.jpg', type=str, help='|onnx模型位置位置|')
args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
# 程序
def inference_onnx():
    image = cv2.imread(args.image_path)
    data = cv2.dnn.blobFromImage(image, size=(args.input_size, args.input_size), swapRB=True, scalefactor=1 / 255)
    session = onnxruntime.InferenceSession(args.model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    pred = session.run([output_name], {input_name: data})
    1


if __name__ == '__main__':
    inference_onnx()
