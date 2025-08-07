import os
import cv2
import argparse
import onnxruntime
import numpy as np

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|模型预测|')
parser.add_argument('--input_size', default=224, type=int, help='|模型输入图片大小|')
parser.add_argument('--device', default='cpu', type=str, help='|设备|')
parser.add_argument('--float16', default=True, type=bool, help='|数据类型|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()


# -------------------------------------------------------------------------------------------------------------------- #
class predict_class:
    def __init__(self, model_path='best.onnx', args=args):
        self.args = args
        self.device = args.device
        self.float16 = args.float16
        self.input_size = args.input_size
        provider = 'CUDAExecutionProvider' if args.device.lower() in ['gpu', 'cuda'] else 'CPUExecutionProvider'
        self.model = onnxruntime.InferenceSession(model_path, providers=[provider])  # 加载模型和框架
        self.input_name = self.model.get_inputs()[0].name  # 获取输入名称
        self.output_name = self.model.get_outputs()[0].name  # 获取输出名称

    def __call__(self, image):
        array = self.image_process(image)
        output = self.model.run([self.output_name], {self.input_name: array})[0][0]
        output = np.round(output, 2)
        return output

    def image_process(self, image):
        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = image.astype(dtype=np.float16 if self.float16 else np.float32)
        image = image / 255
        image = image[np.newaxis].transpose(0, 3, 1, 2)
        return image


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    image_path = 'img.png'
    model = predict_class(model_path='best.onnx')
    result = model(image_path)
    print(result)
