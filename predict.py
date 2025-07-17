import os
import cv2
import torch
import argparse
import numpy as np
from model.layer import deploy

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|模型预测|')
parser.add_argument('--input_size', default=224, type=int, help='|模型输入图片大小|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
args.device = 'cpu' if not torch.cuda.is_available() else args.device


# -------------------------------------------------------------------------------------------------------------------- #
class predict_class:
    def __init__(self, model_path, args=args):
        self.device = args.device
        self.input_size = args.input_size
        model_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        model = deploy(model_dict['model'])
        self.model = model.float().eval().to(args.device)

    def __call__(self, image):
        with torch.no_grad():
            tensor = torch.tensor(self.image_process(image)).to(self.device)
            output = self.model(tensor).detach().cpu().numpy()[0]
            output = np.round(output, 2)
        return output

    def image_process(self, image):
        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = image.astype(dtype=np.float32)
        image = image / 255
        image = image[np.newaxis].transpose(0, 3, 1, 2)
        return image


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    image_path = 'img.png'
    model = predict_class(model_path='best.pt')
    result = model(image_path)
    print(result)
