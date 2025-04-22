import os
import cv2
import torch
import argparse
import numpy as np
import albumentations
from model.layer import deploy

# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|模型推理|')
parser.add_argument('--model_path', default='best.pt', type=str, help='|模型位置|')
parser.add_argument('--image_dir', default='image', type=str, help='|图片文件夹位置|')
parser.add_argument('--input_size', default=320, type=int, help='|模型输入图片大小|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--float16', default=True, type=bool, help='|数据类型|')
args, _ = parser.parse_known_args()  # 防止传入参数冲突，替代args = parser.parse_args()
args.device = 'cpu' if not torch.cuda.is_available() else args.device
args.float16 = True if torch.cuda.is_available() else False  # cpu使用float16会自动转为float32，但速度会很慢


# -------------------------------------------------------------------------------------------------------------------- #
class predict_class:
    def __init__(self, args=args):
        self.args = args
        self.device = args.device
        self.float16 = args.float16
        model_dict = torch.load(args.model_path, map_location='cpu', weights_only=False)
        model = deploy(model_dict['model'])
        self.model = model.half().eval().to(args.device) if args.float16 else model.float().eval().to(args.device)
        self.transform = albumentations.Compose([
            albumentations.LongestMaxSize(args.input_size),
            albumentations.PadIfNeeded(min_height=args.input_size, min_width=args.input_size,
                                       border_mode=cv2.BORDER_CONSTANT, value=(128, 128, 128))])

    def predict(self, image_dir=args.image_dir):
        image_name_list = sorted(os.listdir(image_dir))
        image_path_list = [f'{image_dir}/{_}' for _ in image_name_list]
        result = []
        for path in image_path_list:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 读取图片
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB通道
            tensor = torch.tensor(self.image_process(image)).to(self.device)
            with torch.no_grad():
                output = self.model(tensor).detach().cpu().numpy()[0]
            result.append(output)
        result = np.round(result, 2)
        return result

    def image_process(self, image):
        image = self.transform(image=image)['image']  # 缩放和填充图片
        image = image.astype(dtype=np.float16 if self.float16 else np.float32)
        image = image / 255
        image = image[np.newaxis].transpose(0, 3, 1, 2)
        return image


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    model = predict_class()
    result = model.predict()
    print(result)
