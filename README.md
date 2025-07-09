## pytorch图片分类训练框架
### 1，环境
>torch: https://pytorch.org/get-started/previous-versions/
>```
>pip install timm tqdm wandb opencv-python albumentations -i https://pypi.tuna.tsinghua.edu.cn/simple  
>pip install onnx onnxsim onnxruntime -i https://pypi.tuna.tsinghua.edu.cn/simple
>```
### 2，数据格式
>├── 数据集路径: data_path  
>&emsp; &emsp; └── image: 存放图片(以标签中的图片路径为准)  
>&emsp; &emsp; └── train.txt: 训练图片的标签。相对路径和类别号(如: image/0.jpg 0 2)，类别号可以为空  
>&emsp; &emsp; └── val.txt: 验证图片的标签
### 3，run.py
>模型训练，argparse中有每个参数的说明
### 4，predict.py
>模型预测
### 5，export_onnx.py
>onnx模型导出
### 6，predict_onnx.py
>onnx模型预测
### 其他
>github链接: https://github.com/TWK2022/classification  
>学习笔记: https://github.com/TWK2022/notebook  
>邮箱: 1024565378@qq.com
