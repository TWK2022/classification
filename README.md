## pytorch图片分类训练代码
>包含onnx、trt导出及预测  
>代码兼容性较强，使用的是一些基本的库、基础的函数  
>在argparse中可以选择使用wandb，能在wandb网站中生成可视化的训练过程  
### 数据格式如下  
>├── 数据集路径：data_path  
>&emsp; &emsp; └── image：存放所有图片  
>&emsp; &emsp; └── train.txt：训练图片的绝对路径(或相对data_path下路径)和类别号，  
>&emsp; &emsp; &emsp; &emsp; (如-->image/mask/0.jpg 0 2<--表示该图片类别为0和2，空类别图片无类别号)  
>&emsp; &emsp; └── val.txt：验证图片的绝对路径(或相对data_path下路径)和类别  
>&emsp; &emsp; └── class.txt：所有的类别名称  
### 1，run.py
>模型训练时运行该文件，argparse中有对每个参数的说明
### 2，test_pt.py
>使用训练好的pt模型预测
### 3，export_onnx.py
>将pt模型导出为onnx模型
### 4，test_onnx.py
>使用导出的onnx模型预测
### 5，export_trt_record.txt
>文档中有onnx模型导出为tensort模型的详细说明
### 6，test_trt.py
>使用导出的trt模型预测