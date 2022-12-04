# 数据需准备成以下格式
# ├── 数据集路径：data_path
#     └── image：存放所有图片
#     └── train.csv：存放训练图片的相对路径和类别号，如->image\mask\0.jpg,1
#     └── val.csv：存放验证图片的类别和绝对路径
#     └── class.csv：存放所有的类别名称
# class.csv内容如下:
# 类别1
# 类别2
# ...
# -------------------------------------------------------------------------------------------------------------------- #
import os
import torch
import argparse
from block.data_get import data_get
from block.model_get import model_get
from block.loss_get import loss_get
from block.train_get import train_get

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
parser = argparse.ArgumentParser(description='分类任务')
parser.add_argument('--data_path', default=r'D:\dataset\classification\mask', type=str, help='|数据根目录路径|')
parser.add_argument('--weight', default='best.pt', type=str, help='|已有模型的位置，如果没找到模型则会创建新模型|')
parser.add_argument('--save_name', default='best.pt', type=str, help='|保存模型的位置|')
parser.add_argument('--wandb', default=False, type=bool, help='|是否使用wandb可视化|')
parser.add_argument('--wandb_project', default='mask', type=str, help='|wandb项目名称|')
parser.add_argument('--wandb_name', default='train', type=str, help='|wandb项目中的训练名称|')
parser.add_argument('--timm', default=False, type=bool, help='|是否使用timm模型|')
parser.add_argument('--model', default='cls', type=str, help='|模型选择，timm为True时为timm中的模型|')
parser.add_argument('--model_type', default='s', type=str, help='|模型型号参数，部分模型有|')
parser.add_argument('--input_size', default=160, type=int, help='|输入图片大小|')
parser.add_argument('--input_dim', default=3, type=int, help='|输入图片维度|')
parser.add_argument('--output_class', default=2, type=int, help='|输出分类类别数(独热编码)|')
parser.add_argument('--epoch', default=25, type=int, help='|训练轮数|')
parser.add_argument('--batch', default=16, type=int, help='|训练批量大小|')
parser.add_argument('--loss', default='bce', type=str, help='|损失函数|')
parser.add_argument('--lr', default=0.002, type=int, help='|初始学习率，训练中采用adam算法|')
parser.add_argument('--device', default='cuda', type=str, help='|训练设备|')
parser.add_argument('--latch', default=False, type=bool, help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--only_test', default=False, type=bool, help='|只测试模型|')
parser.add_argument('--bgr_mean', default=(0.485, 0.456, 0.406), type=tuple, help='|图片预处理时BGR通道减去的均值|')
args = parser.parse_args()
args.weight = args.weight.split('.')[0] + '.pt'
args.save_name = args.save_name.split('.')[0] + '.pt'
print('| args:{} |'.format(args))
# 为CPU设置随机种子
torch.manual_seed(999)
# 为所有GPU设置随机种子
torch.cuda.manual_seed_all(999)
# 固定每次返回的卷积算法
torch.backends.cudnn.deterministic = True
# cuDNN使用非确定性算法
torch.backends.cudnn.enabled = True
# 训练前cuDNN会先搜寻每个卷积层最适合实现它的卷积算法，加速运行；但对于复杂变化的输入数据，可能会有过长的搜寻时间，对于训练比较快的网络建议设为False
torch.backends.cudnn.benchmark = False
# wandb可视化:https://wandb.ai
if args.wandb:
    import wandb

    args.wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
assert os.path.exists(args.data_path + '/' + 'image'), 'data_path中缺少image'
assert os.path.exists(args.data_path + '/' + 'train.csv'), 'data_path中缺少train.csv'
assert os.path.exists(args.data_path + '/' + 'val.csv'), 'data_path中缺少val.csv'
assert os.path.exists(args.data_path + '/' + 'class.csv'), 'data_path中缺少class.csv'
if os.path.exists(args.weight):
    print('| 加载已有模型:{} |'.format(args.weight))
elif args.timm:
    import timm

    assert timm.list_models(args.model) != [], 'timm中没有此模型{}'.format(args.model)
    print('| 使用timm创建模型:{} |'.format(args.model))
else:
    assert os.path.exists('model/' + args.model + '.py'), '没有此自定义模型'.format(args.model)
    print('| 创建自定义模型:{} | 型号为:{}|'.format(args.model, args.model_type))
if args.device.lower() in ['cuda', 'gpu']:
    assert torch.cuda.is_available(), 'GPU不可用'
    args.device = 'cuda'
else:
    args.device = 'cpu'
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    # 数据(已预处理)
    dataset_dict = data_get(args)
    # 模型
    model_dict = model_get(args)
    # 损失
    loss = loss_get(args)
    print('| 训练集:{} | 验证集:{} | 模型:{} | 损失函数:{} |'
          .format(len(dataset_dict['train']), len(dataset_dict['val']), args.model, args.loss))
    # 训练(包括训练、验证、保存模型)
    model_dict = train_get(args, dataset_dict, model_dict, loss)
