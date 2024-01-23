# 数据需准备成以下格式
# ├── 数据集路径：data_path
#     └── image：存放所有图片
#     └── train.txt：训练图片的绝对路径(或相对data_path下路径)和类别号，(image/mask/0.jpg 0 2\n)表示该图片类别为0和2，空类别图片无类别号
#     └── val.txt：验证图片的绝对路径(或相对data_path下路径)和类别
#     └── class.txt：所有的类别名称
# class.csv内容如下：
# 类别1
# 类别2
# ...
# -------------------------------------------------------------------------------------------------------------------- #
# 分布式训练：
# python -m torch.distributed.launch --master_port 9999 --nproc_per_node n run.py --distributed True
# master_port为GPU之间的通讯端口，空闲的即可
# n为GPU数量
# -------------------------------------------------------------------------------------------------------------------- #
import os
import wandb
import torch
import argparse
from block.data_get import data_get
from block.loss_get import loss_get
from block.model_get import model_get
from block.train_get import train_get

# -------------------------------------------------------------------------------------------------------------------- #
# 设置
# 模型加载/创建的优先级为：加载已有模型>创建剪枝模型>创建timm库模型>创建自定义模型
parser = argparse.ArgumentParser(description='|图片分类|')
parser.add_argument('--wandb', default=False, type=bool, help='|是否使用wandb可视化|')
parser.add_argument('--wandb_project', default='classification', type=str, help='|wandb项目名称|')
parser.add_argument('--wandb_name', default='train', type=str, help='|wandb项目中的训练名称|')
parser.add_argument('--wandb_image_num', default=16, type=int, help='|wandb保存图片的数量|')
parser.add_argument('--data_path', default=r'D:\dataset\classification\mask', type=str, help='|数据目录|')
parser.add_argument('--input_size', default=320, type=int, help='|输入图片大小|')
parser.add_argument('--output_class', default=1, type=int, help='|输出的类别数|')
parser.add_argument('--weight', default='last.pt', type=str, help='|已有模型的位置，没找到模型会创建剪枝/新模型|')
parser.add_argument('--prune', default=False, type=bool, help='|模型剪枝后再训练(部分模型有)，需要提供prune_weight|')
parser.add_argument('--prune_weight', default='best.pt', type=str, help='|模型剪枝的参考模型，会创建剪枝模型和训练模型|')
parser.add_argument('--prune_ratio', default=0.5, type=float, help='|模型剪枝时的保留比例|')
parser.add_argument('--prune_save', default='prune_best.pt', type=str, help='|保存最佳模型，每轮还会保存prune_last.pt|')
parser.add_argument('--timm', default=False, type=bool, help='|是否使用timm库创建模型|')
parser.add_argument('--model', default='yolov7_cls', type=str, help='|自定义模型选择，timm为True时为timm库中模型|')
parser.add_argument('--model_type', default='s', type=str, help='|自定义模型型号|')
parser.add_argument('--save_path', default='best.pt', type=str, help='|保存最佳模型，除此之外每轮还会保存last.pt|')
parser.add_argument('--epoch', default=120, type=int, help='|训练轮数|')
parser.add_argument('--batch', default=8, type=int, help='|训练批量大小，分布式时为总批量|')
parser.add_argument('--loss', default='bce', type=str, help='|损失函数|')
parser.add_argument('--lr_start', default=0.001, type=float, help='|初始学习率，adam算法，3轮预热训练，基准为0.001|')
parser.add_argument('--lr_end_ratio', default=0.1, type=float, help='|最终学习率=lr_end_ratio*lr_start，基准为0.1|')
parser.add_argument('--lr_adjust_num', default=50, type=int, help='|学习率下降调整次数，余玄下降法，要小于总轮次|')
parser.add_argument('--lr_adjust_threshold', default=0.9, type=float, help='|损失下降比较快时不调整学习率，基准为0.9|')
parser.add_argument('--regularization', default='L2', type=str, help='|正则化，有L2、None|')
parser.add_argument('--r_value', default=0.0005, type=float, help='|正则化权重系数，基准为0.0005|')
parser.add_argument('--device', default='cuda', type=str, help='|训练设备|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存，True为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|CPU处理数据的进程数，0表示只有一个主进程，一般为0、2、4、8|')
parser.add_argument('--ema', default=True, type=bool, help='|使用平均指数移动(EMA)调整参数|')
parser.add_argument('--amp', default=False, type=bool, help='|混合float16精度训练，CPU时不可用|')
parser.add_argument('--noise', default=0.5, type=float, help='|训练数据加噪概率|')
parser.add_argument('--class_threshold', default=0.5, type=float, help='|计算指标时，大于阈值判定为图片有该类别|')
parser.add_argument('--distributed', default=False, type=bool, help='|单机多卡分布式训练，分布式训练时batch为总batch|')
parser.add_argument('--local_rank', default=0, type=int, help='|分布式训练使用命令后会自动传入的参数|')
args = parser.parse_args()
args.device_number = max(torch.cuda.device_count(), 1)  # 使用的GPU数，可能为CPU
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
if args.wandb and args.local_rank == 0:  # 分布式时只记录一次wandb
    args.wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
# 混合float16精度训练
if args.amp:
    args.amp = torch.cuda.amp.GradScaler()
# 分布式训练
if args.distributed:
    torch.distributed.init_process_group(backend="nccl")  # 分布式训练初始化
    args.device = torch.device("cuda", args.local_rank)
# -------------------------------------------------------------------------------------------------------------------- #
# 初步检查
if args.local_rank == 0:
    print(f'| args:{args} |')
    assert os.path.exists(f'{args.data_path}/image'), '! data_path中缺少:image !'
    assert os.path.exists(f'{args.data_path}/train.txt'), '! data_path中缺少:train.txt !'
    assert os.path.exists(f'{args.data_path}/val.txt'), '! data_path中缺少:val.txt !'
    assert os.path.exists(f'{args.data_path}/class.txt'), '! data_path中缺少:class.txt !'
    if os.path.exists(args.weight):  # 优先加载已有模型args.weight继续训练
        print(f'| 加载已有模型:{args.weight} |')
    elif args.prune:
        print(f'| 加载模型+剪枝训练:{args.prune_weight} |')
    elif args.timm:  # 创建timm库中模型args.timm
        import timm

        assert timm.list_models(args.model), f'! timm中没有模型:{args.model}，使用timm.list_models()查看所有模型 !'
        print(f'| 创建timm库中模型:{args.model} |')
    else:  # 创建自定义模型args.model
        assert os.path.exists(f'model/{args.model}.py'), f'! 没有自定义模型:{args.model} !'
        print(f'| 创建自定义模型:{args.model} | 型号:{args.model_type} |')
# -------------------------------------------------------------------------------------------------------------------- #
# 程序
if __name__ == '__main__':
    # 数据(只是图片路径和标签，读取和预处理在训练/验证中完成)
    data_dict = data_get(args)
    # 模型
    model_dict = model_get(args)
    # 损失
    loss = loss_get(args)
    # 摘要
    print('| 训练集:{} | 验证集:{} | 批量{} | 模型:{} | 输入尺寸:{} | 损失函数:{} | 初始学习率:{} |'
          .format(len(data_dict['train']), len(data_dict['val']), args.batch, args.model, args.input_size, args.loss,
                  args.lr_start)) if args.local_rank == 0 else None
    # 训练
    train_get(args, data_dict, model_dict, loss)
