import wandb
import torch
import argparse
from train_class import train_class

# -------------------------------------------------------------------------------------------------------------------- #
# 数据格式
# ├── 数据集路径: data_path
#     └── train.txt: 训练图片的标签。相对路径和类别号(如:image/000.jpg 0 2)，类别号可以为空
#     └── val.txt: 验证图片的标签
# -------------------------------------------------------------------------------------------------------------------- #
# 分布式数据并行训练:
# python -m torch.distributed.launch --master_port 9999 --nproc_per_node n run.py --distributed True
# master_port为gpu之间的通讯端口，空闲的即可。n为gpu数量
# -------------------------------------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='|图片分类|')
parser.add_argument('--log', default=True, type=bool, help='|日志|')
parser.add_argument('--tqdm', default=True, type=bool, help='|每轮进度条|')
parser.add_argument('--print_info', default=True, type=bool, help='|打印信息|')
parser.add_argument('--wandb', default=False, type=bool, help='|wandb可视化|')
parser.add_argument('--data_path', default='dataset', type=str, help='|数据位置|')
parser.add_argument('--weight_path', default='last.pt', type=str, help='|加载模型，优先级:加载模型>剪枝训练>创建新模型|')
parser.add_argument('--weight_again', default=True, type=bool, help='|重置学习率等状态，在weight_path上重新训练|')
parser.add_argument('--prune_weight_path', default='prune_weight.pt', type=str, help='|剪枝参考模型|')
parser.add_argument('--prune_ratio', default=0.8, type=float, help='|剪枝保留比例|')
parser.add_argument('--model', default='yolov7_cls', type=str, help='|模型选择|')
parser.add_argument('--model_type', default='s', type=str, help='|模型型号|')
parser.add_argument('--save_epoch', default=5, type=int, help='|每x轮和最后一轮保存模型|')
parser.add_argument('--save_path', default='last.pt', type=str, help='|保存模型|')
parser.add_argument('--save_best', default='best.pt', type=str, help='|保存最佳模型|')
parser.add_argument('--input_size', default=224, type=int, help='|输入图片大小|')
parser.add_argument('--output_class', default=4, type=int, help='|输出类别数|')
parser.add_argument('--epoch', default=50, type=int, help='|训练总轮数(包含之前已训练轮数)|')
parser.add_argument('--batch', default=128, type=int, help='|训练批量大小，分布式时为总批量|')
parser.add_argument('--warmup_ratio', default=0.01, type=float, help='|预热训练步数占总步数比例，最少5步，基准为0.01|')
parser.add_argument('--lr_start', default=5e-4, type=float, help='|初始学习率，adam算法，批量小时要减小，基准为5e-4|')
parser.add_argument('--lr_end_ratio', default=0.01, type=float, help='|最终学习率=lr_end_ratio*lr_start，基准为0.01|')
parser.add_argument('--regularization', default='L2', type=str, help='|正则化，有L2、None|')
parser.add_argument('--r_value', default=5e-4, type=float, help='|正则化权重系数，基准为5e-4|')
parser.add_argument('--device', default='cuda', type=str, help='|设备|')
parser.add_argument('--latch', default=True, type=bool, help='|模型和数据是否为锁存|')
parser.add_argument('--num_worker', default=0, type=int, help='|cpu处理数据进程数，0为一个主进程，一般为0、8、16|')
parser.add_argument('--ema', default=True, type=bool, help='|平均指数移动(EMA)调整参数|')
parser.add_argument('--amp', default=True, type=bool, help='|混合float16精度训练，cpu时不可用，出现nan可能与gpu有关|')
parser.add_argument('--noise', default=0.8, type=float, help='|训练数据加噪概率|')
parser.add_argument('--class_threshold', default=0.5, type=float, help='|计算指标时，大于阈值判定为有该类别|')
parser.add_argument('--distributed', default=False, type=bool, help='|单机多卡分布式训练，分布式训练时batch为总batch|')
parser.add_argument('--local_rank', default=0, type=int, help='|分布式训练使用命令后会自动传入的参数|')
args = parser.parse_args()
if not torch.cuda.is_available():  # 没有gpu
    args.device = 'cpu'
    args.amp = False
args.device_number = max(torch.cuda.device_count(), 1)  # 使用的gpu数，可能为cpu
# wandb可视化:https://wandb.ai
if args.wandb and args.local_rank == 0:  # 分布式时只记录一次wandb
    args.wandb_run = wandb.init(project='classification', name='train', config=args)
# 混合float16精度训练
if args.amp:
    args.amp = torch.cuda.amp.GradScaler()
# 分布式训练
if args.distributed:
    torch.distributed.init_process_group(backend='nccl')  # 分布式训练初始化
    args.device = torch.device('cuda', args.local_rank)
# 设置
torch.manual_seed(999)  # 为cpu设置随机种子
torch.cuda.manual_seed_all(999)  # 为所有gpu设置随机种子
torch.backends.cudnn.deterministic = True  # 固定每次返回的卷积算法
torch.backends.cudnn.enabled = True  # cuDNN使用非确定性算法
torch.backends.cudnn.benchmark = False  # 训练前cuDNN会先搜寻每个卷积层最适合实现它的卷积算法，加速运行；但对于复杂变化的输入数据，可能会有过长的搜寻时间，对于训练比较快的网络建议设为False
# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    train = train_class(args)
    train.train()
