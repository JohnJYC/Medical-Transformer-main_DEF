import torch
import lib
import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
from utils import JointTransform2D, ImageToImage2D, Image2D
from metrics import jaccard_index, f1_score, LogNLLLoss, classwise_f1
from utils import chk_mkdir, Logger, MetricList
import cv2
import timeit

# 导入 wandb
import wandb

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int, default=10)

parser.add_argument('--modelname', default='MedT', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str,
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./medt', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)

args = parser.parse_args()
gray_ = args.gray
aug = args.aug
direc = args.direc
modelname = args.modelname
imgsize = args.imgsize

if gray_ == "yes":
    from utils_gray import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

if modelname == "axialunet":
    model = lib.models.axialunet(img_size=imgsize, imgchan=imgchant)
elif modelname == "MedT":
    model = lib.models.axialnet.MedT(img_size=imgsize, imgchan=imgchant)
elif modelname == "gatedaxialunet":
    model = lib.models.axialnet.gated(img_size=imgsize, imgchan=imgchant)
elif modelname == "logo":
    model = lib.models.axialnet.logo(img_size=imgsize, imgchan=imgchant)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

# 使用 CrossEntropyLoss 适用于多分类任务
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                             weight_decay=args.weight_decay)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# 初始化 wandb
wandb.init(project='MedT', config={
    'epochs': args.epochs,
    'batch_size': args.batch_size,

    'learning_rate': args.learning_rate,
    'model_name': args.modelname,
    'imgsize': args.imgsize,
    'gray': args.gray,
})
# 可选：监视模型
wandb.watch(model, log='all')

for epoch in range(args.start_epoch, args.epochs):

    model.train()
    epoch_running_loss = 0

    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).squeeze(1).long()  # 标签形状调整

        # 前向传播
        output = model(X_batch)

        # 计算损失
        loss = criterion(output, y_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_running_loss += loss.item()

    # 计算平均损失
    avg_loss = epoch_running_loss / (batch_idx + 1)

    # 记录指标到 wandb
    wandb.log({'epoch': epoch + 1, 'loss': avg_loss})

    # 打印日志
    print('Epoch [{}/{}], Loss: {:.4f}'
          .format(epoch + 1, args.epochs, avg_loss))

    if (epoch + 1) % args.save_freq == 0:

        model.eval()  # 设置模型为评估模式
        total_correct = 0
        total_pixels = 0
        dice_scores = []

        with torch.no_grad():
            for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
                # 获取文件名
                if isinstance(rest[0][0], str):
                    image_filename = rest[0][0]
                else:
                    image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).squeeze(1).long()  # 标签形状调整

                y_out = model(X_batch)
                y_pred = y_out.argmax(dim=1)  # 获取预测类别索引

                correct = (y_pred == y_batch).sum().item()
                total = y_batch.numel()
                accuracy = correct / total

                total_correct += correct
                total_pixels += total

                # 计算 Dice 系数
                y_pred_np = y_pred.cpu().numpy()
                y_true_np = y_batch.cpu().numpy()
                smooth = 1e-5
                intersection = (y_pred_np * y_true_np).sum()
                dice = (2. * intersection + smooth) / (y_pred_np.sum() + y_true_np.sum() + smooth)
                dice_scores.append(dice)

                # 记录验证准确率
                wandb.log({'val_accuracy': accuracy})

                # 保存预测结果图像
                y_pred_image = (y_pred.cpu().numpy()[0] * 255).astype(np.uint8)
                fulldir = os.path.join(direc, str(epoch + 1))
                if not os.path.isdir(fulldir):
                    os.makedirs(fulldir)
                cv2.imwrite(os.path.join(fulldir, image_filename), y_pred_image)

                # 记录图像到 wandb
                wandb.log({
                    'Input Image': wandb.Image(X_batch.cpu()[0]),
                    'Prediction': wandb.Image(y_pred_image),
                    'Ground Truth': wandb.Image((y_batch.cpu().numpy()[0] * 255).astype(np.uint8)),
                })

        # 计算整个验证集的平均准确率和 Dice 系数
        avg_accuracy = total_correct / total_pixels
        avg_dice = np.mean(dice_scores)
        print(f'Validation Accuracy: {avg_accuracy:.4f}')
        print(f'Validation Dice Coefficient: {avg_dice:.4f}')
        wandb.log({'epoch': epoch + 1, 'val_avg_accuracy': avg_accuracy, 'val_dice_coefficient': avg_dice})

        model.train()  # 重新设置模型为训练模式

        # 保存模型
        fulldir = os.path.join(direc, str(epoch + 1))
        torch.save(model.state_dict(), os.path.join(fulldir, args.modelname + ".pth"))
        torch.save(model.state_dict(), os.path.join(direc, "final_model.pth"))

# 结束 wandb 运行
wandb.finish()