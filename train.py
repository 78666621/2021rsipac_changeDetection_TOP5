import os
import numpy as np
import torch
from utils import losses,train,metrics
from torch.utils.data import DataLoader
from datasets.change_convert import Change_Convert
from pytorch_toolbelt import losses as L
import random
from unetplusplus.model import UnetPlusPlus

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_data(DATA_DIR_train,DATA_DIR_via,model_path='',
          save_path='./best_model.pth',epochs=40,train_txt='train.txt',
          val_txt='val.txt',max_score=0):
    #  数据集所在的目录

    ENCODER = 'efficientnet-b0'
    # ENCODER = 'efficientnet-b1'
    ENCODER_WEIGHTS = 'imagenet'
    ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multiclass segmentation
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 使用unet++模型
    model = UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1,
        activation=ACTIVATION,
        in_channels=6
    )

    train_dataset = Change_Convert(DATA_DIR_train,
                                    sub_dir_1='A',
                                    sub_dir_2='B',
                                    img_suffix='.tif',
                                    ann_dir=DATA_DIR_train + '/label',
                                    size=512,
                                    debug=False,
                                    split=train_txt)

    valid_dataset = Change_Convert(DATA_DIR_via,
                                    sub_dir_1='A',
                                    sub_dir_2='B',
                                    img_suffix='.tif',
                                    ann_dir=DATA_DIR_via + '/label',
                                    size=512,
                                    debug=False,
                                    test_mode=True,
                                    split=val_txt)

    # 需根据显卡的性能进行设置，batch_size为每次迭代中一次训练的图片数，num_workers为训练时的工作进程数，如果显卡不太行或者显存空间不够，将batch_size调低并将num_workers调为0
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=2)

    DiceLoss = losses.DiceLoss()
    SoftBCEWithLogitsLoss = losses.BCEWithLogitsLoss(pos_weight=torch.tensor([30]))
    loss = L.JointLoss(first=DiceLoss, second=SoftBCEWithLogitsLoss,
                       first_weight=1.0, second_weight=1.0).cuda()


    metricss = [
        metrics.Fscore(threshold=0.5),
        metrics.Precision(threshold=0.5),
        metrics.Recall(threshold=0.5),
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=2,  # T_0就是初始restart的epoch数目
        T_mult=2,  # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
        eta_min=1e-5  # 最低学习率
    )

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, dampening=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True, min_lr=0.0000001)

    # 创建一个简单的循环，用于迭代数据样本
    train_epoch = train.TrainEpoch(
        model,
        loss=loss,
        metrics=metricss,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = train.ValidEpoch(
        model,
        loss=loss,
        metrics=metricss,
        device=DEVICE,
        verbose=True,
    )

    val_f1 = 0

    for i in range(1, epochs+1):

        print('\nEpoch: {}'.format(i))
        print('lr', optimizer.param_groups[0]['lr'])
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        scheduler.step()
        val_precision = valid_logs['precision']
        val_recall = valid_logs['recall']
        val_f1 = (2 * val_precision * val_recall) / (val_precision + val_recall)


        # 每次迭代保存下训练最好的模型
        if max_score < val_f1:
            max_score = val_f1
            torch.save(model, save_path)
            print('val_f1:', val_f1)
            print('Model saved!')

    # return val_f1


# 创建模型并训练
# ---------------------------------------------------------------
if __name__ == '__main__':
    train_dir = '/media/liulin/MyPassport1/competition/初赛审核/wuda_data/train' # 训练集
    val_dir = '/media/liulin/MyPassport1/competition/初赛审核/wuda_data/train'  # 验证集
    train_data(train_dir, val_dir,save_path='/save_dir/b1-llk0.pth',
          epochs=150,train_txt='txt/k0_train.txt',val_txt='txt/k0.txt')
