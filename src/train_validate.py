#!/usr/bin/env python3
import random
import argparse, os, sys, time, shutil, tqdm
import warnings, json, gzip
import numpy as np
import copy
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset

import epi_models
import epi_dataset
import misc_utils

import functools

print = functools.partial(print, flush=True)


def split_train_valid_test(groups, train_keys, valid_keys, test_keys=None):
    """
    groups: length N, the number of samples
    train
    """
    assert isinstance(train_keys, list)
    assert isinstance(valid_keys, list)
    assert test_keys is None or isinstance(test_keys, list)
    index = np.arange(len(groups))
    train_idx = index[np.isin(groups, train_keys)]
    valid_idx = index[np.isin(groups, valid_keys)]
    if test_keys is not None:
        test_idx = index[np.isin(groups, test_keys)]
        return train_idx, valid_idx, test_idx
    else:
        return train_idx, valid_idx


def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir


def model_summary(model):
    """
    model: pytorch model
    """
    import torch
    total_param = 0
    trainable_param = 0
    for i, p in enumerate(model.parameters()):
        num_p = torch.numel(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}


def predict(model: nn.Module, data_loader: DataLoader, device=torch.device('cuda')):
    model.eval()
    result, true_label = None, None
    for feats, _, enh_idxs, prom_idxs, labels in data_loader:
        feats, labels = feats.to(device), labels.to(device)
        # enh_idxs, prom_idxs = enh_idxs.to(device), prom_idxs.to(device)
        pred = model(feats, enh_idx=enh_idxs, prom_idx=prom_idxs)
        pred = pred.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        if result is None:
            result = pred
            true_label = labels
        else:
            result = np.concatenate((result, pred), axis=0)
            true_label = np.concatenate((true_label, labels), axis=0)
    return (result.squeeze(), true_label.squeeze())


def train_validate_test(
        model, optimizer,
        train_loader, valid_loader, test_loader,
        num_epoch, patience, outdir,
        checkpoint_prefix, device, use_scheduler=False) -> nn.Module:
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    wait = 0
    best_epoch, best_val_auc, best_val_aupr = -1, -1, -1

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    best_acc,  best_F1 = 0,0
    for epoch_idx in range(num_epoch):
        model.train()
        # 这段代码是一个训练过程的循环，用于训练一个深度学习模型。其中，如果这个模型有一个名为"att_C"的属性，就会进行以下的操作：
        # 1、使用模型对输入数据 feats、enh_idxs、prom_idxs进行前向传播，得到模型的输出 pred、预测的距离   pred_dists  和注意力矩阵   att
        # 2、将注意力矩阵转置并计算其与单位矩阵之差的二范数，即正则化项 penal。
        # 3、计算损失函数 loss，包括二分类交叉熵损失 bce_loss、正则化项 penal 以及平方误差损失 mse_loss(dists, pred_dists)，其中 model.att_C 表示正则化项的系数。
        # 4、删除正则化项 penal 和单位矩阵 identity。
        # 5、如果模型没有 "att_C" 属性，则只进行前向传播并计算二分类交叉熵损失 bce_loss。
        train_true, train_pred  = None, None
        for feats, dists, enh_idxs, prom_idxs, labels in tqdm.tqdm(train_loader):
            feats, dists, labels = feats.to(device), dists.to(device), labels.to(device)
            if hasattr(model, "att_C"):
                pred, pred_dists, att = model(feats, enh_idxs, prom_idxs, return_att=True)
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1)).to(device)
                identity = Variable(identity.unsqueeze(0).expand(labels.size(0), att.size(1), att.size(1)))
                penal = model.l2_matrix_norm(torch.matmul(att, attT) - identity)
                loss = bce_loss(pred, labels) + (model.att_C * penal / labels.size(0)).type(
                    torch.cuda.FloatTensor) + mse_loss(dists, pred_dists)
                del penal, identity

                if train_true is None:
                    train_pred = pred.detach().cpu().numpy()
                    train_true = labels.detach().cpu().numpy()
                else:
                    train_pred = np.concatenate((train_pred, pred.detach().cpu().numpy()), axis=0)
                    train_true = np.concatenate((train_true, labels.detach().cpu().numpy()), axis=0)
            else:
                pred = model(feats, dists)
                loss = bce_loss(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()

        model.eval()
        accuracy, precision, recall, f1, tp, fp, tn, fn = misc_utils.calc_metrics(train_true.squeeze(), train_pred.squeeze(), 0.5)
        print("====================================训练=============================================",wait)
        print("训练集合正例准确率,Recall(召回率):", recall,",负例准确率:",tn *1.0 / (tn +fp),",F1",f1)
        valid_pred, valid_true = predict(model, valid_loader)
        val_AUC, val_AUPR = misc_utils.evaluator(valid_true, valid_pred, out_keys=["AUC", "AUPR"])
        print("\nvalid_result({})\t{:.4f}\t{:.4f}\t({})".format(epoch_idx, val_AUC, val_AUPR, time.asctime()))
        accuracy, precision, recall, f1,tp, fp, tn, fn  =  misc_utils.calc_metrics(valid_true,valid_pred,0.5)
        accuracy1 = recall
        print("====================================验证=============================================")
        print("验证集合正例准确率,Recall(召回率):", recall,",负例准确率:",tn *1.0 / (tn +fp),",F1",f1)

        # if val_AUC + val_AUPR > best_val_auc + best_val_aupr:
        # if val_AUC + val_AUPR > best_val_auc + best_val_aupr:
        if accuracy1 > best_acc:
            best_acc = accuracy1
            best_F1  = f1
            wait = 0
            best_epoch, best_val_auc, best_val_aupr = epoch_idx, val_AUC, val_AUPR
            test_pred, test_true = predict(model, test_loader)
            accuracy, precision, recall, f1, tp, fp, tn, fn = misc_utils.calc_metrics(test_true, test_pred, 0.5)
            print("====================================测试=============================================")
            print("测试集合正例准确率,Recall(召回率):", recall,",负例准确率:",tn *1.0 / (tn +fp),",F1",f1)
            np.savetxt(
                "{}/test_result.{}.txt.gz".format(outdir, epoch_idx),
                X=np.concatenate((test_pred.reshape(-1, 1), test_true.reshape(-1, 1)), axis=1),
                fmt="%.5f",
                delimiter='\t'
            )
            test_AUC, test_AUPR, F1 = misc_utils.evaluator(test_true, test_pred, out_keys=["AUC", "AUPR", "F1"])
            print("Test_result\t{:.4f}\t{:.4f}\t{:.4f}\t({})".format(test_AUC, test_AUPR, F1, time.asctime()))

            if use_scheduler:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                }, "{}/checkpoint.{}.pt".format(outdir, epoch_idx))
            else:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, "{}/checkpoint.{}.pt".format(outdir, epoch_idx))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopped ({})".format(time.asctime()))
                print("Best epoch/AUC/AUPR/Recall: {}\t{:.4f}\t{:.4f}".format(best_epoch, best_val_auc, best_val_aupr))
                print("Best epoch/Recall/F1: {}\t{:.4f}\t{:.4f}".format(best_epoch, best_acc, best_F1))
                break
            else:
                print("Wait{} ({})".format(wait, time.asctime()))


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        '--train',
        required=True,
        nargs='+'
    )
    p.add_argument(
        '--valid',
        required=False,
        nargs='+'
    )
    p.add_argument(
        "--test",
        nargs='+',
        default=None,
        help="Optional test set"
    )
    p.add_argument('-b', "--batch-size", type=int, default=256)
    p.add_argument('-c', "--config", required=True)
    p.add_argument('-o', "--outdir", required=True)
    p.add_argument("--threads", default=32, type=int)
    p.add_argument('--seed', type=int, default=2020)
    p.add_argument('--gpu', type=int, default=-1)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = json.load(open(args.config))

    # all_data = epi_dataset.EPIDataset(**config["data_opts"])
    train_config = config.copy()
    train_config["data_opts"]["datasets"] = args.train
    # train_config["data_opts"]["use_reverse"] = args.use_reverse
    # train_config["data_opts"]["max_aug"] = args.aug_num
    train_data = epi_dataset.EPIDataset(
        **train_config["data_opts"]
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)

    if args.test is None:
        valid_test_config = copy.deepcopy(config)
        if args.valid == None:
            valid_test_config["data_opts"]["datasets"] = args.train
            valid_test_data = epi_dataset.EPIDataset(
                **valid_test_config["data_opts"]
            )
            chr = []
            count =0
            for i in valid_test_data.samples:
                chr.append(count)
                count +=1
            random.shuffle(chr)
            train_idx = chr[0:int(len(chr)*0.8)]
            valid_idx = chr[int(len(chr)*0.8):int(len(chr)*0.9)]
            test_idx = chr[int(len(chr)*0.9):]
            valid_data = Subset(valid_test_data, indices=valid_idx)
            test_data = Subset(valid_test_data, indices=test_idx)
            train_sub_data = Subset(valid_test_data, indices=train_idx)
            valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
            train_loader = DataLoader(train_sub_data, batch_size=args.batch_size, shuffle=False)
            print("----------------------------------------------------------------------------")
        else:
            valid_test_config["data_opts"]["datasets"] = args.valid
            valid_test_data = epi_dataset.EPIDataset(
                **valid_test_config["data_opts"]
            )
            chr = []
            for i in valid_test_data.samples:
                chr.append(i[-4])
            valid_idx, test_idx = split_train_valid_test(
                np.array(chr),
                train_keys=["chr{}".format(i).replace("23", "X") for i in range(1, 24, 2)],
                valid_keys=["chr{}".format(i) for i in range(2, 22, 2)]
            )

            valid_data = Subset(valid_test_data, indices=valid_idx)
            test_data = Subset(valid_test_data, indices=test_idx)
            valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    else:
        valid_config = copy.deepcopy(config)
        valid_config["data_opts"]["datasets"] = args.valid
        valid_data = epi_dataset.EPIDataset(
            **valid_config["data_opts"]
        )
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

        test_config = copy.deepcopy(config)
        test_config["data_opts"]["datasets"] = args.test
        test_data = epi_dataset.EPIDataset(
            **test_config["data_opts"]
        )
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)

    config["model_opts"]["in_dim"] = train_data.feat_dim
    config["model_opts"]["seq_len"] = config["data_opts"]["seq_len"] // config["data_opts"]["bin_size"]

    print("##{}".format(time.asctime()))
    print("##command: {}".format(' '.join(sys.argv)))
    print("##args: {}".format(args))
    print("##config: {}".format(config))
    print("##sample size: {}".format(len(train_data)))
    print("## feature size: {}".format([v.size() for v in train_data.__getitem__(0)]))

    if args.gpu == -1 or  torch.cuda.is_available()==False:
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda"
    device = torch.device(device)

    model_class = getattr(epi_models, config["model_opts"]["model"])
    model = model_class(**config["model_opts"]).to(device)

    optimizer_params = {'lr': config["train_opts"]["learning_rate"], 'weight_decay': 0}
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)

    print(model)
    print(model_summary(model))
    print(optimizer)

    if not os.path.isdir(args.outdir):
        args.outdir = make_directory(args.outdir)

    train_validate_test(
        model,
        optimizer,
        train_loader, valid_loader, test_loader,
        num_epoch=config["train_opts"]["num_epoch"],
        patience=config["train_opts"]["patience"],
        outdir=args.outdir,
        checkpoint_prefix="checkpoint",
        device=device,
        use_scheduler=config["train_opts"]["use_scheduler"]
    )

# nohup python -u train_validate.py --gpu 0 --c TransEPI_EPI_zdf_train_val.json --train ../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz --valid  ../data/BENGI/GM12878.HiC-Benchmark.v3.tsv.gz -o zdf_train_val  > train_val_samplesData_gpu.log 2>&1 &
# nohup python -u train_validate.py --gpu -1 --c TransEPI_EPI_zdf_train_val.json --train ../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz --valid  ../data/BENGI/GM12878.HiC-Benchmark.v3.tsv.gz -o zdf_train_val  > train_val_samplesData_cpu.log 2>&1 &
# 3455058

# --gpu 0 --c TransEPI_EPI_zdf_train_val.json --train ../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz ../data/BENGI/GM12878.HiC-Benchmark.v3.tsv.gz ../data/BENGI/GM12878.RNAPII-ChIAPET-Benchmark.v3.tsv.gz --valid ../data/BENGI/HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz --test ../data/BENGI/HeLa.HiC-Benchmark.v3.tsv.gz -b 128
# --gpu 0 --c TransEPI_EPI_zdf_train_val.json --train ../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz  --valid ../data/BENGI/HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz --test ../data/BENGI/HeLa.HiC-Benchmark.v3.tsv.gz -b 128
# --gpu 0 --c TransEPI_EPI_zdf_train_val.json --train ../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz --valid ../data/BENGI/HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz --test ../data/BENGI/HeLa.HiC-Benchmark.v3.tsv.gz -b 128 -o zdfoutput1
# --gpu 0 --c TransEPI_EPI_zdf_train_val.json--train ../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz ../data/BENGI/GM12878.HiC-Benchmark.v3.tsv.gz ../data/BENGI/GM12878.RNAPII-ChIAPET-Benchmark.v3.tsv.gz ../data/BENGI/HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz ../data/BENGI/HeLa.HiC-Benchmark.v3.tsv.gz ../data/BENGI/IMR90.HiC-Benchmark.v3.tsv.gz--valid ../data/BENGI/K562.HiC-Benchmark.v3.tsv.gz --test ../data/BENGI/NHEK.HiC-Benchmark.v3.tsv.gz -b 128 -o zdfoutput1
#全部训练集 --gpu 0 --c TransEPI_EPI_zdf_train_val.json --train ../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz ../data/BENGI/GM12878.HiC-Benchmark.v3.tsv.gz ../data/BENGI/GM12878.RNAPII-ChIAPET-Benchmark.v3.tsv.gz ../data/BENGI/HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz --valid ../data/BENGI/K562.HiC-Benchmark.v3.tsv.gz --test ../data/BENGI/NHEK.HiC-Benchmark.v3.tsv.gz -b 128 -o zdfoutput1
#1个训练集 --gpu 0 --c TransEPI_EPI_zdf_train_val.json --train ../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz --valid ../data/BENGI/K562.HiC-Benchmark.v3.tsv.gz --test ../data/BENGI/NHEK.HiC-Benchmark.v3.tsv.gz -b 128 -o zdfoutput1

#均衡采样 随机打乱顺序 8:1:1
#../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz   Best epoch/Recall/F1: 22	0.9761	0.8813
#../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz   Best epoch/Recall/F1: 16	0.9705	0.9085
#../data/BENGI/GM12878.RNAPII-ChIAPET-Benchmark.v3.tsv.gz Best epoch/Recall/F1: 16	0.9705	0.9085
#../data/BENGI/HeLa.CTCF-ChIAPET-Benchmark.v3.tsv.gz     Best epoch/Recall/F1: 7	0.8786	0.8849
#../data/BENGI/HeLa.HiC-Benchmark.v3.tsv.gz               Best epoch/Recall/F1: 9	0.9320	0.8118
#../data/BENGI/HeLa.RNAPII-ChIAPET-Benchmark.v3.tsv.gz    Best epoch/Recall/F1: 27	0.9841	0.9254
#../data/BENGI/HMEC.HiC-Benchmark.v3.tsv.gz               Best epoch/Recall/F1: 20	0.9390	0.9352
#../data/BENGI/IMR90.HiC-Benchmark.v3.tsv.gz              Best epoch/Recall/F1: 22	0.8841	0.8592
#../data/BENGI/K562.HiC-Benchmark.v3.tsv.gz               Best epoch/Recall/F1: 28	0.8599	0.7741
#../data/BENGI/NHEK.HiC-Benchmark.v3.tsv.gz               Best epoch/Recall/F1: 30	0.9417	0.9440


# 原来的正负例子采样方式
#../data/BENGI/GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz  Best epoch/Recall/F1: 35	0.8780	0.8016

