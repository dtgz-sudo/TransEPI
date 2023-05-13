#!/usr/bin/env python3

import argparse, os, sys, time
import warnings, json, gzip
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
import argparse, os, sys, time, shutil, tqdm

import epi_models
import epi_dataset
import misc_utils


import functools
print = functools.partial(print, flush=True)


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
    result, true_label = list(), list()
    for feats, _, enh_idxs, prom_idxs, labels in tqdm.tqdm( data_loader):
        feats, labels = feats.to(device), labels.to(device)
        # enh_idxs, prom_idxs = feats.to(device), prom_idxs.to(device)
        pred = model(feats, enh_idx=enh_idxs, prom_idx=prom_idxs)
        pred = pred.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        result.append(pred)
        true_label.append(labels)
    result = np.concatenate(result, axis=0)
    true_label = np.concatenate(true_label, axis=0)
    return (result.squeeze(), true_label.squeeze())


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-t', "--test-data", nargs='+', required=True, help="test dataset")
    p.add_argument('--test-chroms', nargs='+', required=False, default=['all'], help="chromosomes used for evaluation")
    p.add_argument('--gpu', default=-1, type=int, help="GPU ID, (-1 for CPU)")
    p.add_argument('--batch-size',  default=32, type=int, help="batch size")
    p.add_argument('--num-workers',  default=16, type=int, help="number of the processes used by data loader ")
    p.add_argument('-c', "--config", required=True, help="model configuration")
    p.add_argument('-m', "--model", required=True, help="path to trained model")
    p.add_argument('-p', "--prefix", required=True, help="prefix of output file")

    #p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    #np.random.seed(args.seed)

    if not torch.cuda.is_available():
        # warnings.warn("GPU is not available")
        args.gpu = -1

    config = json.load(open(args.config))

    config["data_opts"]["datasets"] = args.test_data
    config["model_opts"]["rand_shift"] = False

    all_data = epi_dataset.EPIDataset(**config["data_opts"])
    print("加载数据")
    config["model_opts"]["in_dim"] = all_data.feat_dim
    config["model_opts"]["seq_len"] = config["data_opts"]["seq_len"] // config["data_opts"]["bin_size"]

    labels = all_data.metainfo["label"]

    chroms = np.array(all_data.metainfo["chrom"])
    enh_names = np.array(all_data.metainfo["enh_name"])
    prom_names = np.array(all_data.metainfo["prom_name"])

    if args.test_chroms[0] != "all":
        config["train_opts"]["valid_chroms"] = args.test_chroms
        chroms = all_data.metainfo["chrom"]
        _, test_idx = misc_utils.split_np_array(chroms, test_chroms=config["train_opts"]["valid_chroms"])
        labels = labels[test_idx]
        all_data = Subset(all_data, indices=test_idx)
        chroms = chroms[test_idx]
        enh_names = enh_names[test_idx]
        prom_names = prom_names[test_idx]

    test_loader = DataLoader(
            all_data,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    print("获取模型")
    model_class = getattr(epi_models, config["model_opts"]["model"])
    model = model_class(**config["model_opts"]).to(device)
    print("加载模型")
    model.load_state_dict(torch.load(args.model, map_location=device)["model_state_dict"])
    model.eval()
    print("开始执行")
    pred, true = predict(model, test_loader, device)

    # np.savetxt(
    #         "{}.prediction.txt".format(args.prefix),
    #         np.concatenate((
    #             true.reshape(-1, 1).astype(int).astype(str),
    #             pred.reshape(-1, 1).round(4).astype(str),
    #             chroms.reshape(-1, 1),
    #             enh_names.reshape(-1, 1),
    #             prom_names.reshape(-1, 1),
    #         ), axis=1),
    #         delimiter='\t',
    #         fmt="%s",
    #         comments="",
    #         header="##command: {}\n#true\tpred\tchrom\tenh_name\tprom_name".format(' '.join(sys.argv))
    #     )

    AUC, AUPR = misc_utils.evaluator(true, pred, out_keys=['AUC', 'AUPR'])

    label_counts = misc_utils.count_unique_itmes(labels)
    print("## datasets: {}".format(config["data_opts"]["datasets"]))
    print("## testing chroms: {}".format(args.test_chroms))
    print("## label classes: {}".format(label_counts))
    print("AUC:\t{:.4f}\nAUPR:\t{:.4f}({:.4f})".format(AUC, AUPR, AUPR / label_counts[1]))


    def calc_roc_auc(y_true, y_pred):
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        # 计算FPR, TPR和阈值
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        # 计算AUC
        roc_auc = auc(fpr, tpr)

        # 绘制ROC曲线
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
        print("roc_auc：", roc_auc)


    def f1_score(y_true, y_pred, average='binary'):
        """计算F1-score

        参数：
        y_true：实际标签，可以是列表或numpy数组
        y_pred：预测标签，可以是列表或numpy数组
        average：计算方法，可以是'binary'、'micro'或'macro'。默认为'binary'

        返回值：
        F1-score值
        """
        from sklearn.metrics import f1_score as sk_f1_score

        if average == 'binary':
            return sk_f1_score(y_true, y_pred, average='binary')
        elif average == 'micro':
            return sk_f1_score(y_true, y_pred, average='micro')
        elif average == 'macro':
            return sk_f1_score(y_true, y_pred, average='macro', labels=np.unique(y_true))
        else:
            raise ValueError("average参数只能是'binary'、'micro'或'macro'中的一个")


    n = 0
    p = 0
    n1 = 0
    p1 = 0
    y_true = []
    y_pred = []
    y_pred1 = []
    # with    open("zdf.prediction.txt") as f:
    length = len(true)
    true_arr = true
    predict_arr = pred
    pred = None
    result = []
    for epoch in range(0, 11):
        num = epoch / 10.0
        # print()
        # print()
        # print()
        # print("epoch:",num)
        pred_arr = (predict_arr > num).astype(int)

        p = np.sum(true_arr == 1)
        n = np.sum(true_arr == 0)
        p1 = np.sum(np.logical_and(true_arr == 1, pred_arr == 1))
        n1 = np.sum(np.logical_and(true_arr == 0, pred_arr == 0))

        y_true.extend(true_arr)
        y_pred.extend(pred_arr)
        y_pred1.extend(predict_arr)

        accuracy = (p1 + n1) * 1.0 / (p + n)
        precision = p1 / p
        recall = n1 / n

        f1 = f1_score(y_true, y_pred, average='binary')
        print(num, accuracy,precision, recall, accuracy, f1)
#         result.append([num, accuracy,precision, recall, accuracy, f1])
#     #
# #
# result_df = pd.DataFrame(result, columns=['num',"accuracy", 'precision', 'recall', 'accuracy', 'f1_score'])
# print(result_df)
