#!/usr/bin/env python3

import argparse, os, sys, time
import random
import warnings, json, gzip
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from misc_utils import hg19_chromsize

import numpy as np

from typing import Dict, List, Union

from functools import partial


def custom_open(fn):
    if fn.endswith("gz"):
        return gzip.open(fn, 'rt')
    else:
        return open(fn, 'rt')


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument()

    p.add_argument('--seed', type=int, default=2020)
    return p


class EPIDataset(Dataset):
    def __init__(self,
            datasets: Union[str, List],
            feats_config: Dict[str, str],
            feats_order: List[str],
            seq_len: int=2500000,
            bin_size: int=500,
            use_mark: bool=False,
            mask_neighbor=False,
            mask_window=False,
            sin_encoding=False,
            rand_shift=False,

            **kwargs):
        super(EPIDataset, self).__init__()

        if type(datasets) is str:
            self.datasets = [datasets]
        else:
            self.datasets = datasets

        self.seq_len = int(seq_len)
        self.bin_size = int(bin_size)
        assert self.seq_len % self.bin_size == 0, "{} / {}".format(self.seq_len, self.bin_size)
        self.num_bins = seq_len // bin_size

        self.feats_order = list(feats_order)  # 所有的特征
        self.num_feats = len(feats_order)  # 所有的特征数量
        self.feats_config = json.load(open(feats_config))  # 加载配置文件
        if "_location" in self.feats_config:  # 是否指定了路径否则就是当前路径
            location = self.feats_config["_location"]
            del self.feats_config["_location"]
            for cell, assays in self.feats_config.items():
                for a, fn in assays.items():
                    self.feats_config[cell][a] = os.path.join(location, fn)
        else:
            location = os.path.dirname(os.path.abspath(feats_config))  # json数据组装起来
            for cell, assays in self.feats_config.items():
                for a, fn in assays.items():
                    self.feats_config[cell][a] = os.path.join(location, fn)  # 全部的数据组装起来 (输入的特征转换由绝对路径改为绝对路径)

        self.feats = dict()  # cell_name -> feature_name -> chrom > features (array)
        self.chrom_bins = {
            chrom: (length // bin_size) for chrom, length in hg19_chromsize.items()
        }  # 每个染色体上面bin的数量

        self.samples = list()
        self.metainfo = {
            'label': list(),
            'dist': list(),
            'chrom': list(),
            'cell': list(),
            'enh_name': list(),
            'prom_name': list(),
            'shift': list()
        }

        self.sin_encoding = sin_encoding
        self.use_mark = use_mark
        self.mask_window = mask_window
        self.mask_neighbor = mask_neighbor
        self.rand_shift = rand_shift

        self.load_datasets()  # 加载数据
        self.feat_dim = len(self.feats_order) + 1
        if self.use_mark:
            self.feat_dim += 1
        if self.sin_encoding:
            self.feat_dim += 1
        # 进行数据的正负例均衡采样
        # self.samples.append((
        #     start_bin + shift, stop_bin + shift,
        #     left_pad_bin, right_pad_bin,
        #     enh_bin, prom_bin,
        #     cell, chrom, np.log2(1 + 500000 / float(dist)),
        #     int(label), knock_range
        # ))
        length = len(self.samples)
        idx_pos = []
        idx_neg = []
        for i in range(length):
            if self.samples[i][-2] == 1:
                idx_pos.append(self.samples[i])
            elif self.samples[i][-2] == 0:
                idx_neg.append(self.samples[i])
            else:
                print("发生错误")
        if len(idx_neg) > len(idx_pos):
            sample = random.sample(idx_neg, len(idx_pos))
            sample = sample + idx_pos
            random.shuffle(sample)
            random.shuffle(sample)
            random.shuffle(sample)
            self.samples = sample
    def load_datasets(self):
        for fn in self.datasets:
            with custom_open(fn) as infile:
                for l in infile:
                    fields = l.strip().split(
                        '\t')  # '1\t92874.5\tchr9\t613609\t614420\tchr9:613609-614420|HMEC|EH37E0985881\tchr9\t705389\t707389\tchr9:706888-706889|HMEC|ENSG00000107104.14|ENST00000382293.3|+\n'
                    label, dist, chrom, enh_start, enh_end, enh_name, \
                        _, prom_start, prom_end, prom_name = fields[
                                                             0:10]  # 标签 距离 染色体号 增强子开始位置 结束位置 增强子名字 染色体号（-）  启动子开始位置 结束位置 启动子名字
                    knock_range = None
                    if len(fields) > 10:
                        assert len(fields) == 11
                        knock_range = list()
                        for knock in fields[10].split(';'):
                            knock_start, knock_end = knock.split('-')
                            knock_start, knock_end = int(knock_start), int(knock_end)
                            knock_range.append((knock_start, knock_end))

                    cell = enh_name.split('|')[1]  # 细胞名称
                    strand = prom_name.split('|')[-1]  # 染色体正链（+）还是负链（-）

                    enh_coord = (int(enh_start) + int(enh_end)) // 2  # 计算出增强子的中心位置
                    p_start, p_end = prom_name.split('|')[0].split(':')[-1].split('-')
                    tss_coord = (int(p_start) + int(p_end)) // 2  # 启动子中心位置

                    seq_begin = (enh_coord + tss_coord) // 2 - self.seq_len // 2  # 选取的序列区间的开始位置
                    seq_end = (enh_coord + tss_coord) // 2 + self.seq_len // 2  # 选取的序列区间的结束位置

                    # enh_bin = (enh_coord - seq_begin) // self.bin_size
                    enh_bin = enh_coord // self.bin_size  # 增强子bin的个数
                    # prom_bin = (tss_coord - seq_begin) // self.bin_size
                    prom_bin = tss_coord // self.bin_size  # 启动子bin的个数
                    start_bin, stop_bin = seq_begin // self.bin_size, seq_end // self.bin_size  # 开始bin的ID 结束bin的ID

                    left_pad_bin, right_pad_bin = 0, 0
                    if start_bin < 0:
                        left_pad_bin = abs(start_bin)
                        start_bin = 0
                    if stop_bin > self.chrom_bins[chrom]:
                        right_pad_bin = stop_bin - self.chrom_bins[chrom]
                        stop_bin = self.chrom_bins[chrom]

                    shift = 0
                    if self.rand_shift:  # 如果 self.rand_shift 为 True，则在序列区间和增强子/启动子的位置差距较大时进行随机平移。
                        if left_pad_bin > 0:
                            shift = left_pad_bin
                            start_bin = -left_pad_bin
                            left_pad_bin = 0
                        elif right_pad_bin > 0:
                            shift = -right_pad_bin
                            stop_bin = self.chrom_bins[chrom] + right_pad_bin
                            right_pad_bin = 0
                        else:
                            min_range = min(min(enh_bin, prom_bin) - start_bin, stop_bin - max(enh_bin, prom_bin))
                            if min_range > (self.num_bins / 4):
                                shift = np.random.randint(-self.num_bins // 5, self.num_bins // 5)
                            if start_bin + shift <= 0 or stop_bin + shift >= self.chrom_bins[chrom]:
                                shift = 0

                    self.samples.append((
                        start_bin + shift, stop_bin + shift,
                        left_pad_bin, right_pad_bin,
                        enh_bin, prom_bin,
                        cell, chrom, np.log2(1 + 500000 / float(dist)),
                        int(label), knock_range
                    ))
                    # print(l.strip())
                    # print(self.samples[-1])
                    # print(enh_coord, enh_coord // self.bin_size, tss_coord, tss_coord // self.bin_size, seq_begin, seq_begin // self.bin_size, seq_end, seq_end // self.bin_size, start_bin, stop_bin, left_pad_bin, right_pad_bin)

                    self.metainfo['label'].append(int(label))
                    self.metainfo['dist'].append(float(dist))
                    self.metainfo['chrom'].append(chrom)
                    self.metainfo['cell'].append(cell)
                    self.metainfo['enh_name'].append(enh_name)
                    self.metainfo['prom_name'].append(prom_name)
                    self.metainfo['shift'].append(shift)

                    if cell not in self.feats:  # 读取对应的特征
                        self.feats[cell] = dict()
                        for feat in self.feats_order:
                            try:
                                self.feats[cell][feat] = torch.load(self.feats_config[cell][feat])
                            except:
                                print("文件不存在:", self.feats_config[cell][feat])
        for k in self.metainfo:
            self.metainfo[k] = np.array(self.metainfo[k])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 这段代码实现了一个 PyTorch 数据集的 __getitem__ 方法，用于返回数据集中指定索引位置的数据。该数据集包含了启动子-增强子对的信息以及相关的特征向量，用于训练一个二分
        # 类器来预测这些启动子-增强子对是否在染色体三维结构中相互作用。具体来说，该方法的实现过程如下：
        # 1、从数据集中获取指定索引位置的信息，该信息包含了该条启动子-增强子对的起始和结束 bin 位置，以及该条对应的基因在三维染色体结构中的相对距离、标签等信息。
        # 2、根据 bin 位置和填充信息，将数据集中保存的特征向量提取出来。这些特征向量包含了多种基因表达和染色体结构的相关特征。
        # 3、对于可能的缺失值进行填充，如有 knock_range 信息则将其置为 0，否则将其置为 1。
        # 4、根据设置的窗口大小和标记信息，将提取的特征向量进行处理，得到最终的输入特征向量。
        # 5、将该条数据的特征向量、距离、增强子和启动子位置、标签等信息打包成一个元组返回
        # 该 __getitem__ 方法通常用于 PyTorch 的 DataLoader 中，用于迭代访问数据集并获取相应的数据。
        # minibach 采样的方法
        start_bin, stop_bin, left_pad, right_pad, enh_bin, prom_bin, cell, chrom, dist, label, knock_range = \
            self.samples[idx]  # 该条启动子_增强子信息 起始 bin、终止 bin、左右填充长度、增强子 bin、启动子 bin、细胞类型、染色体、距离、标签和敲除范围
        enh_idx = enh_bin - start_bin + left_pad
        prom_idx = prom_bin - start_bin + left_pad

        # print(self.samples[idx], self.metainfo["shift"][idx])
        ar = torch.zeros((0, stop_bin - start_bin))
        # print(start_bin - left_pad, stop_bin + right_pad, enh_bin, prom_bin, enh_idx, prom_idx)
        for feat in self.feats_order:
            ar = torch.cat((ar, self.feats[cell][feat][chrom][start_bin:stop_bin].view(1, -1)),
                           dim=0)  # 根据染色体bin的距离计算出特征
        ar = torch.cat((
            torch.zeros((self.num_feats, left_pad)),
            ar,
            torch.zeros((self.num_feats, right_pad))
        ), dim=1)

        if knock_range is not None:
            dim, length = ar.size()
            mask = [1 for _ in range(self.num_bins)]
            for knock_start, knock_end in knock_range:
                knock_start = knock_start // self.bin_size - start_bin + left_pad
                knock_end = knock_end // self.bin_size - start_bin + left_pad
                for pos in range(max(0, knock_start), min(knock_end + 1, self.num_bins)):
                    mask[pos] = 0
            mask = np.array(mask, dtype=np.float32).reshape(1, -1)
            mask = np.concatenate([mask for _ in range(dim)], axis=0)
            mask = torch.FloatTensor(mask)
            ar = ar * mask

        if self.mask_window:
            shift = min(abs(enh_bin - prom_bin) - 5, 0)
            mask = torch.cat((
                torch.ones(ar.size(0), min(enh_bin, prom_bin) - start_bin + left_pad + 3 + shift),
                torch.zeros(ar.size(0), max(abs(enh_idx - prom_idx) - 5, 0)),
                torch.ones(ar.size(0), stop_bin + right_pad - max(enh_bin, prom_bin) + 2)
            ), dim=1)
            assert mask.size() == ar.size(), "{}".format(mask.size())
            ar = ar * mask
        if self.mask_neighbor:
            mask = torch.cat((
                torch.zeros(ar.size(0), min(enh_bin, prom_bin) - start_bin + left_pad - 2),
                torch.ones(ar.size(0), abs(enh_bin - prom_bin) + 5),
                torch.zeros(ar.size(0), stop_bin + right_pad - max(enh_bin, prom_bin) - 3)
            ), dim=1)
            assert mask.size() == ar.size(), "{}".format(mask.size())
            ar = ar * mask

        pos_enc = torch.arange(self.num_bins).view(1, -1)
        pos_enc = torch.cat((pos_enc - min(enh_idx, prom_idx), max(enh_idx, prom_idx) - pos_enc), dim=0)
        if self.sin_encoding:
            pos_enc = torch.sin(pos_enc / 2 / self.num_bins * np.pi).view(2, -1)
        else:
            pos_enc = self.sym_log(pos_enc.min(dim=0)[0]).view(1, -1)
        ar = torch.cat((torch.as_tensor(pos_enc, dtype=torch.float), ar), dim=0)

        if self.use_mark:
            mark = [0 for i in range(self.num_bins)]
            mark[enh_idx] = 1
            mark[enh_idx - 1] = 1
            mark[enh_idx + 1] = 1
            mark[prom_idx] = 1
            mark[prom_idx - 1] = 1
            mark[prom_idx + 1] = 1
            ar = torch.cat((
                torch.as_tensor(mark, dtype=torch.float).view(1, -1),
                ar
            ), dim=0)

        return ar, torch.as_tensor([dist], dtype=torch.float), torch.as_tensor([enh_idx], dtype=torch.float), torch.as_tensor([prom_idx], dtype=torch.float), torch.as_tensor([label], dtype=torch.float)

    def sym_log(self, ar):
        sign = torch.sign(ar)
        ar = sign * torch.log10(1 + torch.abs(ar))
        return ar


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    np.random.seed(args.seed)

    all_data = EPIDataset(
            datasets=["../data/BENGI/GM12878.HiC-Benchmark.v3.tsv"],
            feats_config="../data/genomic_features/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            seq_len=2500000,
            bin_size=500,
            mask_window=True,
            mask_neighbor=True,
            sin_encoding=True,
            rand_shift=True
        )

    for i in range(0, len(all_data), 411):
        np.savetxt(
                "data_{}".format(i),
                all_data.__getitem__(i)[0].T,
                fmt="%.4f",
                header="{}\t{}\t{}\n{}".format(all_data.metainfo["label"][i], all_data.metainfo["enh_name"][i], all_data.metainfo["prom_name"][i], all_data.samples[i])
            )


#     batch_size = 16
#     data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, num_workers=8)
#
#     # # import epi_models
#     # # model = epi_models.LstmAttModel(in_dim=6,
#     # #         lstm_size=32, lstm_layer=2, lstm_dropout=0.2,
#     # #         da=64, r=32,
#     # #         fc=[64, 32], fc_dropout=0.2)
#     # import epi_models
#     # model = epi_models.PerformerModel(
#     #         in_dim=6,
#     #         cnn_channels=[128],
#     #         cnn_sizes=[11],
#     #         cnn_pool=[5],
#     #         enc_layers=4,
#     #         num_heads=4,
#     #         d_inner=128,
#     #         fc=[32, 16],
#     #         fc_dropout=0.1
#     #     ).cuda()
#     for i, (feat, dist, label) in enumerate(data_loader):
#         print()
#         print(feat.size(), dist.size(), label.size())
#         # torch.save({'feat': feat, 'label': label}, "tmp.pt")
#         # feat = model(feat.cuda())
#         print(feat.size())
#         # for k in all_data.metainfo:
#         #     print(k, all_data.metainfo[k][i])
#         # if i > 200:
#         #     break
#
