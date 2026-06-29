"""
CTNet: A Convolution-Transformer Network for EEG-Based Motor Imagery Classification

author: zhaowei701@163.com

Due to memory constraints, the data augmentation method in LOSO classification was slightly optimized based on the approach used in subject-specific classification (main.py).


Cite this work
Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024). https://doi.org/10.1038/s41598-024-71118-7


"""

import os

gpus = [0]
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import random
import datetime
import time
import math
from pandas import ExcelWriter
from torchsummary import summary
import torch
from torch.backends import cudnn
from utils import calMetrics
from utils import calculatePerClass
from utils import numberClassChannel

import warnings

warnings.filterwarnings("ignore")
cudnn.benchmark = False
cudnn.deterministic = True

import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

from utils import numberClassChannel
from utils import load_data_evaluate

import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from model import EEGTransformer


class ExP:
    def __init__(
        self,
        nsub,
        data_dir,
        result_name,
        epochs=2000,
        number_aug=2,
        number_seg=8,
        gpus=[0],
        evaluate_mode="subject-dependent",
        heads=4,
        emb_size=40,
        depth=6,
        dataset_type="A",
        eeg1_f1=20,
        eeg1_kernel_size=64,
        eeg1_D=2,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.3,
        flatten_eeg1=600,
        validate_ratio=0.2,
        learning_rate=0.001,
        batch_size=72,
    ):

        super(ExP, self).__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.lr = learning_rate
        self.b1 = 0.5
        self.b2 = 0.999
        self.n_epochs = epochs
        self.nSub = nsub
        self.number_augmentation = number_aug
        self.number_seg = number_seg
        self.root = data_dir
        self.heads = heads
        self.emb_size = emb_size
        self.depth = depth
        self.result_name = result_name
        self.evaluate_mode = evaluate_mode
        self.validate_ratio = validate_ratio

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.number_class, self.number_channel = numberClassChannel(self.dataset_type)
        self.model = EEGTransformer(
            heads=self.heads,
            emb_size=self.emb_size,
            depth=self.depth,
            database_type=self.dataset_type,
            eeg1_f1=eeg1_f1,
            eeg1_D=eeg1_D,
            eeg1_kernel_size=eeg1_kernel_size,
            eeg1_pooling_size1=eeg1_pooling_size1,
            eeg1_pooling_size2=eeg1_pooling_size2,
            eeg1_dropout_rate=eeg1_dropout_rate,
            eeg1_number_channel=self.number_channel,
            flatten_eeg1=flatten_eeg1,
        ).cuda()
        # self.model = nn.DataParallel(self.model, device_ids=gpus)
        self.model = self.model.cuda()
        self.model_filename = self.result_name + "/model_{}.pth".format(self.nSub)

    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):
        aug_data = []
        aug_label = []

        number_segmentation_points = 1000 // self.number_seg
        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug + 1)
            tmp_data = timg[cls_idx]
            number_records_by_augmentation = (
                self.number_augmentation * tmp_data.shape[0]
            )
            tmp_aug_data = np.zeros(
                (number_records_by_augmentation, 1, self.number_channel, 1000)
            )
            for ri in range(number_records_by_augmentation):
                for rj in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    tmp_aug_data[
                        ri,
                        :,
                        :,
                        rj * number_segmentation_points : (rj + 1)
                        * number_segmentation_points,
                    ] = tmp_data[
                        rand_idx[rj],
                        :,
                        :,
                        rj * number_segmentation_points : (rj + 1)
                        * number_segmentation_points,
                    ]

            aug_data.append(tmp_aug_data)
            aug_label.append([clsAug + 1] * number_records_by_augmentation)
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        # aug_shuffle = np.random.permutation(len(aug_data))
        # aug_data = aug_data[aug_shuffle, :, :]
        # aug_label = aug_label[aug_shuffle]

        # aug_data = torch.from_numpy(aug_data).cuda()
        # aug_data = aug_data.float()
        # aug_label = torch.from_numpy(aug_label-1).cuda()
        # aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):
        (
            self.train_data,  # (batch, channel, length)
            self.train_label,
            self.test_data,
            self.test_label,
        ) = load_data_evaluate(
            self.root, self.dataset_type, self.nSub, mode_evaluate=self.evaluate_mode
        )

        self.train_data = np.expand_dims(self.train_data, axis=1)  # (288, 1, 22, 1000)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]
        # split original allData into training and validate datasets

        train_data_list = []
        train_label_list = []
        validate_data_list = []
        validate_label_list = []
        for cls in range(self.number_class):
            # filter by class
            cls_idx = np.where(self.allLabel == cls + 1)
            cat_data = self.allData[cls_idx]
            cat_label = self.allLabel[cls_idx]

            # each category split
            number_sample = cat_data.shape[0]
            number_validate = int(self.validate_ratio * number_sample)
            # shuffle index
            index_shuffle = np.random.permutation(len(cat_data))
            index_train = index_shuffle[:-number_validate]
            index_validate = index_shuffle[-number_validate:]

            train_data_list.append(cat_data[index_train])
            train_label_list.append(cat_label[index_train])

            validate_data_list.append(cat_data[index_validate])
            validate_label_list.append(cat_label[index_validate])

        # data augmentation
        aug_data, aug_label = self.interaug(self.allData, self.allLabel)

        train_data_list.append(aug_data)
        train_label_list.append(aug_label)

        self.trainData = np.concatenate(train_data_list)
        self.trainLabel = np.concatenate(train_label_list)
        self.validateData = np.concatenate(validate_data_list)
        self.validateLabel = np.concatenate(validate_label_list)

        # shuffle in all category
        shuffle_num = np.random.permutation(len(self.trainData))
        self.trainData = self.trainData[
            shuffle_num, :, :, :
        ]  # (number of training sample, 1, 22, 1000)
        self.trainLabel = self.trainLabel[shuffle_num]

        # self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        # self.allData = (self.allData - target_mean) / target_std
        self.trainData = (self.trainData - target_mean) / target_std
        self.validateData = (self.validateData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        isSaveDataLabel = False  # True
        if isSaveDataLabel:
            np.save("./gradm_data/train_data_{}.npy".format(self.nSub), self.allData)
            np.save("./gradm_data/train_lable_{}.npy".format(self.nSub), self.allLabel)
            np.save("./gradm_data/test_data_{}.npy".format(self.nSub), self.testData)
            np.save("./gradm_data/test_label_{}.npy".format(self.nSub), self.testLabel)
        print(
            self.trainData.shape,
            self.trainLabel.shape,
            self.validateData.shape,
            self.validateLabel.shape,
            self.testData.shape,
            self.testLabel.shape,
        )
        # data shape: (trial, conv channel, electrode channel, time samples)
        return (
            self.trainData,
            self.trainLabel,
            self.validateData,
            self.validateLabel,
            self.testData,
            self.testLabel,
        )

    def fit_test(self, model, loss_fn, testloader):
        y_list = []
        y_pred_list = []

        test_correct = 0
        test_total = 0
        test_running_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in testloader:
                x = Variable(x.type(self.Tensor))
                y = Variable(y.type(self.LongTensor))

                features, y_pred = model(x)
                loss = loss_fn(y_pred, y)
                y_pred = torch.argmax(y_pred, dim=1)
                test_correct += (y_pred == y).sum().item()
                test_total += y.size(0)
                test_running_loss += loss.item()
                y_pred = y_pred.cpu().numpy()
                y = y.cpu().numpy()
                y_list.extend(y)
                y_pred_list.extend(y_pred)

        acc_score = accuracy_score(y_list, y_pred_list)
        epoch_test_loss = test_running_loss / len(testloader.dataset)

        return epoch_test_loss, acc_score, y_list, y_pred_list

    def fit_train(self, model, loss_fn, dataloader, optimizer, trainData, trainLabel):
        correct = 0
        total = 0
        running_loss = 0
        model.train()

        for train_data, train_label in dataloader:
            # real train dataset
            img = Variable(train_data.type(self.Tensor))
            label = Variable(train_label.type(self.LongTensor))

            # training model
            features, y_pred = model(img)
            # print("train outputs: ", outputs.shape, type(outputs))
            # print(features.size())
            loss = loss_fn(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_pred = torch.argmax(y_pred, dim=1)
                correct += (y_pred == label).sum().item()
                total += label.size(0)
                running_loss += loss.item()

        epoch_train_loss = running_loss / len(dataloader.dataset)
        epoch_train_acc = correct / total

        return epoch_train_loss, epoch_train_acc

    def train(self):
        img, label, validate_data, validate_label, test_data, test_label = (
            self.get_source_data()
        )
        # train dataset
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True
        )
        # validate dataset
        validate_data = torch.from_numpy(validate_data)
        validate_label = torch.from_numpy(validate_label - 1)
        validate_dataset = torch.utils.data.TensorDataset(validate_data, validate_label)

        self.validate_dataloader = torch.utils.data.DataLoader(
            dataset=validate_dataset, batch_size=288, shuffle=False
        )
        # test dataset
        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=288, shuffle=False
        )

        # Optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        best_epoch = 0
        num = 0
        min_loss = 100
        # recording train_acc, train_loss, test_acc, test_loss
        result_process = []
        # Train the CTNet model
        for e in range(self.n_epochs):
            epoch_process = {}
            epoch_process["epoch"] = e
            # train model
            self.model.train()
            train_loss, train_acc = self.fit_train(
                self.model,
                self.criterion_cls,
                self.dataloader,
                self.optimizer,
                self.allData,
                self.allLabel,
            )
            epoch_process["train_acc"] = train_acc
            epoch_process["train_loss"] = train_loss

            # validate model
            (validate_loss, validate_acc, y_list, y_pred_list) = self.fit_test(
                self.model, self.criterion_cls, self.validate_dataloader
            )
            epoch_process["val_acc"] = validate_acc
            epoch_process["val_loss"] = validate_loss

            #             train_pred = torch.max(outputs, 1)[1]
            #             train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
            num = num + 1

            if min_loss > validate_loss:
                min_loss = validate_loss
                best_epoch = e
                epoch_process["epoch"] = e
                torch.save(self.model, self.model_filename)

                (test_loss, test_acc, y_list, y_pred_list) = self.fit_test(
                    self.model, self.criterion_cls, self.test_dataloader
                )
                epoch_process["test_acc"] = test_acc
                epoch_process["test_loss"] = test_loss
                print(
                    "{}_{} train_acc: {:.4f} train_loss: {:.6f}\tval_acc: {:.6f} val_loss: {:.9f}, acc:{:.6f}".format(
                        self.nSub,
                        epoch_process["epoch"],
                        epoch_process["train_acc"],
                        epoch_process["train_loss"],
                        epoch_process["val_acc"],
                        epoch_process["val_loss"],
                        epoch_process["test_acc"],
                    )
                )

            result_process.append(epoch_process)

        # load model for test
        self.model.eval()
        self.model = torch.load(self.model_filename).cuda()
        # test model
        (test_loss, test_acc, y_list, y_pred_list) = self.fit_test(
            self.model, self.criterion_cls, self.test_dataloader
        )

        print("epoch: ", best_epoch, "\tThe test accuracy is:", test_acc)

        df_process = pd.DataFrame(result_process)

        return (
            test_acc,
            torch.tensor(y_list),
            torch.tensor(y_pred_list),
            df_process,
            best_epoch,
        )
        # writer.close()


def main(
    dirs,
    evaluate_mode="subject-dependent",  # LOSO or not
    heads=8,  # heads of MHA
    emb_size=48,  # token embding dim
    depth=3,  # Transformer encoder depth
    dataset_type="A",  # A->'BCI IV2a', B->'BCI IV2b'
    eeg1_f1=20,  # features of temporal conv
    eeg1_kernel_size=64,  # kernel size of temporal conv
    eeg1_D=2,  # depth-wise conv
    eeg1_pooling_size1=8,  # p1
    eeg1_pooling_size2=8,  # p2
    eeg1_dropout_rate=0.3,
    flatten_eeg1=600,
    validate_ratio=0.2,
    batch_size=72,
):

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    result_write_metric = ExcelWriter(dirs + "/result_metric.xlsx")

    result_metric_dict = {}
    y_true_pred_dict = {}

    process_write = ExcelWriter(dirs + "/process_train.xlsx")
    pred_true_write = ExcelWriter(dirs + "/pred_true.xlsx")
    subjects_result = []
    best_epochs = []

    for i in range(0, N_SUBJECT):
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(2024)
        print("seed is " + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        index_round = 0
        print("Subject %d" % (i + 1))
        exp = ExP(
            i + 1,
            DATA_DIR,
            dirs,
            EPOCHS,
            N_AUG,
            N_SEG,
            gpus,
            evaluate_mode=evaluate_mode,
            heads=heads,
            emb_size=emb_size,
            depth=depth,
            dataset_type=dataset_type,
            eeg1_f1=eeg1_f1,
            eeg1_kernel_size=eeg1_kernel_size,
            eeg1_D=eeg1_D,
            eeg1_pooling_size1=eeg1_pooling_size1,
            eeg1_pooling_size2=eeg1_pooling_size2,
            eeg1_dropout_rate=eeg1_dropout_rate,
            flatten_eeg1=flatten_eeg1,
            validate_ratio=validate_ratio,
            batch_size=batch_size,
        )

        testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
        true_cpu = Y_true.cpu().numpy().astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int)
        df_pred_true = pd.DataFrame({"pred": pred_cpu, "true": true_cpu})
        df_pred_true.to_excel(pred_true_write, sheet_name=str(i + 1))
        y_true_pred_dict[i] = df_pred_true

        accuracy, precison, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
        subject_result = {
            "accuray": accuracy * 100,
            "precision": precison * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "kappa": kappa * 100,
        }
        subjects_result.append(subject_result)
        df_process.to_excel(process_write, sheet_name=str(i + 1))
        best_epochs.append(best_epoch)

        print(" THE BEST ACCURACY IS " + str(testAcc) + "\tkappa is " + str(kappa))

        endtime = datetime.datetime.now()
        print("subject %d duration: " % (i + 1) + str(endtime - starttime))

        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))

        df_result = pd.DataFrame(subjects_result)
    process_write.close()
    pred_true_write.close()

    print(
        "**The average Best accuracy is: "
        + str(df_result["accuray"].mean())
        + "kappa is: "
        + str(df_result["kappa"].mean())
        + "\n"
    )
    print("best epochs: ", best_epochs)
    # df_result.to_excel(result_write_metric, index=False)
    result_metric_dict = df_result

    mean = df_result.mean(axis=0)
    mean.name = "mean"
    std = df_result.std(axis=0)
    std.name = "std"
    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])

    df_result.to_excel(result_write_metric, index=False)
    print("-" * 9, " all result ", "-" * 9)
    print(df_result)

    print("*" * 40)

    result_write_metric.close()

    return result_metric_dict


if __name__ == "__main__":
    # ----------------------------------------
    DATA_DIR = r"./data/BCICIV_2a_mat_raw/"
    EVALUATE_MODE = (
        "LOSO"  # leaving one subject out subject-dependent  subject-indenpedent
    )

    N_SUBJECT = 9  # BCI
    N_AUG = 3  # data augmentation times for benerating artificial training data set
    N_SEG = 8  # segmentation times for S&R

    EPOCHS = 600
    EMB_DIM = 16
    HEADS = 2
    DEPTH = 6
    TYPE = "A"
    validate_ratio = (
        0.3  # split raw train dataset into real train dataset and validate dataset
    )
    BATCH_SIZE = 512
    EEGNet1_F1 = 8
    EEGNet1_KERNEL_SIZE = 64
    EEGNet1_D = 2
    EEGNet1_POOL_SIZE1 = 8
    EEGNet1_POOL_SIZE2 = 8
    FLATTEN_EEGNet1 = 240

    if EVALUATE_MODE != "LOSO":
        EEGNet1_DROPOUT_RATE = 0.5
    else:
        EEGNet1_DROPOUT_RATE = 0.25

    parameters_list = [0, 1, 2]
    for i in parameters_list:
        number_class, number_channel = numberClassChannel(TYPE)
        RESULT_NAME = "Loso_{}_heads_{}_depth_{}_{}".format(TYPE, HEADS, DEPTH, i)

        sModel = EEGTransformer(
            heads=HEADS,
            emb_size=EMB_DIM,
            depth=DEPTH,
            database_type=TYPE,
            eeg1_f1=EEGNet1_F1,
            eeg1_D=EEGNet1_D,
            eeg1_kernel_size=EEGNet1_KERNEL_SIZE,
            eeg1_pooling_size1=EEGNet1_POOL_SIZE1,
            eeg1_pooling_size2=EEGNet1_POOL_SIZE2,
            eeg1_dropout_rate=EEGNet1_DROPOUT_RATE,
            eeg1_number_channel=number_channel,
            flatten_eeg1=FLATTEN_EEGNet1,
        ).cuda()
        summary(sModel, (1, number_channel, 1000))

        print(time.asctime(time.localtime(time.time())))

        result = main(
            RESULT_NAME,
            evaluate_mode=EVALUATE_MODE,
            heads=HEADS,
            emb_size=EMB_DIM,
            depth=DEPTH,
            dataset_type=TYPE,
            eeg1_f1=EEGNet1_F1,
            eeg1_kernel_size=EEGNet1_KERNEL_SIZE,
            eeg1_D=EEGNet1_D,
            eeg1_pooling_size1=EEGNet1_POOL_SIZE1,
            eeg1_pooling_size2=EEGNet1_POOL_SIZE2,
            eeg1_dropout_rate=EEGNet1_DROPOUT_RATE,
            flatten_eeg1=FLATTEN_EEGNet1,
            validate_ratio=validate_ratio,
            batch_size=BATCH_SIZE,
        )
        print(time.asctime(time.localtime(time.time())))
