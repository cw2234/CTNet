"""
CTNet: A Convolution-Transformer Network for EEG-Based Motor Imagery Classification

author: zhaowei701@163.com

Cite this work
Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024). https://doi.org/10.1038/s41598-024-71118-7

"""

import os

import numpy as np
import pandas as pd
import random
import datetime
import time
from pandas import ExcelWriter
import torch
import torch.nn as nn
from torchsummary import summary
from sklearn.model_selection import train_test_split

from utils import calMetrics
from utils import calculatePerClass
from utils import numberClassChannel
from utils import load_data_evaluate


from model import EEGTransformer


def set_seed(seed: int = 42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Experiment:
    def __init__(
        self,
        nsub,
        data_dir,
        result_name,
        epochs=2000,
        number_aug=2,
        number_seg=8,
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
        batch_size=72,  # each batch of raw train dataset, real training batchsize =  batch_size * (1 + N_AUG) for additional data augmentation.
    ):

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

        self.criterion_cls = nn.CrossEntropyLoss()

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
        ).to(device)

        self.model_filename = self.result_name + "/model_{}.pth".format(self.nSub)

    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        number_records_by_augmentation = self.number_augmentation * int(
            self.batch_size / self.number_class
        )
        number_segmentation_points = 1000 // self.number_seg
        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug)
            tmp_data = timg[cls_idx]

            tmp_aug_data = np.zeros(
                (number_records_by_augmentation, 1, self.number_channel, 1000),
                dtype=timg.dtype,
            )
            for ri in range(number_records_by_augmentation):
                rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                for rj in range(self.number_seg):
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
            aug_label.append(
                np.full(number_records_by_augmentation, clsAug, dtype=label.dtype)
            )
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data)
        aug_label = torch.from_numpy(aug_label)
        return aug_data, aug_label

    def prepare_train_val_test_data(self):
        """
        准备训练集、验证集和测试集，将数据z-score，用训练集的均值和标准差进行归一化
        """
        (
            all_data,  # (batch, channel, length)
            all_label,
            test_data,
            test_label,
        ) = load_data_evaluate(
            self.root, self.dataset_type, self.nSub, mode_evaluate=self.evaluate_mode
        )

        all_data = np.expand_dims(all_data, axis=1)  # (288, 1, 22, 1000)
        all_label = np.transpose(all_label)
        all_label = all_label[0]

        test_data = np.expand_dims(test_data, axis=1)
        test_label = np.transpose(test_label)
        test_label = test_label[0]

        # 划分训练集和验证集
        train_data, val_data, train_label, val_label = train_test_split(
            all_data,
            all_label,
            test_size=self.validate_ratio,
            shuffle=True,
            stratify=all_label,
        )
        print("-" * 20)

        print(
            "train size：",
            train_data.shape,
            "val size：",
            val_data.shape,
            "test size：",
            test_data.shape,
        )

        # standardize
        target_mean = np.mean(train_data)
        target_std = np.std(train_data)

        train_data = (train_data - target_mean) / target_std
        val_data = (val_data - target_mean) / target_std
        test_data = (test_data - target_mean) / target_std

        isSaveDataLabel = False  # True
        if isSaveDataLabel:
            np.save("./gradm_data/train_data_{}.npy".format(self.nSub), train_data)
            np.save("./gradm_data/train_lable_{}.npy".format(self.nSub), train_label)
            np.save("./gradm_data/val_data_{}.npy".format(self.nSub), val_data)
            np.save("./gradm_data/val_lable_{}.npy".format(self.nSub), val_label)
            np.save("./gradm_data/test_data_{}.npy".format(self.nSub), test_data)
            np.save("./gradm_data/test_label_{}.npy".format(self.nSub), test_label)

        # data shape: (trial, conv channel, electrode channel, time samples)
        train_data = train_data.astype(np.float32)
        test_data = test_data.astype(np.float32)
        val_data = val_data.astype(np.float32)
        train_label = train_label.astype(np.int64)
        train_label = train_label - 1
        val_label = val_label.astype(np.int64)
        val_label = val_label - 1
        test_label = test_label.astype(np.int64)
        test_label = test_label - 1

        return train_data, train_label, val_data, val_label, test_data, test_label

    def prepare_dataloader(self, data: np.ndarray, label: np.ndarray, shuffle):
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(data), torch.from_numpy(label)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=shuffle
        )
        return dataloader

    def start_train(self):
        (
            train_data_np,
            train_label_np,
            val_data_np,
            val_label_np,
            test_data_np,
            test_label_np,
        ) = self.prepare_train_val_test_data()
        # print("label size:", label.shape)
        # print("label size:", label)

        train_dataloader = self.prepare_dataloader(
            train_data_np, train_label_np, shuffle=True
        )

        val_dataloader = self.prepare_dataloader(
            val_data_np, val_label_np, shuffle=False
        )

        test_dataloader = self.prepare_dataloader(
            test_data_np, test_label_np, shuffle=False
        )

        # Optimizers
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2)
        )

        best_epoch = 0
        min_loss = 1000
        # recording train_acc, train_loss, test_acc, test_loss
        result_process = []
        # Train the cnn model
        for epoch in range(self.n_epochs):
            epoch_process = {}
            epoch_process["epoch"] = epoch
            # in_epoch = time.time()
            self.model.train()

            # train model
            train_acc, train_loss = self.train_epoch(
                train_dataloader, train_data_np, train_label_np
            )
            epoch_process["train_acc"] = train_acc
            epoch_process["train_loss"] = train_loss

            # validate model
            val_acc, val_loss = self.validate_epoch(val_dataloader)

            epoch_process["val_acc"] = val_acc
            epoch_process["val_loss"] = val_loss

            # if min_loss>val_loss:
            if min_loss > val_loss:
                min_loss = val_loss
                best_epoch = epoch
                epoch_process["epoch"] = epoch
                torch.save(self.model, self.model_filename)
                print(
                    f"{self.nSub}_{epoch_process['epoch']} train_acc: {epoch_process['train_acc']:.4f} train_loss: {epoch_process['train_loss']:.6f}\tval_acc: {epoch_process['val_acc']:.6f} val_loss: {epoch_process['val_loss']:.7f}"
                )

            result_process.append(epoch_process)

        # test model
        test_acc, y_pred = self.evaluate(test_dataloader)
        print("epoch: ", best_epoch, "\tThe test accuracy is:", test_acc)

        df_process = pd.DataFrame(result_process)

        return test_acc, torch.from_numpy(test_label_np), y_pred, df_process, best_epoch
        # writer.close()

    def train_epoch(self, train_dataloader, train_data_np, train_label_np):
        """训练模型，每个epoch的"""
        # in_epoch = time.time()
        self.model.train()

        correct = 0
        total = 0
        train_loss = 0
        for i, (batch_x, batch_y) in enumerate(train_dataloader):
            # split raw train dataset into real train dataset and validate dataset
            train_data = batch_x
            train_label = batch_y

            # data augmentation
            aug_data, aug_label = self.interaug(train_data_np, train_label_np)
            # concat real train dataset and generate aritifical train dataset
            train_data = torch.cat((train_data, aug_data))
            train_label = torch.cat((train_label, aug_label))
            train_data = train_data.to(device)
            train_label = train_label.to(device)

            # training model
            features, outputs = self.model(train_data)

            loss = self.criterion_cls(outputs, train_label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录train acc, train loss
            train_loss += loss.item() * train_label.size(0)
            _, train_pred = torch.max(outputs, 1)
            correct += (train_pred == train_label).sum().item()
            total += train_label.size(0)

        train_acc = correct / total
        train_loss = train_loss / total

        return train_acc, train_loss

    def validate_epoch(self, val_dataloader):
        """验证模型，每个epoch的验证集准确率和损失"""

        self.model.eval()

        outputs_list = []
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_dataloader):
                # val model
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                _, logits = self.model(batch_x)

                val_loss += self.criterion_cls(logits, batch_y).item() * batch_y.size(0)

                _, val_pred = torch.max(logits, 1)
                correct += (val_pred == batch_y).sum().item()
                total += batch_y.size(0)

        val_acc = correct / total
        val_loss = val_loss / total

        return val_acc, val_loss

    def evaluate(self, test_dataloader):
        """测试模型"""
        # load model for test
        self.model = torch.load(self.model_filename, weights_only=False).to(device)
        self.model.eval()

        correct = 0
        total = 0
        y_pred_list = []
        with torch.no_grad():
            for i, (img, label) in enumerate(test_dataloader):
                img_test = img.to(device).float()
                label_test = label.to(device)

                # test model
                features, outputs = self.model(img_test)
                _, pred = torch.max(outputs, 1)
                y_pred_list.append(pred)
                correct += (pred == label_test).sum().item()
                total += label_test.size(0)

        test_acc = correct / total
        y_pred = torch.cat(y_pred_list, dim=0)

        return test_acc, y_pred


def main(
    dirs,
    evaluate_mode="subject-dependent",  # "LOSO" or other
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

    # 提前生成种子列表
    np.random.seed(MAIN_SEED)
    seed_list = [np.random.randint(2024) for i in range(N_SUBJECT)]
    for i in range(N_SUBJECT):
        starttime = datetime.datetime.now()
        seed_n = seed_list[i]
        print("seed is " + str(seed_n))
        set_seed(seed_n)
        index_round = 0
        print("Subject %d" % (i + 1))
        exp = Experiment(
            i + 1,
            DATA_DIR,
            dirs,
            EPOCHS,
            N_AUG,
            N_SEG,
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
        )

        testAcc, Y_true, Y_pred, df_process, best_epoch = exp.start_train()
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
        "LOSO-No"  # leaving one subject out subject-dependent  subject-indenpedent
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    N_SUBJECT = 9  # BCI
    N_AUG = 3  # data augmentation times for generating artificial training data set
    N_SEG = 8  # segmentation times for S&R

    MAIN_SEED = 42
    EPOCHS = 1000
    EMB_DIM = 16
    HEADS = 2
    DEPTH = 6
    TYPE = "A"
    validate_ratio = (
        0.3  # split raw train dataset into real train dataset and validate dataset
    )

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

    number_class, number_channel = numberClassChannel(TYPE)
    RESULT_NAME = "{}_heads_{}_depth_{}".format(TYPE, HEADS, DEPTH)

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
    ).to(device)
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
    )
    print(time.asctime(time.localtime(time.time())))
