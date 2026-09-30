import pandas as pd
import numpy as np
import torch
import os
import glob

import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .reading_utils.loading_utils import *
from .reading_utils.preprocessing_utils import *


class DataReader:
    """统一读取 UCR / UEA 等时序数据集的工具。

    参数说明
    ----------
    UCR_data_path : str
        UCR 数据根目录路径，内部会在其下拼接 ``UCRArchive_2018/``。
    UEA_data_path : str
        UEA 数据根目录路径。兼容两种目录结构：
        ``<UEA_data_path>/UEA/<dataset>/<dataset>_TRAIN.ts``，或
        ``<UEA_data_path>/<dataset>/<dataset>_TRAIN.ts``。
    transform_ts_size : int | None
        将所有时间序列统一插值到的长度（默认 512）。
        - 如果为 None，则不调整长度，保持原始序列长度。
    resize_func : callable | None
        自定义序列重采样函数，签名形如 ``resize_func(X: torch.Tensor) -> torch.Tensor``，
        其中 ``X`` 的形状为 ``[batch, channels, seq_len]``。
        - 若为 None，默认使用 ``torch.nn.functional.interpolate`` 做线性插值。
    univariate : bool
        若为 True，则把形状从 ``(n_samples, n_channels, seq_len)`` 重排为
        ``(n_samples * n_channels, 1, seq_len)``，即把多通道展开为多个单通道样本。
    channel_concat : bool
        若为 True（默认 False），则把多通道沿时间维拼接为单通道长序列：
        ``(n_samples, n_channels, seq_len) -> (n_samples, 1, n_channels * seq_len)``。
        随后再用 ``resize_func`` 将拼接后的长序列重采样到 ``transform_ts_size``。
        - 与 ``univariate`` 互斥：如果 ``univariate=True``，则不会启用 channel_concat。
    nan_fill_mode : str
        控制在读取 UEA ``.ts`` 文件时如何处理 NaN。
        目前支持：
        - ``"zero"`` （默认）：先将 NaN 用 0 填充（通过 ``set_nan_to_zero``），
          再统一对齐长度并线性插值到 ``transform_ts_size``。
        - 预留扩展位（例如 "mean"、"forward_fill" 等），后续可在 ``_read_ts_files``
          中根据该参数实现更复杂的插值策略。

    数据处理流程简要说明
    ----------------------
    对 UEA 数据集，调用 :meth:`_read_ts_files` 时会：

    1. 读取原始 ``.ts`` 文件，得到形状约为 ``[n_samples, n_channels, seq_len]`` 的张量 ``X``；
    2. 根据 ``nan_fill_mode`` 对 ``X`` 中的 NaN 做预处理（默认全部置 0）；
    3. 若设置了 ``univariate=True``，则展平通道维度；
    4. 使用 ``resize_func`` 将每个样本沿时间维线性插值 / 重采样到长度
       ``transform_ts_size``，从而得到对齐到统一长度的序列表示。
    """

    def __init__(self, UCR_data_path='/data/', UEA_data_path='/data2/', transform_ts_size=512,
                 resize_func=None, univariate=False, nan_fill_mode: str = "zero",
                 log_processing: bool = False, channel_concat: bool = False):
        self.UCR_data_path = UCR_data_path
        self.UEA_data_path = UEA_data_path
        self.UniTS_data_path = None
        self.transform_ts_size = transform_ts_size

        # 说明：默认重采样函数为沿时间维线性插值到 transform_ts_size 长度。
        # 也就是说，先保证每个样本的时序长度对齐，然后统一插值为指定长度。
        if resize_func is None:
            self.resize_func = lambda X: F.interpolate(X, size=self.transform_ts_size, mode='linear', align_corners=False)
        else:
            self.resize_func = resize_func

        # NaN 处理策略（目前在 _read_ts_files 中使用）
        self.nan_fill_mode = nan_fill_mode

        # 是否打印数据预处理方式（数据集名、NaN 处理方式、插值长度等）。
        self.log_processing = log_processing

        self.univariate = univariate
        self.channel_concat = channel_concat

        # Per-dataset label encoders (fit on train split, reused for test/val).
        # Keyed by base dataset name (e.g., "ECG200" for "ECG200:0").
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._get_dataset_lists()

    @staticmethod
    def _list_dataset_dirs(root: str) -> list[str]:
        if not os.path.isdir(root):
            return []
        return sorted(
            name
            for name in os.listdir(root)
            if os.path.isdir(os.path.join(root, name))
        )

    @staticmethod
    def _list_complete_ts_dataset_dirs(root: str) -> list[str]:
        dataset_names = []
        for name in DataReader._list_dataset_dirs(root):
            train_file = DataReader._resolve_ts_split_file(root, name, "train")
            test_file = DataReader._resolve_ts_split_file(root, name, "test")
            if train_file is not None and test_file is not None:
                dataset_names.append(name)
        return dataset_names

    @staticmethod
    def _resolve_ts_split_file(root: str, dataset_name: str, which_set: str) -> str | None:
        if which_set == "train":
            split = "TRAIN"
        elif which_set == "test":
            split = "TEST"
        else:
            raise KeyError("This collection has only train and test sets.")

        dataset_dir = os.path.join(root, dataset_name)
        standard_file = os.path.join(dataset_dir, f"{dataset_name}_{split}.ts")
        if os.path.isfile(standard_file):
            return standard_file

        candidates = sorted(glob.glob(os.path.join(dataset_dir, f"*_{split}.ts")))
        if len(candidates) == 1:
            return candidates[0]
        return None

    @staticmethod
    def _resolve_uea_root(uea_data_path: str) -> tuple[str, str]:
        """Return (base_path, collection) for UEA-style .ts datasets.

        Historical UEA archives are laid out as root/UEA/<dataset>.
        Multiverse-style benchmark exports can be laid out directly as
        root/<dataset>. The returned collection is passed to
        _read_ts_files and may be an empty string for the direct layout.
        """
        nested_root = os.path.join(uea_data_path, "UEA")
        if os.path.isdir(nested_root):
            return uea_data_path, "UEA"
        return uea_data_path, ""

    def _maybe_channel_concat(self, X: torch.Tensor) -> torch.Tensor:
        """可选：把多通道 (N,C,L) 拼接为单通道长序列 (N,1,C*L)。

        - 仅在 self.channel_concat=True 且 self.univariate=False 时启用。
        - C<=1 时为 no-op。
        """
        if (not self.channel_concat) or self.univariate:
            return X

        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)

        if X.ndim == 2:
            # (N, L) -> (N, 1, L)
            return X.unsqueeze(-2)

        if X.ndim != 3:
            return X

        n, c, l = X.shape
        if c <= 1:
            return X

        # (N, C, L) -> (N, 1, C*L)
        return X.reshape(n, 1, c * l)

    # def _get_dataset_lists(self,):
    #     self.dataset_list_ucr = os.listdir(self.data_path + "UCRArchive_2018/")
    #     self.dataset_list_uea = os.listdir(self.data_path + "UEA/")
    #     self.dataset_list_units = os.listdir(self.data_path + "UniTS/")
    #     self.dataset_list_others = os.listdir(self.data_path + "Others/")
    #     # self.dataset_list_forecasting = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'ExchangeRate', 
    #     #                                  'Illness', 'Traffic', 'Weather']
    #     self.dataset_list_forecasting = [
    #     'ETTh1_pred=24_feat=0', 'ETTh2_pred=24_feat=0', 
    #     'ETTm1_pred=96_feat=0', 'ETTm2_pred=96_feat=0',
    #     'Electricity_pred=24_feat=0', 'ExchangeRate_pred=30_feat=0',
    #     'Illness_pred=24_feat=0', 'Traffic_pred=24_feat=0', 'Weather_pred=24_feat=0'
    #     ]

    #     self.classification_datasets = self.dataset_list_ucr + self.dataset_list_uea + self.dataset_list_units + self.dataset_list_others
    def _get_dataset_lists(self,):
        ucr_root = os.path.join(self.UCR_data_path, "UCRArchive_2018")
        self.dataset_list_ucr = self._list_dataset_dirs(ucr_root)

        self._uea_base_path, self._uea_collection = self._resolve_uea_root(self.UEA_data_path)
        uea_root = os.path.join(self._uea_base_path, self._uea_collection)
        if self._uea_collection:
            self.dataset_list_uea = self._list_dataset_dirs(uea_root)
        else:
            self.dataset_list_uea = self._list_complete_ts_dataset_dirs(uea_root)
        
        # self.dataset_list_forecasting = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'ExchangeRate', 
        #                                  'Illness', 'Traffic', 'Weather']


        self.classification_datasets =  self.dataset_list_uea + self.dataset_list_ucr
    
    def _get_dataset_collection(self, dataset_name):
        if dataset_name in self.dataset_list_uea:
            return "UEA/"
        elif dataset_name in self.dataset_list_units:
            return "UniTS/"
        elif dataset_name in self.dataset_list_ucr:
            return "UCRArchive_2018/"
    
    def read_dataset(self, dataset_name, which_set='train'):
        """
        dataset_name:
            For multivariate data: if the name is of the form "name:idx", then it will return only channel idx in this dataset.
        which_set:
            Either 'train', 'val' or 'test'.
        """
        if ":" in dataset_name:
            dataset_name, channel_idx = dataset_name.split(":")
            channel_idx = int(channel_idx)
        else:
            channel_idx = None

        base_dataset_name = dataset_name

        # UCR
        if dataset_name in self.dataset_list_ucr:
            if self.log_processing:
                print(f"[DataReader] Loading UCR dataset '{dataset_name}' (set={which_set}), "
                      f"resize_to={self.transform_ts_size}, nan_fill_mode=not_applicable")
            data = self._read_ucr_dataset(dataset_name, which_set=which_set)
        # UEA
        elif dataset_name in self.dataset_list_uea:
            if self.log_processing:
                print(f"[DataReader] Loading UEA dataset '{dataset_name}' (set={which_set}), "
                      f"nan_fill_mode={self.nan_fill_mode}, resize_to={self.transform_ts_size}")
            # Use time-series reader for UEA datasets. For InsectWingbeatSubset we
            # will additionally subsample to a fixed-size split for benchmarking.
            read_func = self._read_npy_files if dataset_name == 'InsectWingbeatSubset' else self._read_ts_files
            data = read_func(
                self._uea_base_path,
                dataset_name,
                collection=self._uea_collection,
                which_set=which_set,
                channel_idx=channel_idx,
            )
        # UniTS
        # Datasets from UniTS paper
        elif dataset_name in self.dataset_list_units:
            data = self._read_ts_files(self.UniTS_data_path,dataset_name, collection='UniTS', which_set=which_set, channel_idx=channel_idx)
        # Other datasets used in pre-training
        elif dataset_name in self.dataset_list_others:
            data = self._read_other_dataset(dataset_name, which_set=which_set, channel_idx=channel_idx)
        # Forecasting datasets
        # They follow special convention:
        # The suffix "_pred={number}" specifies prediction horizon, 
        # The suffix "_feat={number1,number2,...}" specifies features to use for x
        # The suffix "_target={number1,number2,...}" specifies targets to use for y
        # To choose the sequence length for x, vary self.transform_ts_size
        # TODO: allow to choose time_increment
        elif np.any([dataset_name.startswith(ds_name) for ds_name in self.dataset_list_forecasting]):
            dataset_name_, pred_len, feature_idx, target_idx = process_forecasting_dataset_name(dataset_name)
            data = self._read_forecasting_dataset(dataset_name_, which_set=which_set, seq_len=self.transform_ts_size, pred_len=pred_len,
                                                  time_increment=1, feature_idx=feature_idx, target_idx=target_idx)
        # Some other datasets, not well investigated
        elif dataset_name == "ForecastBaseline":
            data = self._read_forecast_baseline_dataset() 
        elif dataset_name == 'SPO-PPG':
            data = self._read_spo_ppg_dataset()
        elif dataset_name == 'EEG':
            data = self._read_eeg_dataset() 
        else:
            raise KeyError('Unknown dataset name.')
        
        # Encode labels to 0...K-1 for classification datasets.
        # Important: fit encoder on train split only, reuse for test/val to avoid
        # inconsistent label-id mappings across splits.
        x, y = data
        if base_dataset_name in self.classification_datasets:
            if which_set == 'train':
                lab_encoder = LabelEncoder()
                y = lab_encoder.fit_transform(y)
                self._label_encoders[base_dataset_name] = lab_encoder
            else:
                lab_encoder = self._label_encoders.get(base_dataset_name)
                if lab_encoder is None:
                    # If test/val is loaded before train, proactively load the
                    # train split once to fit an encoder, then reuse it.
                    try:
                        _ = self.read_dataset(base_dataset_name, which_set='train')
                        lab_encoder = self._label_encoders.get(base_dataset_name)
                    except Exception:
                        lab_encoder = None

                if lab_encoder is None:
                    # Last-resort fallback (keeps prior behavior): fit on the
                    # current split if we still don't have an encoder.
                    lab_encoder = LabelEncoder()
                    y = lab_encoder.fit_transform(y)
                    self._label_encoders[base_dataset_name] = lab_encoder
                else:
                    try:
                        y = lab_encoder.transform(y)
                    except ValueError:
                        # Unseen labels in this split (unexpected for UCR/UEA).
                        # Fall back to per-split fitting to avoid hard failure.
                        lab_encoder = LabelEncoder()
                        y = lab_encoder.fit_transform(y)
                        self._label_encoders[base_dataset_name] = lab_encoder
        
        return x, y

    def _read_ucr_dataset(self, dataset_name, which_set):
        if which_set == 'train':
            filename_suffix = "_TRAIN.tsv"
        elif which_set == 'test':
            filename_suffix = "_TEST.tsv"
        else:
            raise KeyError('This collection has only train and test sets.')

        file_name = os.path.join(self.UCR_data_path + "UCRArchive_2018/", dataset_name, dataset_name + filename_suffix)
        data = pd.read_csv(file_name, sep='\t', header=None).to_numpy()
        X, y = torch.tensor(data[:, 1:], dtype=torch.float), data[:, 0]
        X = X.unsqueeze(-2)

        # optional: concatenate multi-channel into one long univariate series
        X = self._maybe_channel_concat(X)

        # interpolate time-series to self.transform_ts_size if not None
        if self.transform_ts_size is not None:
            X = self.resize_func(X)

        return X.numpy(), y
    
    def _read_npy_files(self, data_path, dataset_name, collection, which_set, channel_idx=None):
        if which_set == 'train':
            filename_suffix = '_train.npy'
        elif which_set == 'test':
            filename_suffix = '_test.npy'
        else:
            raise KeyError('This collection has only train and test sets.')

        base_path = os.path.join(data_path, collection, dataset_name)
        X, y = [np.load(base_path + '/' + s + filename_suffix) for s in ['x', 'y']]
        if self.nan_fill_mode == "zero":
            X = set_nan_to_zero(X)
        else:
            # 预留其他模式，例如按均值填充、前向填充等
            X = set_nan_to_zero(X)
        X = torch.tensor(X, dtype=torch.float)

        # select only this channel if specified
        if channel_idx is not None:
            X = X[:, [channel_idx], :]

        # optional: concatenate multi-channel into one long univariate series
        X = self._maybe_channel_concat(X)

        # reshape to univariate if specified
        if self.univariate:
            # flat the variable dimension
            X = X.reshape(-1, X.shape[-1])
            # insert back a variable dimension for convention
            X = X.unsqueeze(-2)
            # each entry of y is repeated the number of channels times
            y = np.repeat(y, X.shape[1])

        # interpolate time-series to self.transform_ts_size if not None
        if self.transform_ts_size is not None:
            X = self.resize_func(X)
        
        return X.numpy(), y
    

    def _read_ts_files(self, data_path,dataset_name, collection, which_set, channel_idx=None):
        root = os.path.join(data_path, collection)
        file_name = self._resolve_ts_split_file(root, dataset_name, which_set)
        if file_name is None:
            if which_set == 'train':
                filename_suffix = '_TRAIN.ts'
            elif which_set == 'test':
                filename_suffix = '_TEST.ts'
            else:
                raise KeyError('This collection has only train and test sets.')
            expected_file = os.path.join(root, dataset_name, dataset_name + filename_suffix)
            raise FileNotFoundError(f"Could not find a unique UEA split file for {dataset_name}: {expected_file}")

        label_dict = get_label_dict(file_name)
        X, y = get_data_and_label_from_ts_file(file_name, label_dict)

        # 先根据 nan_fill_mode 处理 NaN。
        # 当前仅实现 "zero"：将所有 NaN 置 0，便于后续插值；
        # 如需更复杂的插值策略，可在此分支中扩展。
        if self.nan_fill_mode == "zero":
            X = set_nan_to_zero(X)
        else:
            # 预留其他模式，例如按均值填充、前向填充等
            X = set_nan_to_zero(X)

        X = torch.tensor(X, dtype=torch.float)

        # select only this channel if specified
        if channel_idx is not None:
            X = X[:, [channel_idx], :]

        # optional: concatenate multi-channel into one long univariate series
        X = self._maybe_channel_concat(X)

        # reshape to univariate if specified
        if self.univariate:            
            # Each row of X is (instance-channel) entry from original 3d matrix
            # Each entry of y is repeated the number of channels times
            y = np.repeat(y, X.shape[1])
            # flat the variable dimension
            X = X.reshape(-1, X.shape[-1])
            # insert back a variable dimension for convention
            X = X.unsqueeze(-2)

        # interpolate time-series to self.transform_ts_size if not None
        if self.transform_ts_size is not None:
            X = self.resize_func(X)
        
        return X.numpy(), y

    # def _subsample_split(self, data, which_set, max_samples: int = 1000, base_seed: int = 12345):
    #     """Subsample X/y to at most `max_samples` rows using a fixed RNG seed.

    #     Keeps X/y aligned. `which_set` chooses an offset so train/test use different
    #     but deterministic seeds.
    #     """
    #     X, y = data
    #     X = np.asarray(X)
    #     y = np.asarray(y)

    #     total = len(y)
    #     if total <= max_samples:
    #         return X, y

    #     offset = {'train': 0, 'test': 1, 'val': 2}.get(which_set, 3)
    #     rng = np.random.default_rng(base_seed + offset)
    #     indices = rng.choice(total, size=max_samples, replace=False)

    #     return X[indices], y[indices]
        
    def _read_other_dataset(self, dataset_name, which_set, channel_idx=None):
        if which_set == 'train':
            filename_suffix = "train.pt"
        elif which_set == 'val':
            filename_suffix = "val.pt"
        else:
            filename_suffix = "test.pt"
        file_name = os.path.join(self.data_path + "Others/", dataset_name, filename_suffix)
        
        data = torch.load(file_name)
        X, y = data['samples'], data['labels']
        X = torch.tensor(X, dtype=torch.float)
        
        # select only this channel if specified
        if channel_idx is not None:
            X = X[:, [channel_idx], :]

        # optional: concatenate multi-channel into one long univariate series
        X = self._maybe_channel_concat(X)

        # reshape to univariate if specified
        if self.univariate:            
            # Each row of X is (instance-channel) entry from original 3d matrix
            # Each entry of y is repeated the number of channels times
            y = np.repeat(y, X.shape[1])
            # flat the variable dimension
            X = X.reshape(-1, X.shape[-1])
            # insert back a variable dimension for convention
            X = X.unsqueeze(-2)

        # interpolate time-series to self.transform_ts_size if not None
        if self.transform_ts_size is not None:
            X = self.resize_func(X)
        
        return X.numpy(), y
    
    def _read_forecasting_dataset(self, dataset_name, which_set, seq_len, pred_len, time_increment=1, feature_idx=None, target_idx=None):
        """

        """
        file_name = os.path.join(self.data_path, 'forecasting', dataset_name, dataset_name + '.csv')
        df_raw = pd.read_csv(file_name, index_col=0)

        # split train / valid / test
        n = len(df_raw)
        if 'ETTm' in dataset_name:
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif 'ETTh' in dataset_name:
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n
        
        train_df = df_raw[:train_end]
        val_df = df_raw[train_end - seq_len : val_end]
        test_df = df_raw[val_end - seq_len : test_end]

        # standardize by training set
        # according to Romain, TSMixer does not scale back, 
        # so MSE is reported on this scaled data
        scaler = StandardScaler()
        scaler.fit(train_df.values)

        train_df, val_df, test_df = [scaler.transform(df.values) for df in [train_df, val_df, test_df]]

        # train
        if which_set == 'train':
            x, y = construct_sliding_window_data(train_df, seq_len, pred_len, time_increment, feature_idx, target_idx)
        # val
        elif which_set == 'val':
            x, y = construct_sliding_window_data(val_df, seq_len, pred_len, time_increment, feature_idx, target_idx)
        # test
        else:
            x, y = construct_sliding_window_data(test_df, seq_len, pred_len, time_increment, feature_idx, target_idx)

        
        return x, y

    def _read_forecast_baseline_dataset(self):
        dataset_list=glob.glob(self.data_path+"/forecast_baseline_dataset/ALL_data/*")
        data=torch.tensor([],dtype=torch.float)
        for dataset in dataset_list[:]:
            if "txt" in dataset:
                continue
            data_file=glob.glob(dataset+"/*.csv")
            for file in data_file:
                ori_data=pd.read_csv(file)
                seg_length=ori_data.shape[0]
                while True:
                    if seg_length<64:
                        break
                    tmp=ori_data.iloc[:int(ori_data.shape[0]/seg_length)*seg_length,1:].to_numpy().T
                    tmp=tmp.reshape(-1,seg_length)
                    tmp=self.resize_func(size=(1, self.transform_ts_size),antialias=True)(torch.tensor(tmp, dtype=torch.float).unsqueeze(-2))
                    data=torch.cat((data,tmp),dim=0)
                    seg_length=int(seg_length/2)
        return data
      
    def _read_spo_ppg_dataset(self):
        dataset_list = glob.glob(self.data_path+"/ppg/osahs/*")
        data = torch.tensor([],dtype=torch.float)
        for dataset in dataset_list[:]:
            if "good" in dataset:
                continue
            tmp = torch.load(dataset).unsqueeze(dim=1)
            tmp = self.resize_func(size=(1, self.transform_ts_size),antialias=True)(torch.tensor(tmp.clone().detach(),dtype=torch.float))
            data = torch.cat((data, tmp),dim=0)
        return data
      
    def _read_eeg_dataset(self):
        file_list = ["test_data_onemodelTrue_onlyiiFalse_interTrue.npy",
                     "train_data_onemodelTrue_onlyiiFalse_interTrue.npy",
                     "train_data_icen.npy"]
        data = torch.tensor([],dtype=torch.float)
        for file in file_list[:]:
            tmp=self.resize_func(size=(1, self.transform_ts_size),antialias=True)(torch.tensor(np.load(self.data_path+"/ECG/"+file,allow_pickle=True), dtype=torch.float).unsqueeze(-2))
            data=torch.cat((data,tmp),dim=0)  
        return data
