import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy import stats
from isoweek import Week
from torch.cuda.amp import autocast

from datetime import datetime
from distutils.util import strtobool
import pandas as pd

from utils.metrics import metric, MAE, MSE, MAPE, SMAPE

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    # if args.decay_fac is None:
    #     args.decay_fac = 0.5
    # if args.lradj == 'type1':
    #     lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    # elif args.lradj == 'type2':
    #     lr_adjust = {
    #         2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
    #         10: 5e-7, 15: 1e-7, 20: 5e-8
    #     }
    if args.lradj =='type1':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj =='type2':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch - 1) // 1))}
    elif args.lradj =='type4':
        lr_adjust = {epoch: args.learning_rate * (args.decay_fac ** ((epoch) // 1))}
    else:
        args.learning_rate = 1e-4
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    print("lr_adjust = {}".format(lr_adjust))
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(date, true, preds, r1=None, r2=None, name='./pic/test.pdf', model='GPT2'):
    """
    Results visualization
    """
    if model == 'GPT4TS':
        model = 'GPT2'
        
    path = os.path.dirname(name)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} successfully created')
    else:
        print(f'{path} already exists')

    _d = date.values

    plt.figure(figsize=(12, 4))
    if preds is not None:
        plt.plot(_d, preds*100, label='Prediction', linewidth=2)
    plt.plot(_d, true*100, label='GroundTruth', linewidth=2)
    if r1 is not None and r2 is not None:
        plt.title('spearmanR: {:.3f}    pearsonR: {:.3f}'.format(r1, r2))
    else:
        plt.title(f"{model}")

    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 20))

    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close()


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )

def vali(model, vali_data, vali_loader, criterion, args, device, itr):
    total_loss = []
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.eval()
    else:
        model.in_layer.eval()
        model.out_layer.eval() ## 每次改名
    with torch.no_grad():
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.to(device)
            batch_loss = 0
            # batch_x_mark = batch_x_mark.float().to(device)
            # batch_y_mark = batch_y_mark.float().to(device)

            with autocast():
                # for t in range(1, batch_x.shape[1]):
                # sub_x = batch_x[:, :t, :]
                # target = batch_x[:, t, :]
                outputs = model(batch_x, itr)
                # outputs = model(batch_x, itr, batch_x_mark)
                # outputs = model(sub_x, itr)
                
                # encoder - decoder
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                # true = target.detach().cpu()

                loss = criterion(pred, true)
                batch_loss += loss.item()

                total_loss.append(batch_loss)

    total_loss = np.average(total_loss)
    if args.model == 'PatchTST' or args.model == 'DLinear' or args.model == 'TCN':
        model.train()
    else:
        model.in_layer.train()
        model.out_layer.train()
        model.leaky_relu.train() 
    return total_loss

def MASE(x, freq, pred, true):
    masep = np.mean(np.abs(x[:, freq:] - x[:, :-freq]))
    return np.mean(np.abs(pred - true) / (masep + 1e-8))

def inverse_diff2(DF, start, end, _diff1, _df): ##
    """
    Restore the predicted results to the state before differencing.

    DF: Sequence to be restored
    _df: Original sequence
    _diff1: Original first-order differenced sequence
    start: Start time corresponding to the prediction start point
    end: End time corresponding to the prediction end point
    """
    seq_len = DF.shape[0]
    matches1 = np.where(_diff1.index==start)[0]
    if len(matches1) != 1:
        if len(matches1) == 0:
            raise ValueError(f'matches1 missing, _diff1 miss date:{start}')
        else:
            raise ValueError(f'matches1 cannot find a unique match, find {len(matches1)} matches')
    idx1 =  matches1.item()
    inverse1 = DF + _diff1[idx1-52:idx1-52+seq_len].values.reshape(-1)

    matches2 = np.where(_df.index==start)[0]
    if len(matches2) != 1:
        if len(matches2) == 0:
            raise ValueError(f'matches2 missing, _df miss date:{start}')
        else:
            raise ValueError(f'matches2 cannot find a unique match, find {len(matches2)} matches')
    idx2 = matches2.item()
    inverse2 = inverse1 + _df[idx2-1:idx2-1+seq_len].values.reshape(-1)

    df_inverse = _df[start:end].copy()
    df_inverse['positive_rate'] = inverse2

    return df_inverse

def inverse_diff1(DF, start, end, _df): ##
    """
    Restore the predicted results to the state before differencing.

    DF: Sequence to be restored
    _df: Original sequence
    start: Start time corresponding to the prediction start point
    end: End time corresponding to the prediction end point
    """
    seq_len = DF.shape[0]

    matches = np.where(_df.index==start)[0]
    if len(matches) != 1:
        if len(matches) == 0:
            raise ValueError(f'matches missing, _df miss date:{start}')
        else:
            raise ValueError(f'matches cannot find a unique match, find {len(matches)} matches')
    idx = np.where(_df.index==start)[0].item()
    Inverse = DF + _df[idx-1:idx-1+seq_len].values.reshape(-1)

    df_inverse = _df[start:end].copy()
    df_inverse['num'] = Inverse

    return df_inverse

def change_date(df): ##
    """
    combine year and week
    """
    date = []
    year = df['year'].values
    week = df['week'].values
    
    for t in range(len(year)):
        dt = Week(int(year[t]), int(week[t])).monday()
        date.append(str(dt))
    
    df_new = pd.DataFrame([])
    df_new['date'] = date
    df_new['positive_rate'] = df.iloc[:, -1]
    
    df_new['date'] = pd.to_datetime(df_new['date'])
    df_new = df_new.set_index('date')

    return df_new

def test(model, test_data, test_loader, args, device, itr):
    preds = []
    trues = []
    X = []
    Date = []
    # mases = []
    ## CNCDC
    if args.if_inverse==1:
        data = pd.read_excel('/data_disk/lichx/CN_CDC/data.xlsx', sheet_name=args.order) # south, north, usa(flu), south, north(ILI) 
        df = change_date(data)
        diff1 = df.diff(1).dropna()
        diff2 = diff1.diff(52).dropna() # df: index = date, values = flu+%_diff
     ## CQCDC
    elif args.if_inverse==2:
        df = pd.read_csv('/data_disk/lichx/CQ_CDC/2010-2023流感数据/Weekly/Weekly_pre.csv',
                        parse_dates=['date'], index_col='date')
    elif args.if_inverse==3:
        df = pd.read_csv('/data_disk/lichx/CQ_CDC/2010-2023流感数据/positive_rate_pre.csv',
                        parse_dates=['date'], index_col='date')
        diff1 = df.diff(1).dropna()
        diff2 = diff1.diff(52).dropna()
    model.eval()
    with torch.no_grad():
        # print('**TEST**')
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, date) in enumerate(test_loader): ##

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.to(device)
            
            x = batch_x.detach().cpu().numpy()
            date = date.detach().cpu().numpy() ##

            with autocast():
                outputs = model(batch_x, itr)

            # encoder - decoder
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()

            # x = test_data.inverse_transform(x)
            # pred = test_data.inverse_transform(pred)
            # true = test_data.inverse_transform(true)
            
            preds.append(pred)
            trues.append(true)
            X.append(x)
            Date.append(date) ##

    preds = np.array(preds)
    trues = np.array(trues)
    X = np.array(X)
    Date = np.array(Date) ##
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    X = np.concatenate(X, axis=0)
    Date = np.concatenate(Date, axis=0)
    
    # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    # X = X.reshape(-1, X.shape[-2], X.shape[-1])
    # Date = Date.reshape(-1, Date.shape[-2], Date.shape[-1])

    corr_sperman, corr_pearson = [], []
    p_spearman, p_pearson = [], []
    for j in range(preds.shape[0]):

        dt = [f"{year}-{month}-{day}" for year, month, day in Date[j, :, :].astype(int)]
        start = dt[-args.pred_len]
        end = dt[-1]
        ## CNCDC
        if args.if_inverse==1 or args.if_inverse==3:
            preds[j,:,0] = inverse_diff2(preds[j,:,0], start=start, end=end, _diff1=diff1, _df=df).values.reshape(-1)
            trues[j,:,0] = inverse_diff2(trues[j,:,0], start=start, end=end, _diff1=diff1, _df=df).values.reshape(-1)
            ## start到end索引时包括end
            X[j,:,0] = inverse_diff2(X[j,:,0], start=dt[0], end=dt[-args.pred_len-1], _diff1=diff1, _df=df).values.reshape(-1)
        ## CQCDC
        elif args.if_inverse==2:
            preds[j,:,0] = inverse_diff1(preds[j,:,0], start=start, end=end, _df=df).values.reshape(-1)
            trues[j,:,0] = inverse_diff1(trues[j,:,0], start=start, end=end, _df=df).values.reshape(-1)
            X[j,:,0] = inverse_diff1(X[j,:,0], start=dt[0], end=dt[-args.pred_len-1], _df=df).values.reshape(-1)

        if (args.if_inverse == 1 and end == '2020-3-23') or \
            (args.if_inverse == 2 and end == '2020-1-7') or \
            (args.if_inverse == 3 and end == '2020-1-6'):
            print('Last 13 week')
            print('seq_len: ', args.seq_len, '\tif_inverse: ', args.if_inverse)
            print('pred: ', preds[j,:,:].reshape(-1))
            folder_path = f"/data_disk/lichx/NeurIPS2023-One-Fits-All/Long-term_Forecasting/Visual_relu_new/{args.data_path.split('.')[0]}/{args.model}/"
            dt2 = [f"{year}-{month}-{day}" for year, month, day in Date[j, :, :].astype(int)] # 112, 3 -> 112, 1
            dt2 = pd.to_datetime(dt2)
            GT = np.concatenate((X[j, :, 0], trues[j, :, 0]), axis=0).reshape(-1,1) ## ground truth 第i个样本的第一个特征
            PD = np.concatenate((X[j, :, 0], preds[j, :, 0]), axis=0).reshape(-1,1) ## predicted 第i个样本的第一个特征

            # visual(dt2, GT, PD, name=os.path.join(folder_path, 'if_inverse-'+f'{args.if_inverse}_' + 'last_13' + '.pdf'), model=args.model)
            if args.pred_len > 1:
                cor1, _ = stats.spearmanr(preds[j,:,:].reshape(-1), trues[j,:,:].reshape(-1))
                cor2, _ = stats.pearsonr(preds[j,:,:].reshape(-1), trues[j,:,:].reshape(-1))
            else:
                cor1, cor2 = np.nan, np.nan
            mae, mse, rmse, mape, mspe, smape, nd = metric(preds[j,:,:], trues[j,:,:])
            print('spearmanR:{:.5f}\tpearsonR:{:.5f}'.format(cor1, cor2))
            print('MAE:{:.5f} \t MSE:{:.5f}'.format(mae, mse))
            print('MAPE:{:.5f} \t SMAPE:{:.5f}\n'.format(mape, smape))

        if args.pred_len > 1:
            corr_s, _ = stats.spearmanr(preds[j, :, :].reshape(-1), trues[j, :, :].reshape(-1))
            corr_p, _ = stats.pearsonr(preds[j, :, :].reshape(-1), trues[j, :, :].reshape(-1))
        
            corr_sperman.append(corr_s)
            corr_pearson.append(corr_p)
            
        corr1 = np.nanmean(corr_sperman)
        corr2 = np.nanmean(corr_pearson)

    mae, mse, rmse, mape, mspe, smape, nd = metric(preds, trues)
    
    print('seq_len: ', args.seq_len, '\tif_inverse: ', args.if_inverse)
    print('spearmanR:{:.4f}, pearsonR:{:.4f}'.format(corr1, corr2)) 
    print('mae:{:.4f}, mse:{:.4f}, mape:{:.4f}, smape:{:.4f}\n'.format(mae, mse, mape, smape))

    if args.plt != 0:
        folder_path = f"/data_disk/lichx/NeurIPS2023-One-Fits-All/Long-term_Forecasting/Visual_relu_new/{args.data_path.split('.')[0]}/{args.model}/"
            
        for k in range(0, preds.shape[0], preds.shape[0] // 10):
            dt2 = [f"{year}-{month}-{day}" for year, month, day in Date[k, :, :].astype(int)]
            dt2 = pd.to_datetime(dt2)
            GT = np.concatenate((X[k, :, 0], trues[k, :, 0]), axis=0)
            PD = np.concatenate((X[k, :, 0], preds[k, :, 0]), axis=0)

            # GT = test_data.inverse_transform(GT.reshape(1, -1))
            # PD = test_data.inverse_transform(PD.reshape(1, -1))
            GT = GT.reshape(-1,1)
            PD = PD.reshape(-1,1)

            visual(dt2, GT, PD, corr_sperman[k], corr_pearson[k], os.path.join(folder_path, f'if_inverse-{args.if_inverse}_' + f'pred_len-{args.pred_len}_' + f'epochs-{args.train_epochs}_' + str(k) + '.pdf'))
        
    return mse, mae, corr1, corr2
