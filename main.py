from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.Llama2 import Llama2
from models.Gemma2 import Gemma2
from models.Llama3 import Llama3


import numpy as np
import torch
import torch.nn as nn
from torch import optim
import gc
from torch.cuda.amp import GradScaler, autocast

import os
import time
import psutil

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--fix_seed', type=int, default=2021)
parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--llama_layers', type=int, default=6)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)

parser.add_argument('--plt', type=int, default=0)
parser.add_argument('--read_model', type=int, default=0)
parser.add_argument('--write_model', type=int, default=0)
parser.add_argument('--if_inverse', type=int, default=0)
parser.add_argument('--order', type=int, default=0)
parser.add_argument('--fc_layer', type=int, default=512)

args = parser.parse_args()

fix_seed = args.fix_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []
corr_s, corr_p = [], []
p_s, p_p = [], []

process = psutil.Process()

for ii in range(args.itr):
    device = torch.device('cuda:0')

    start_time = time.time()
    cpu_mem_start = process.memory_info().rss / (1024 * 1024)
    if torch.cuda.is_available():
        gpu_mem_start_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
        gpu_mem_start_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    else:
        gpu_mem_start_alloc, gpu_mem_start_reserved = 0, 0

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.freq == 0:
        args.freq = 'h'

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))
    
    # device = torch.device('cpu')

    time_now = time.time()
    train_steps = len(train_loader)

    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(args, device)
        model.to(device)
    elif args.model == 'Llama2':
        model = Llama2(args, device)
        # model.to(device)
    elif args.model =='Gemma2':
        model = Gemma2(args, device)
    elif args.model == 'Llama3':
        model = Llama3(args, device)
    else:
        model = GPT4TS(args, device)

    if args.read_model:
        # print('\nLoading Model\n')
        pre_train_path = './PretrainModel/weather2/temp.pth'
        state_dict_model = torch.load(pre_train_path)

        # Check for any unexpected keys in the state dictionary
        unexpected_keys = [k for k in state_dict_model.keys() if k not in model.state_dict()]
        if unexpected_keys:
            print(f"Unexpected keys in state_dict: {unexpected_keys}")
        else:
            print("MATCH")

        # Filter out unexpected keys and load the state dictionary
        filtered_state_dict = {k: v for k, v in state_dict_model.items() if k in model.state_dict()}
        # print('filtered dict', filtered_state_dict)
        model.load_state_dict(filtered_state_dict, strict=False)
        print('Loading completed')

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
    
    scaler = GradScaler()
    accumulation_steps = 4  
    for epoch in range(args.train_epochs):

        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()

            with autocast():
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.to(device)

                outputs = model(batch_x, ii)

                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)
                loss = criterion(outputs, batch_y) / accumulation_steps
                train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                scaler.scale(loss).backward()
                if (i+1) % accumulation_steps == 0:
                    scaler.step(model_optim)
                    scaler.update()
                
                del batch_x, batch_y, outputs
                torch.cuda.empty_cache()

        
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
        #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))
        
        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        
        # earlystopping
        # early_stopping(vali_loss, model, path)
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break
    
    # best_model_path = path + '/' + 'checkpoint.pth'
    # model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
    print("------------------------------------")
    mse, mae, corr1, corr2 = test(model, test_data, test_loader, args, device, ii)

    end_time = time.time()
    cpu_mem_end = process.memory_info().rss / (1024 * 1024)
    if torch.cuda.is_available():
        gpu_mem_end_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
        gpu_mem_end_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    else:
        gpu_mem_end_alloc, gpu_mem_end_reserved = 0, 0
    
    print(f"Iteration {ii + 1}/{args.itr} completed in {end_time - start_time:.2f} seconds")
    print(f"CPU Memory Usage: Start {cpu_mem_start:.2f} MB, End {cpu_mem_end:.2f} MB")
    print(f"GPU Memory Usage: Allocated Start {gpu_mem_start_alloc:.2f} MB, End {gpu_mem_end_alloc:.2f} MB")
    print(f"GPU Memory Usage: Reserved Start {gpu_mem_start_reserved:.2f} MB, End {gpu_mem_end_reserved:.2f} MB")

    del model
    del model_optim
    del test_loader
    gc.collect()
    torch.cuda.empty_cache()


if args.write_model:
    # save_path = './PretrainModel/' + args.model_id.split('_')[0] + '_' + args.model_id.split('_')[1]
    save_path = './PretrainModel/' + args.model_id.split('_')[0] 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + '/' + args.model + '-' + 'temp.pth')

mses = np.array(mses)
maes = np.array(maes)
corr_s = np.array(corr_s)
corr_p = np.array(corr_p)
p_s = np.array(p_s)
p_p = np.array(p_p)

print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}\n".format(np.mean(maes), np.std(maes)))
print("spearmanR_mean = {:.4f}, spearmanP_mean = {:.4f}".format(np.mean(corr_s), np.mean(p_s)))
print("pearsonR_mean = {:.4f},pearsonP_mean = {:.4f}".format(np.mean(corr_p), np.mean(p_p)))