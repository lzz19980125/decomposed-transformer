import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from model import *
from data_loader import *
from torch.backends import cudnn
# import wandb
from sklearn.metrics import mean_squared_error
from deepod.metrics import ts_metrics
from deepod.metrics import point_adjustment
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def plotter_recon(recon_list, forecast, forecast_label_list ,test_labels, start, end):
    recon_list = recon_list[window_size:, :]
    forecast_label_list = forecast_label_list[:-window_size,:]
    for i in range(recon_list.shape[1]):
        plt.figure(figsize=(12, 4))
        # plt.plot(recon_list[window_size:,i],label='recon')
        # plt.plot(forecast_label_list[:-window_size,i],label='input')
        # plt.ylim([np.min(recon_list[window_size:, i]), np.max(recon_list[window_size:, i])])
        plt.plot(forecast_label_list[start:end,i],label='input')
        plt.plot(recon_list[start:end, i], label='recon')
        plt.ylim([np.min(recon_list[start:end:,i]),np.max(recon_list[start:end:,i])])
        plt.legend()
        plt.show()
        plt.close()

def plotter_forecast(recon_list, forecast, forecast_label_list ,test_labels, start, end):
    forecast = forecast[:-window_size,:]
    forecast_label_list = forecast_label_list[:-window_size, :]
    for i in range(recon_list.shape[1]):
        plt.figure(figsize=(12, 4))
        # plt.plot(forecast[:-window_size,i],label='forecast')
        # plt.plot(forecast_label_list[:-window_size,i],label='input')
        # plt.ylim([np.min(forecast[:-window_size,i]),np.max(forecast[:-window_size,i])])
        plt.plot(forecast_label_list[start:end, i], label='input')
        plt.plot(forecast[start:end,i],label='forecast')
        plt.ylim([np.min(forecast[start:end,i]),np.max(forecast[start:end,i])])
        plt.legend()
        plt.show()
        plt.close()

def vali(model, test_loader,lambda_):
    model.eval()

    recon_list = []
    predict_list = []
    total_list = []
    for i, (input_data, _) in enumerate(test_loader):
        input = input_data[:,:-1,:].float().to(device)
        forecast_label = input_data[:,-1,:].float().to(device)
        recons, predictions = model(input)

        mse_loss = torch.nn.MSELoss(reduction='none')
        loss_per_element = mse_loss(recons, input)
        recon_loss = torch.mean(
            torch.sqrt((loss_per_element.sum(dim=-1)).sum(dim=-1) / (input_data.shape[-1] * input_data.shape[-2])))
        recon_list.append(recon_loss.item())

        predict_loss_pre_element = torch.sqrt(
            mse_loss(predictions, forecast_label).sum(dim=-1) / forecast_label.shape[-1])
        predict_loss = torch.mean(predict_loss_pre_element)
        predict_list.append(predict_loss.item())
        total_loss = predict_loss + lambda_ * recon_loss
        total_list.append(total_loss.item())

    return np.average(predict_loss.item()), np.average(recon_loss.item()), np.average(total_list)

def build_model(n_features, window_size, out_dim, kernel_size, gru_n_layers, forecast_n_layers, forecast_hid_dim, dropout, lr):
    model = FREQ_ATT(n_features=n_features, window_size=100, out_dim=n_features, kernel_size=3, gru_n_layers=1,
                     forecast_n_layers=1, forecast_hid_dim=150, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if torch.cuda.is_available():
        model.cuda()
    return model, optimizer

def save_model(model, path):
    # if not os.path.exists(path):
    #     os.makedirs(path)
    torch.save(model.state_dict(), path + 'model.pth')


def train(model, train_loader, vali_loader, num_epochs, lambda_, device, criterion, optimizer, dataset_name):
    print("======================TRAIN MODE======================")
    time_now = time.time()
    train_steps = len(train_loader)

    for epoch in range(num_epochs):
        iter_count = 0
        recon_list = []
        predict_list = []
        total_list = []

        epoch_time = time.time()
        model.train()
        for i, (input_data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            iter_count += 1
            input = input_data[:,:-1,:].float().to(device)
            forecast_label = input_data[:,-1,:].float().to(device)
            recons, predictions = model(input)

            mse_loss = torch.nn.MSELoss(reduction='none')
            loss_per_element = mse_loss(recons, input)
            recon_loss = torch.mean(torch.sqrt((loss_per_element.sum(dim=-1)).sum(dim=-1) / (input_data.shape[-1] * input_data.shape[-2])))
            recon_list.append(recon_loss.item())

            predict_loss_pre_element = torch.sqrt(mse_loss(predictions, forecast_label).sum(dim=-1)/forecast_label.shape[-1])
            predict_loss = torch.mean(predict_loss_pre_element)
            predict_list.append(predict_loss.item())
            total_loss = predict_loss + lambda_ * recon_loss
            total_list.append(total_loss.item())

            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((num_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            total_loss.backward()
            optimizer.step()

        vali_pre_loss, vali_recons_loss, vali_total_loss = vali(model, vali_loader,lambda_)
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(total_list)
        print(
            "Epoch: {0}, Steps: {1} | Train pre_Loss: {2:.7f} Train recons_Loss: {3:.7f}  | Vali pre_Loss: {4:.7f} Vali recons_Loss: {5:.7f} | Train total_Loss: {6:.7f} Vali total_Loss: {7:.7f}".format(
                epoch + 1, train_steps, np.average(predict_list), np.average(recon_list), vali_pre_loss, vali_recons_loss, train_loss, vali_total_loss))
    save_model(model, f'./checkpoints/{dataset_name}')

def test(model,dataset_name, beta, test_loader, device, window_size):
    model.load_state_dict(
        torch.load(f'./checkpoints/{dataset_name}'+'model.pth'))
    model.eval()

    print("======================TEST MODE======================")
    criterion = nn.L1Loss(reduce=False)
    anomaly_score = []
    test_labels = []
    forecast = []
    recon_list = []
    forecast_label_list = []
    recon_loss = []
    for i, (input_data, labels) in enumerate(test_loader):
        input = input_data.float().to(device)
        forecast_label = input_data[:,-1,:].float().to(device)
        recons, _ = model(input[:,1:,:])
        _, predictions = model(input[:,:-1,:])
        point_loss = torch.mean(criterion(input[:,-1,:], recons[:,-1,:]), dim=-1)

        recon_list.append(recons[:,-1,:].detach().cpu().numpy())
        test_labels.append(labels[:,-1].detach().cpu().numpy())
        forecast_label_list.append(forecast_label.detach().cpu().numpy())
        forecast.append(predictions.detach().cpu().numpy())
        recon_loss.append(point_loss.detach().cpu().numpy())

    recon_list = np.concatenate(recon_list, axis=0)
    forecast = np.concatenate(forecast, axis=0)
    forecast_label_list = np.concatenate(forecast_label_list,axis=0)
    forecast_loss = np.mean(np.abs(forecast - forecast_label_list),axis=1)
    recon_loss = np.concatenate(recon_loss, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    anomaly_score = forecast_loss + beta * recon_loss

    # plotter_recon(recon_list, forecast, forecast_label_list, test_labels, 23000, 28000)
    # plotter_forecast(recon_list, forecast, forecast_label_list, test_labels, 23000, 28000)
    return anomaly_score, test_labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lambda_', type=int, default=1)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--drop_out', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()
    args = vars(config)

    dataset = config.dataset
    batch_size = config.batch_size
    window_size = config.win_size + 1
    lr = config.lr
    num_epochs = config.num_epochs
    lambda_ = config.lambda_
    data_path = f'./datasets/{dataset}/'
    train_flag = config.mode == 'train'

    train_loader, _ = get_loader_segment(data_path, batch_size=batch_size, win_size=window_size,mode='train',dataset=dataset)
    test_loader, scaler = get_loader_segment(data_path, batch_size=batch_size, win_size=window_size,mode='test', dataset=dataset)

    n_features = train_loader.dataset.train.shape[1]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()

    model, optimizer = build_model(n_features, config.win_size, n_features, 3, 1, 1, 150, config.drop_out, lr)
    if train_flag:
        train(model, train_loader, test_loader, num_epochs, lambda_, device, criterion, optimizer, dataset)
        # anomaly_score, test_labels = test(model, dataset, 1, test_loader, device, window_size)
    else:
        anomaly_score, test_labels = test(model, dataset, 1, test_loader, device, window_size)
        adj_eval_metrics = ts_metrics(test_labels, point_adjustment(test_labels, anomaly_score))
        print("Adjusted evaluation metrics: ", adj_eval_metrics)    # The third value is the best F1 with point adjustment