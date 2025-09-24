""" 
This code was taken from the Github Repo: https://github.com/hoangthangta/FC_KAN/blob/main/run.py

and edited for new W-CSRBF Models and regularizers

"""
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import requests

#import numpy as np
from file_io import *
from models import EfficientKAN, FastKAN, BSRBF_KAN, FasterKAN, MLP, FC_KAN, WCSRBFKAN, KAN, WCSRBFKAN
from kan.utils import create_dataset_from_data

from pathlib import Path
from PIL import Image
from prettytable import PrettyTable
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.model_selection import ParameterGrid


def remove_unused_params(model):
    
    unused_params, _ = count_unused_params(model)
    for name in unused_params:
        #attr_name = name.split('.')[0]  # Get the top-level attribute name (e.g., 'unused')
        if hasattr(model, name):
            #print(f"Removing unused layer: {name}")
            delattr(model, name)  # Dynamically remove the unused layer
    return model

def count_unused_params(model):
    # Detect and count unused parameters
    unused_params = []
    unused_param_count = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)
            unused_param_count += param.numel()  # Add the number of elements in this parameter
    
    return unused_params, unused_param_count

def count_params(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    
    # Detect and count unused parameters
    unused_params, unused_param_count = count_unused_params(model)
    
    if (unused_param_count != 0):
        print("Unused Parameters:", unused_params)
        print(f"Total Trainable Params: {total_params}")
        print(f"Total Number of Unused Parameters: {unused_param_count}")
        print(f"Total Number of Used Parameters: {total_params - unused_param_count}")
    else:
        print(f"Total Trainable Params: {total_params}")
        print(f"Total Number of Used Parameters: {total_params - unused_param_count}")
    
    return total_params

def weights_l2(model_name, model, lambda_w=1e-4):
    reg = 0.0
    if model_name == "mlp":
        for l in model.layers:
            reg += l.base_weight.pow(2).sum()
        return lambda_w * reg
    if model_name == "efficient_kan":
        for l in model.layers:
            reg += l.base_weight.pow(2).sum() + l.spline_weight.pow(2).sum() + l.spline_scaler.pow(2).sum()
        return lambda_w * reg
    if model_name == "fast_kan":
        for l in model.layers:
            reg += l.base_linear.weight.pow(2).sum() + l.base_linear.bias.pow(2).sum() + l.spline_linear.weight.pow(2).sum()
        return lambda_w * reg
    if model_name == "faster_kan":
        for l in model.layers:
            reg += l.spline_linear.weight.pow(2).sum()
        return lambda_w * reg
    if model_name == "bsrbf_kan":
        for l in model.layers:
            reg += l.base_weight.pow(2).sum() + l.spline_weight.pow(2).sum()
        return lambda_w * reg
    if model_name == "fc_kan":
        for l in model.layers:
            reg += l.base_weight.pow(2).sum() + l.spline_weight.pow(2).sum()
        return lambda_w * reg
    else: 
        return reg

def run(args):
    
    start = time.time()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    
    trainset, valset = [], []
    if (args.ds_name == 'mnist'):
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        valset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif(args.ds_name == 'fashion_mnist'):
        trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )

        valset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )

    if (args.n_examples > 0):
        if (args.n_examples/args.batch_size > 1):
            trainset = torch.utils.data.Subset(trainset, range(args.n_examples))
        else:
            print('The number of examples is too small!')
            return
    elif(args.n_part > 0):
        if (len(trainset)*args.n_part > args.batch_size):
            trainset = torch.utils.data.Subset(trainset, range(int(len(trainset)*args.n_part)))
        else:
            print('args.n_part is too small!')
            return

    print('trainset: ', len(trainset))
    print('valset: ', len(valset))
    
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False) # we can want to keep the stability of models when training
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    # Create model storage
    output_path = args.folder_name + args.ds_name + '/' + args.model_name + '/'
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    saved_model_name, saved_model_history = '', ''
    # if (args.model_name == 'fc_kan'):
    #     saved_model_name = args.model_name + '__' + args.ds_name + '__' + '-'.join(x for x in args.func_list) + '__' + args.combined_type + '__' + args.note + '.pth'
    #     saved_model_history = args.model_name + '__' + args.ds_name + '__' + '-'.join(x for x in args.func_list) + '__' + args.combined_type + '__' + args.note + '.json' 
    if(args.model_name == 'skan'):
        # args.basis_function
        saved_model_name = args.model_name + '__' + args.ds_name + '__' + args.basis_function + '__' + args.note + '.pth'
        saved_model_history =  args.model_name + '__' + args.ds_name + '__' + args.basis_function + '__' + args.note + '.json'
    else:
        saved_model_name = args.model_name + '__' + args.ds_name + '__' + args.note + '.pth'
        saved_model_history =  args.model_name + '__' + args.ds_name + '__' + args.note + '.json'

    # Define models
    model = {}
    print('model_name: ', args.model_name)
    if (args.model_name == 'kan'):
        model = KAN([args.n_input, args.n_hidden, args.n_output], grid = args.grid_size, k = args.spline_order, ckpt_path=output_path + '/' + saved_model_name)
    if (args.model_name == 'bsrbf_kan'):
        model = BSRBF_KAN([args.n_input, args.n_hidden, args.n_output], grid_size = args.grid_size, spline_order = args.spline_order, layernorm=False)
    elif(args.model_name == 'fast_kan'):
        model = FastKAN([args.n_input, args.n_hidden, args.n_output], num_grids = args.num_grids, layernorm=False)
    elif(args.model_name == 'faster_kan'):
        model = FasterKAN([args.n_input, args.n_hidden, args.n_output], num_grids = args.num_grids, layernorm=False)
    elif(args.model_name == 'wcsrbf_kan_un'):
        model = WCSRBFKAN(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=False, trainable_centers=False, trainable_sigma=False)
    elif(args.model_name == 'wcsrbf_kan_tc_ts_un'):
        model = WCSRBFKAN(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=False, trainable_centers=True, trainable_sigma=True)
    elif(args.model_name == 'wcsrbf_kan_ts_un'):
        model = WCSRBFKAN(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=False, trainable_centers=False, trainable_sigma=True)
    elif(args.model_name == 'wcsrbf_kan_tc_un'):
        model = WCSRBFKAN(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=False, trainable_centers=True, trainable_sigma=False)
    elif(args.model_name == 'wcsrbf_kan'):
        model = WCSRBFKAN(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=True, trainable_centers=False, trainable_sigma=False)
    elif(args.model_name == 'wcsrbf_kan_tc_ts'):
        model = WCSRBFKAN(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=True, trainable_centers=True, trainable_sigma=True)
    elif(args.model_name == 'wcsrbf_kan_ts'):
        model = WCSRBFKAN(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=True, trainable_centers=False, trainable_sigma=True)
    elif(args.model_name == 'wcsrbf_kan_tc'):
        model = WCSRBFKAN(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=True, trainable_centers=True, trainable_sigma=False)
    elif(args.model_name == 'wcsrbf_kan_solo'):
        model = WCSRBFKANSolo(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=False, trainable_centers=False, trainable_sigma=False, use_base=False)
    elif(args.model_name == 'wcsrbf_kan_solo_tc'):
        model = WCSRBFKANSolo(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=False, trainable_centers=True, trainable_sigma=False, use_base=False)
    elif(args.model_name == 'wcsrbf_kan_solo_ts'):
        model = WCSRBFKANSolo(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=False, trainable_centers=False, trainable_sigma=True, use_base=False)
    elif(args.model_name == 'wcsrbf_kan_solo_tc_ts'):
        model = WCSRBFKANSolo(dims=[args.n_input, args.n_hidden, args.n_output], n_centers = args.num_grids, enable_layer_norm=False, trainable_centers=True, trainable_sigma=True, use_base=False)
    elif(args.model_name == 'mlp'):
        model = MLP([args.n_input, args.n_hidden, args.n_output], layernorm=False)
    elif(args.model_name == 'mlp_class'):
        model = MLPClassification([args.n_input, args.n_hidden, args.n_output], layernorm=False)
    elif(args.model_name == 'fc_kan'):
        model = FC_KAN([args.n_input, args.n_hidden, args.n_output], args.func_list, combined_type = args.combined_type, grid_size = args.grid_size, spline_order = args.spline_order, drop_out = args.drop_out, layernorm=False)
    elif(args.model_name == 'efficient_kan'):
        model = EfficientKAN([args.n_input, args.n_hidden, args.n_output], grid_size = args.grid_size, spline_order = args.spline_order)
    # else:
    #     raise ValueError("Unsupported network type.")
    model.to(device)
    best_epoch, best_accuracy = 0, 0
    criterion = nn.CrossEntropyLoss()

    if args.model_name != "kan":
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # Define learning rate scheduler
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

        y_true = [labels.tolist() for images, labels in valloader]
        y_true = sum(y_true, [])
        
        for epoch in range(1, args.epochs + 1):
            # Train
            model.train()
            train_accuracy, train_loss = 0, 0
            with tqdm(trainloader) as pbar:
                for i, (images, labels) in enumerate(pbar):  
                    if (args.model_name != 'cnn'):
                        images = images.view(-1, args.n_input).to(device)
                    optimizer.zero_grad()
                    output = model(images.to(device))
                    loss = criterion(output, labels.to(device))
                    if "wcsrbf" in args.model_name:
                        if model.trainable_sigma_bool:
                            loss += model.sigma_inverse_l2(0.001)
                        loss += model.weights_l2(0.001)
                    else:
                        loss += weights_l2(args.model_name, model, 0.001)

                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    #accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
                    train_accuracy += (output.argmax(dim=1) == labels.to(device)).float().mean().item()
                    pbar.set_postfix(loss=train_loss/len(trainloader), accuracy=train_accuracy/len(trainloader), lr=optimizer.param_groups[0]['lr'])
            
            train_loss /= len(trainloader)
            train_accuracy /= len(trainloader)
                
            # Validation
            model.eval()
            val_loss, val_accuracy = 0, 0
            
            y_pred = []
            with torch.no_grad():
                for images, labels in valloader:
                    if (args.model_name != 'cnn'):
                        images = images.view(-1, args.n_input).to(device)
                    output = model(images.to(device))
                    val_loss += criterion(output, labels.to(device)).item()
                    y_pred += output.argmax(dim=1).tolist()
                    val_accuracy += ((output.argmax(dim=1) == labels.to(device)).float().mean().item())
            
            f1 = f1_score(y_true, y_pred, average='macro')
            pre = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')

            val_loss /= len(valloader)
            val_accuracy /= len(valloader)

            # Update learning rate
            scheduler.step()

            # Choose best model
            if (val_accuracy > best_accuracy):
                best_accuracy = val_accuracy
                best_epoch = epoch
                torch.save(model, output_path + '/' + saved_model_name)
                
            print(f"Epoch [{epoch}/{args.epochs}], Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}")
            print(f"Epoch [{epoch}/{args.epochs}], Val Loss: {val_loss:.6f}, Val Accuracy: {val_accuracy:.6f}, F1: {f1:.6f}, Precision: {pre:.6f}, Recall: {recall:.6f}")
            
            write_single_dict_to_jsonl(output_path + '/' + saved_model_history, {'epoch':epoch, 'val_accuracy':val_accuracy, 'train_accuracy':train_accuracy, 'f1_macro':f1, 'pre_macro':pre, 're_macro':recall, 'best_epoch':best_epoch, 'val_loss': val_loss, 'train_loss':train_loss}, file_access = 'a')
        
        end = time.time()
        print(f"Training time (s): {end-start}")
        write_single_dict_to_jsonl(output_path + '/' + saved_model_history, {'training time':end-start}, file_access = 'a')

        # remove unused parameters and count the number of parameters after that
        remove_unused_params(model)
        torch.save(model, output_path + '/final_' + saved_model_name)
        count_params(model)
    else:
        dtype = torch.get_default_dtype()

        dataset = {}
        dataset['train_input'] = trainset.data.flatten(start_dim=1).type(dtype).to(device)
        dataset['test_input'] = valset.data.flatten(start_dim=1).type(dtype).to(device)
        dataset['train_label'] = trainset.targets[:,None].to(device)
        dataset['test_label'] = valset.targets[:,None].to(device)


        # def train_acc():
        #     return torch.mean((torch.round(model(dataset['train_input'])[:,0]) == dataset['train_label'][:,0]).type(dtype))

        # def test_acc():
        #     return torch.mean((torch.round(model(dataset['test_input'])[:,0]) == dataset['test_label'][:,0]).type(dtype))

        # results = model.fit(dataset, opt="Adam", steps=args.epochs, batch=64, lr=args.lr, metrics=(train_acc, test_acc));
        results = model.fit(dataset, opt="Adam", steps=args.epochs, batch=args.batch_size, lr=args.lr);

        print(results)

        # end = time.time()
        # print(f"Training time (s): {end-start}")
        # write_single_dict_to_jsonl(output_path + '/' + saved_model_history, {'training time':end-start}, file_access = 'a')
    
    return best_accuracy
   
def main(args):
    
    func_list = args.func_list.split(',')
    func_list = [x.strip() for x in func_list]
    args.func_list = func_list
    
    if (args.mode == 'train'):
        run(args)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train') # or 'predict_set', 'grid_search'
    parser.add_argument('--model_name', type=str, default='efficient_kan')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_input', type=int, default=28*28)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_output', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_path', type=str, default='output/model.pth')
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--num_grids', type=int, default=8)
    parser.add_argument('--spline_order', type=int, default=3)
    parser.add_argument('--folder_name', type=str, default='output/')
    parser.add_argument('--ds_name', type=str, default='mnist')
    parser.add_argument('--n_examples', type=int, default=0)
    parser.add_argument('--note', type=str, default='full')
    parser.add_argument('--n_part', type=float, default=0)
    parser.add_argument('--func_list', type=str, default='dog,rbf') # for FC-KAN
    parser.add_argument('--combined_type', type=str, default='quadratic')
    
    parser.add_argument('--wd', type=float, default=1e-4) # weight decay
    parser.add_argument('--lr', type=float, default=1e-3) # learning rate
    parser.add_argument('--gamma', type=float, default=0.8) # learning rate
    parser.add_argument('--drop_out', type=float, default=0) # learning rate
    

    # use for SKAN
    parser.add_argument('--basis_function', type=str, default='sin')
    
    args = parser.parse_args()
    
    global device
    device = args.device
    if (args.device == 'cuda'): # check available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    main(args)

#python run.py --mode "train" --model_name "fc_kan" --epochs 35 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "fashion_mnist" --func_list "dog,sin" --combined_type "sum"

#python run.py --mode "train" --model_name "bsrbf_kan" --epochs 1 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --grid_size 5 --spline_order 3 --ds_name "mnist"

#python run.py --mode "train" --model_name "skan" --epochs 10 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --basis_function "sin"

#python run.py --mode "train" --model_name "fast_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8 --ds_name "mnist"

#python run.py --mode "train" --model_name "faster_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --num_grids 8 --ds_name "mnist"

#python run.py --mode "train" --model_name "gottlieb_kan" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --spline_order 3 --ds_name "mnist"

#python run.py --mode "train" --model_name "mlp" --epochs 25 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --note "full"

#python run.py --mode "train" --model_name "mlp" --epochs 15 --batch_size 64 --n_input 3072 --n_hidden 64 --n_output 10 --ds_name "cifar10" --note "full"

#python run.py --mode "train" --model_name "fc_kan" --epochs 1 --batch_size 64 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --func_list "bs,dog" --combined_type "quadratic" --note "full"

#python run.py --mode "train" --model_name "fc_kan" --epochs 15 --batch_size 64 --n_input 3072 --n_hidden 64 --n_output 10 --ds_name "cifar10" --func_list "bs,dog" --combined_type "quadratic" --note "full"

#python run.py --mode "train" --model_name "cnn" --epochs 15 --batch_size 64 --ds_name "mnist" --note "full"

#python run.py --mode "predict_set" --model_name "bsrbf_kan" --model_path='papers//BSRBF-KAN//bsrbf_paper//mnist//bsrbf_kan//bsrbf_kan__mnist__full_0.pth' --ds_name "mnist" --batch_size 64

#python run.py --mode "grid_search" --model_name "fc_kan" --epochs 25 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --func_list "bs,dog" --combined_type "quadratic" --device cpu


#python run.py --mode "grid_search" --model_name "fc_kan" --epochs 35 --n_input 784 --n_hidden 64 --n_output 10 --ds_name "mnist" --func_list "bs,dog" --combined_type "quadratic" --device cpu
