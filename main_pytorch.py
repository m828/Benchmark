
import os
import torch
import json
import time
import argparse
import yaml
import pandas as pd
import torchprofile
from thop import profile
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pynvml
import threading
import subprocess
from vision.segmentation.unet.pytorch.unet import unet
from vision.detection.yolov5.pytorch.yolov5 import Model
import matplotlib.gridspec as gridspec
# rcParams['font.sans-serif'] = ['SimHei'] 


import importlib.util

# 监控 GPU 使用情况
def get_gpu_info():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0 #W  
    return [utilization.gpu, mem_info.total / 1024**2, mem_info.used / 1024**2, power_draw]  
    
def monitor_gpu_usage(gpu_usage_list, start_event):
    start_event.wait()
    print('=========GPU monitor=========')
    while True:
        t_start = time.time()
        gpu_info = get_gpu_info()
        gpu_usage_list.append(gpu_info)
        t_elapsed = time.time() - t_start
        time_to_sleep = max(0, 0.1 - t_elapsed)
        time.sleep(time_to_sleep)


# 速度测试
def speed_test(model, input, iterations, model_performance, start_event):
    start_event.wait()
    print('=========Model Inference=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iterations):
        model(input)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    model_performance.extend([elapsed_time])

def speed_test_l(model, iterations, model_performance, start_event):

    start_event.wait()
    print('=========Model Inference=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iterations):
        model.forward()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    model_performance.extend([elapsed_time])

    
def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config

def choose_option(options, prompt):
    print(prompt)
    for key, value in options.items():
        print(f"{key}. {value}")
    choice = input("输入选项编号：")
    while choice not in options:
        print("无效的选项，请重新输入。")
        choice = input("输入选项编号：")
    return options[choice]

def count_parameters_and_flops(model, input):

    flops, params = profile(model, inputs=(input,), verbose=False) 

    return flops / 1e9 * 2,  params / 1e6

def plot_matrix(model_name, gpu_usage_list, model_perf_metrix):

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 3])
    
    # 表格
    ax_table = fig.add_subplot(gs[0, :])
    ax_table.axis('off')
    
    col_labels = ['Model','Params (M)', 'FLOPs (G)', 'FPS', 'Latency (ms)', 'Energy (KJ)']
    
    model_perf_metrix = [round(num, 2) for num in model_perf_metrix]
    table = ax_table.table(cellText=[[model_name,*model_perf_metrix]], colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)


    headers = gpu_usage_list[0]
    column3 = [row[2] for row in gpu_usage_list[1:]]
    column4 = [row[3] for row in gpu_usage_list[1:]]

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(column3, label=headers[2])
    ax1.set_title(headers[2])
    ax1.set_xlabel("Time (100ms)")
    ax1.set_ylabel("MiB")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(column4, label=headers[3])
    ax2.set_title(headers[3])
    ax2.set_xlabel("Time (100ms)")
    ax2.set_ylabel("W")
    ax2.legend()
    plt.tight_layout()
    plt.savefig('savefiles/'+model_name+'_matrix.png')

    return 


def generate_paths(data):
    paths = []
    
    # 获取 categories 和 applications 部分的映射
    categories = data.get("categories", {})
    applications = data.get("applications", {})
    models = data.get("models", {})
    
    for cat_key, cat_value in categories.items():
        if cat_value in applications:
            for app_key, app_value in applications[cat_value].items():
                if app_value in models.get(cat_value, {}):
                    for model_key, model_value in models[cat_value].get(app_value, {}).items():
                        path = f"{cat_value}/{app_value}/{model_value}"
                        paths.append(path)
                    
    return paths

def main_fun(model, opt, category):
    device = torch.device('cuda')

    if category == 'language':
        with torch.no_grad():
            for _ in range(10):
                model.forward()      
    else:
        input = torch.randn(1, 3, 640, 640).to(device)
        model.eval()
        model.to(device)
        with torch.no_grad():
            for _ in range(10):
                model(input)

    pynvml.nvmlInit()
    start_event = threading.Event()


    gpu_usage_list = [['GPU Index','GPU Memory','Memory Usage','GPU Power']] ##储存gpu使用情况的时间序列数据
    model_performance = [] ##保存模型推理数据

    #################模型推理和GPU监控双线程启动##############
    

    monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(gpu_usage_list, start_event))
    monitor_thread.daemon = True
    monitor_thread.start()  # 启动 GPU 监控线程

    iterations = opt.iterations
    if category == 'language':
        inference_thread = threading.Thread(target=speed_test_l, args=(model, iterations, model_performance, start_event))
    else:
        inference_thread = threading.Thread(target=speed_test, args=(model, input, iterations,model_performance, start_event))
    inference_thread.start()  # 启动推理线程

    time.sleep(1)
    start_event.set()  

    inference_thread.join() 

    time.sleep(0.03)  # 确保监控线程完成最后的记录


    ###############处理GPU监控结果和模型推理数据#############


    column4 = [row[3] for row in gpu_usage_list[1:]]
    time_interval_s = 100 / 1000.0  # 将时间间隔转换为秒
    total_energy_joules = sum(power * time_interval_s for power in column4)/1000

    latency = model_performance[0] / iterations * 1000
    FPS = 1000 / latency

    if category == 'language':
        flops, params = model.count_parameters_and_flops()
    else:
        flops, params = count_parameters_and_flops(model, input)

    model_perf_metrix = [params, flops, FPS, latency, total_energy_joules]

    return gpu_usage_list, model_perf_metrix

########################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--iterations", type=int, default=1000, help="迭代次数")
parser.add_argument("--imgsize", type=int, default=640, help="模型输入尺寸")
parser.add_argument("--batchsize", type=int, default=1, help="推理时的批次大小")
parser.add_argument("--testmode", type=int, default=1, help="测试模式。0代表全部测试集统一测试; 1代表使用单一模型测试 ")

# 解析命令行参数
opt = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.run(['python', os.path.join(script_dir, 'update_config.py')], check=True) #测试集刷新

if opt.testmode == 0:
    config = load_config('config.json')
    modellist = generate_paths(config)
    banchmark = [['Model','Params (M)', 'FLOPs (G)', 'FPS', 'Latency (ms)', 'Energy (KJ)']]

    for model_path in modellist:
        category = model_path.split('/')[0]
        model_name = model_path.split('/')[-1]
        model_script_path = model_path + '/' + 'pytorch/' + model_name + '.py'
        # 动态加载模块
        spec = importlib.util.spec_from_file_location("model_module", model_script_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        model_class = getattr(model_module, model_name)
        model = model_class()
        
        print(model_script_path)
        gpu_usage_list, model_perf_metrix = main_fun(model, opt, category)

        plot_matrix(model_name, gpu_usage_list, model_perf_metrix)

        banchmark.append([model_path,*model_perf_metrix])

        # 清理缓存
        torch.cuda.empty_cache()

    print(banchmark)
    headers = banchmark[0]
    rows = banchmark[1:]

    banchmark_tf = [[row[0]] + [round(num, 2) for num in row[1:]]for row in rows]

    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])

    # 表格
    ax_table = fig.add_subplot(gs[0, 0])
    ax_table.axis('off')

    table = ax_table.table(cellText=[headers] + banchmark_tf, colLabels=None, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.savefig('savefiles/banchmark.png')


if opt.testmode == 1:
    config = load_config('config.json')
    category = choose_option(config['categories'], "选择AI应用领域：")
    application = choose_option(config['applications'][category], f"选择{category} 中的应用场景：")
    model_name = choose_option(config['models'][category][application], f"选择{application} 中的具体模型：")
    print(f"你选择了 {category} 领域中的 {application} 场景下的 {model_name} 模型")

    model_script_path = category + '/'+ application + '/'+ model_name+'/pytorch/'+ model_name+ '.py'
    # 动态加载模块
    spec = importlib.util.spec_from_file_location("model_module", model_script_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model_class = getattr(model_module, model_name)
    model = model_class()
    

    gpu_usage_list, model_perf_metrix = main_fun(model, opt, category)
    plot_matrix(model_name, gpu_usage_list, model_perf_metrix)

    print(model_perf_metrix)


    # 清理缓存
    torch.cuda.empty_cache()