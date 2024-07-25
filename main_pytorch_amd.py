import os
import torch
import json
import time
import argparse
import yaml
from thop import profile
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pynvml
import threading
import subprocess

# rcParams['font.sans-serif'] = ['SimHei'] 


import importlib.util

# 监控 GPU 使用情况
def get_gpu_info():
    # 调用 rocm-smi 并解析输出
    result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram', '--showuse', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 打印命令行输出用于调试
    print("rocm-smi stdout:", result.stdout.decode('utf-8'))
    print("rocm-smi stderr:", result.stderr.decode('utf-8'))
    
    try:
        gpu_info = json.loads(result.stdout.decode('utf-8'))
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        gpu_info = {}

    # 打印解析后的JSON结构
    print("Parsed GPU Info:", gpu_info)

    if not gpu_info:
        return [0, 0, 0, 0]

    card_key = list(gpu_info.keys())[0]  # 获取第一个卡的键名

    # 提取并转换数值
    mem_total = int(gpu_info[card_key].get('VRAM Total Memory (B)', 0))
    mem_used = int(gpu_info[card_key].get('VRAM Total Used Memory (B)', 0))
    usage = int(gpu_info[card_key].get('GPU use (%)', 0))
    power_draw = int(gpu_info[card_key].get('Average Graphics Package Power (W)', 0))  # 如果存在的话

    mem_info = mem_total / 1024**2, mem_used / 1024**2

    return [usage, mem_info[0], mem_info[1], power_draw]

def monitor_gpu_usage():
    start_event.wait()
    while True:
        gpu_info = get_gpu_info()
        gpu_usage_list.append(gpu_info)
        time.sleep(0.001)  # 1 毫秒



# 速度测试
def speed_test(model, input, iterations):

    if iterations is None:
        elapsed_time = 0
        iterations = 100
        while elapsed_time < 1:
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(iterations):
                model(input)
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_start
            iterations *= 2
        FPS = iterations / elapsed_time
        iterations = int(FPS * 6)

    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iterations):
        model(input)
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start
    latency = elapsed_time / iterations * 1000
    FPS = 1000 / latency
    model_performance.extend([FPS, latency])

    # for i in [FPS, latency]:
    #     model_performance.append(i)

    # return FPS, latency

def speed_test_l(model, iterations):

    if iterations is None:
        elapsed_time = 0
        iterations = 100
        while elapsed_time < 1:
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(iterations):
                model.forward()
            torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed_time = time.time() - t_start
            iterations *= 2
        FPS = iterations / elapsed_time
        iterations = int(FPS * 6)

    print('=========Speed Testing=========')
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iterations):
        model.forward()
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start
    latency = elapsed_time / iterations * 1000
    FPS = 1000 / latency
    model_performance.extend([FPS, latency])

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




########################################################################

script_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.run(['python', os.path.join(script_dir, 'update_config.py')], check=True) #刷新模型文件

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="vision/detection/yolov5/pytorch/configs/yolov5x.yaml", help="model.yaml")
parser.add_argument("--iterations", type=int, default=None, help="迭代次数")
opt = parser.parse_args()

config = load_config('config.json')
category = choose_option(config['categories'], "选择AI应用领域：")
application = choose_option(config['applications'][category], f"选择{category} 中的应用场景：")
model = choose_option(config['models'][category][application], f"选择{application} 中的具体模型：")

print(f"你选择了 {category} 领域中的 {application} 场景下的 {model} 模型")





# 模型脚本的路径
model_script_path = category + '/'+ application + '/'+ model+'/pytorch/'+ model+ '.py'

# 动态加载模块
spec = importlib.util.spec_from_file_location("model_module", model_script_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# 获取模型类
model_class = getattr(model_module, model)

# 实例化模型
model = model_class()

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

# pynvml.nvmlInit()
start_event = threading.Event()

gpu_usage_list = [] ##保存gpu使用情况的时间序列
model_performance = [] ##保存模型推理性能


#################模型推理和GPU监控双线程启动##############

monitor_thread = threading.Thread(target=monitor_gpu_usage)
# monitor_thread.daemon = True
monitor_thread.start()  # 启动 GPU 监控线程

iterations = opt.iterations
inference_thread = threading.Thread(target=speed_test(model, input, iterations = iterations))
inference_thread.start()  # 启动推理线程

start_event.set()  # 触发事件，开始监控和推理

inference_thread.join() # 等待推理完成

# 停止监控（如果需要，可以增加一个标志位来控制循环）
time.sleep(1)  # 确保监控线程完成最后的记录



# 打印或处理 GPU 使用情况数据
# print(gpu_usage_list)
# for usage in gpu_usage_list:
#     print(usage)

headers = gpu_usage_list[0]

column3 = [row[2] for row in gpu_usage_list[1:]]
column4 = [row[3] for row in gpu_usage_list[1:]]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(column3, label=headers[2])
ax1.set_title(headers[2])
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("MiB")
ax1.legend()

ax2.plot(column4, label=headers[3])
ax2.set_title(headers[3])
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("W")
ax2.legend()

plt.tight_layout()

plt.savefig('savefiles/gpu_usage.png')


###############处理GPU监控结果和模型推理数据#############

if category == 'language':
    speed_test_l(model, iterations = None)
    FPS, latency = model_performance[0], model_performance[1]
    flops, params = model.count_parameters_and_flops()
else:
    speed_test(model, input, iterations = None)
    FPS, latency = model_performance[0], model_performance[1]
    flops, params = count_parameters_and_flops(model, input)

print(f"Total parameters: {params:.2f} million")

# print(f"Trainable parameters: {trainable_params:.2f} million")
print(f"Total GFLOPs: {flops:.2f}")
print(f"FPS: {FPS:.2f}")
print(f"Latency: {latency:.2f} ms")


# 清理缓存
torch.cuda.empty_cache()