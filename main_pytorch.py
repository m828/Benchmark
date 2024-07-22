import torch
import time
import argparse
import yaml
import torchprofile
from thop import profile

from vision.classification.resnet.pytorch.resnet import resnet50
from vision.classification.mobilenet.pytorch.mobilenetv2 import mobilenet_v2
from vision.detection.yolov5.pytorch.yolov5 import Model
from vision.segmentation.unet.pytorch.unet import unet
from vision.segmentation.bisenetv2.pytorch.bisenetv2 import BiSeNetV2

device = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="vision/detection/yolov5/pytorch/configs/yolov5x.yaml", help="model.yaml")
opt = parser.parse_args()
with open(opt.cfg) as fp:
        opt.cfg= yaml.safe_load(fp)
# opt.cfg = check_yaml(opt.cfg)  # check YAML

input = torch.randn(1, 3, 640, 640).to(device)

# 分类
# model = mobilenet_v2()
# model = resnet50()

# 检测
# model = Model(opt.cfg).to(device)

# 分割
model = BiSeNetV2(n_classes=4)
# model = unet(in_channels=3, out_channels=8)
model.eval()
model.to(device)

# 预热
with torch.no_grad():
    for _ in range(10):
        model(input)

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
    return FPS, latency
    
# 计算模型参数量和FLOPs
def count_parameters_and_flops(model, input):

    flops, params = profile(model, inputs=(input,), verbose=False) 

    return flops / 1e9 * 2,  params / 1e6


FPS, latency = speed_test(model, input, iterations = None)
flops, params = count_parameters_and_flops(model, input)

print(f"Total parameters: {params:.2f} million")
# print(f"Trainable parameters: {trainable_params:.2f} million")
print(f"Total GFLOPs: {flops:.2f}")
print(f"FPS: {FPS:.2f}")
print(f"Latency: {latency:.2f} ms")


# 清理缓存
torch.cuda.empty_cache()