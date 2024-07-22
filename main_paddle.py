import paddle
import time
import argparse
import yaml
import paddle.nn as nn


from vision.classification.resnet.paddle.resnet import ResNet50
from vision.classification.mobilenet.paddle.mobilenetv2  import MobileNetV2


device = paddle.set_device('gpu')

# parser = argparse.ArgumentParser()
# parser.add_argument("--cfg", type=str, default="vision/detection/yolov5/pytorch/configs/yolov5x.yaml", help="model.yaml")
# opt = parser.parse_args()
# with open(opt.cfg) as fp:
#         opt.cfg= yaml.safe_load(fp)
# opt.cfg = check_yaml(opt.cfg)  # check YAML

input = paddle.randn([1, 3, 640, 640], dtype='float32')

# 分类
# model = MobileNetV2()
model = ResNet50()

# 检测
# model = Model(opt.cfg).to(device)

# 分割

# model = unet(in_channels=3, out_channels=8)
model.eval()
model.to(device)

# 预热
with paddle.no_grad():
    for _ in range(10):
        model(input)

# 速度测试
def speed_test(model, input, iterations):

    if iterations is None:
        elapsed_time = 0
        iterations = 100
        while elapsed_time < 1:
            paddle.device.cuda.synchronize()
            paddle.device.cuda.synchronize()
            t_start = time.time()
            for _ in range(iterations):
                model(input)
            paddle.device.cuda.synchronize()
            paddle.device.cuda.synchronize()
            elapsed_time = time.time() - t_start
            iterations *= 2
        FPS = iterations / elapsed_time
        iterations = int(FPS * 6)

    print('=========Speed Testing=========')
    paddle.device.cuda.synchronize()
    paddle.device.cuda.synchronize()
    t_start = time.time()
    for _ in range(iterations):
        model(input)
    paddle.device.cuda.synchronize()
    paddle.device.cuda.synchronize()
    elapsed_time = time.time() - t_start
    latency = elapsed_time / iterations * 1000
    FPS = 1000 / latency
    return FPS, latency
    
# 计算模型参数量和FLOPs
def count_parameters_and_flops(model, input):
    flops = paddle.flops(model, [1, 3, 640, 640])
    params = sum(p.numel() for p in model.parameters()).item()
    return flops / 1e9 * 2, params / 1e6

# params_info = paddle.summary(model, input)
# print(params_info)

FPS, latency = speed_test(model, input, iterations = None)
flops, params = count_parameters_and_flops(model, input)

print(f"Total parameters: {params:.2f} million")
# print(f"Trainable parameters: {trainable_params:.2f} million")
print(f"Total GFLOPs: {flops:.2f}")
print(f"FPS: {FPS:.2f}")
print(f"Latency: {latency:.2f} ms")


# 清理缓存
paddle.device.cuda.empty_cache()