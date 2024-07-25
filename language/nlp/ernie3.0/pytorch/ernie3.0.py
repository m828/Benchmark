# import torch
# from transformers import BertTokenizer
# import time
# import onnxruntime as ort
# import numpy as np
# # 检查是否有可用的GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model_path = 'language/nlp/ernie3.0/pytorch/ernie-3.0-medium-zh.onnx'
# local_tokenizer_path = 'language/nlp/ernie3.0/pytorch/vocab'
# tokenizer = BertTokenizer.from_pretrained(local_tokenizer_path)
# print(tokenizer)

# # 创建会话
# session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# # 打印当前使用的设备
# print("Providers used by the ONNX runtime session:", session.get_providers())
# # 载入ONNX模型


# text = "我觉得你很好！"
# # 将文本转换为模型输入格式
# inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=256)
# input_ids = inputs['input_ids']  # 将 input_ids tensor 转换为 numpy 数组
# token_type_ids = inputs['token_type_ids']    # 将 token_type_ids tensor 转换为 numpy 数组
# # input_ids = input_ids0.astype(np.int32)
# # token_type_ids = token_type_ids0.astype(np.int32)
# # print(input_ids)
# print(input_ids.shape)
# # ort_inputs = {session.get_inputs()[0].name: input_ids,session.get_inputs()[1].name: token_type_ids}
# ort_inputs = {
#     session.get_inputs()[0].name: input_ids.astype(np.int64),
#     session.get_inputs()[1].name: token_type_ids.astype(np.int64)
# }
# # 记录开始时间
# start_time = time.time()
# iterations = 128
# for _ in range(iterations):
#     # 执行推断
#     outputs = session.run(None, ort_inputs)
# # 记录结束时间
# end_time = time.time()
# # 计算执行时间
# execution_time = end_time - start_time
# latency = execution_time / iterations * 1000
# FPS = 1000 / latency
# print(f"FPS: {FPS:.2f}")



import torch
from transformers import BertTokenizer
import time
import onnxruntime as ort
import numpy as np

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'language/nlp/ernie3.0/pytorch/ernie-3.0-medium-zh.onnx'
local_tokenizer_path = 'language/nlp/ernie3.0/pytorch/vocab'
tokenizer = BertTokenizer.from_pretrained(local_tokenizer_path)
print(tokenizer)

print(ort.get_device())

ort_session = ort.InferenceSession("language/nlp/ernie3.0/pytorch/ernie-3.0-medium-zh.onnx",
providers=['CUDAExecutionProvider'])
print(ort_session.get_providers())

# # 创建会话
# session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

# # 打印当前使用的设备
# print("Providers used by the ONNX runtime session:", session.get_providers())

# text = "我觉得你很好！"
# # 将文本转换为模型输入格式
# inputs = tokenizer(text, return_tensors='np', padding='max_length', max_length=256)
# input_ids = inputs['input_ids']  # input_ids tensor 转换为 numpy 数组
# token_type_ids = inputs['token_type_ids']  # token_type_ids tensor 转换为 numpy 数组

# # 打印输入的形状
# print(input_ids.shape)

# # 准备输入数据
# ort_inputs = {
#     session.get_inputs()[0].name: input_ids.astype(np.int64),
#     session.get_inputs()[1].name: token_type_ids.astype(np.int64)
# }

# # 记录开始时间
# start_time = time.time()
# iterations = 128
# for _ in range(iterations):
#     # 执行推断
#     outputs = session.run(None, ort_inputs)
# # 记录结束时间
# end_time = time.time()

# # 计算执行时间
# execution_time = end_time - start_time
# latency = execution_time / iterations * 1000
# FPS = 1000 / latency
# print(f"FPS: {FPS:.2f}")
