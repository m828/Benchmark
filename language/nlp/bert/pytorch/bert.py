import torch
# from transformers import BertTokenizer, BertModel
# import time



# # 加载本地的 BERT 模型和分词器
# model_path = 'language/nlp/bert/pytorch/vocab'
# tokenizer_path = 'language/nlp/bert/pytorch/vocab'

# tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
# model = BertModel.from_pretrained(model_path).to(device)

# # 输入文本
# text = "Hello, how are you?"

# # 对输入文本进行编码
# inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=256).to(device)

# outputs = model(**inputs) 
import os
import torch
import requests
from transformers import BertTokenizer, BertModel

def download_model_weights(model_path):
    if not os.path.exists(os.path.join(model_path, 'pytorch_model.bin')):
        print(f"权重文件不存在，正在从 Hugging Face 下载权重...")
        model_url = "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/pytorch_model.bin?download=true"
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(os.path.join(model_path, 'pytorch_model.bin'), 'wb') as f:
                f.write(response.content)
            print("权重下载完成。")
        else:
            print("权重下载失败，请检查网络连接或 URL。")

class bert:
    def __init__(self, text="Hello, how are you?", max_length=256, model_path='language/nlp/bert/pytorch/vocab', tokenizer_path='language/nlp/bert/pytorch/vocab'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        download_model_weights(model_path)
        self.model = BertModel.from_pretrained(model_path).to(self.device)
        self.text = text
        self.max_length = max_length

    def forward(self):
        
        # 对输入文本进行编码
        inputs = self.tokenizer(text=self.text, return_tensors='pt', padding='max_length', max_length=self.max_length).to(self.device)
        # 获取模型输出
        outputs = self.model(**inputs)
        print(outputs)
        return outputs


if __name__=='__main__':
    model = bert()
    
    print(model)
    for _ in range(1):
        y = model.forward()

    x = 1
