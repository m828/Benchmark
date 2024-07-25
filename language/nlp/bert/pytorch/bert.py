
import os
import time
import torch
import requests
from transformers import BertTokenizer, BertModel

from thop import profile

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
        self.inputs = self.tokenizer(text=self.text, return_tensors='pt', padding='max_length', max_length=self.max_length).to(self.device)
    
    def count_parameters_and_flops(self):

        flops, _ = profile(self.model, (self.inputs.input_ids, self.inputs.attention_mask), verbose=False)
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return flops / 1e9 * 2,  params / 1e6
    
    def forward(self):      
        # 获取模型输出
        outputs = self.model(**self.inputs)

        return outputs


if __name__=='__main__':
    model = bert()
    
    print(model)
    for _ in range(10):
        y = model.forward()


    flops, params = model.count_parameters_and_flops()
    print(flops, params)
    # import time
    # iterations = None
    # if iterations is None:
    #     elapsed_time = 0
    #     iterations = 100
    #     while elapsed_time < 1:
    #         torch.cuda.synchronize()
    #         torch.cuda.synchronize()
    #         t_start = time.time()
    #         for _ in range(iterations):
    #             model.forward()
    #         torch.cuda.synchronize()
    #         torch.cuda.synchronize()
    #         elapsed_time = time.time() - t_start
    #         iterations *= 2
    #     FPS = iterations / elapsed_time
    #     iterations = int(FPS * 6)

    # print('=========Speed Testing=========')
    # torch.cuda.synchronize()
    # torch.cuda.synchronize()
    # t_start = time.time()
    # for _ in range(iterations):
    #     model.forward()
    # torch.cuda.synchronize()
    # torch.cuda.synchronize()
    # elapsed_time = time.time() - t_start
    # latency = elapsed_time / iterations * 1000
    # FPS = 1000 / latency
    # print(f"FPS: {FPS:.2f}")
