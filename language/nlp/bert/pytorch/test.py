from transformers import BertTokenizer, BertModel
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from transformers import pipeline
# unmasker = pipeline('fill-mask', model='bert-base-uncased')
# unmasker("Hello I'm a [MASK] model.")

def bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased").to(device)
    text = "Replace me by any text you'd like."
    inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=256).to(device)
    
    return model,inputs



# model_path = 'language/nlp/bert/pytorch/vocab'
# tokenizer_path = 'language/nlp/bert/pytorch/vocab'
# tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
# model = BertModel.from_pretrained(model_path).to(device)
# text = "Hello, how are you?"
# inputs = tokenizer(text, return_tensors='pt', padding='max_length', max_length=256).to(device)




model, inputs=bert()
import time
start_time = time.time()
iterations = 128

for _ in range(iterations):
    # 执行推断
    with torch.no_grad():
        # outputs = model(**inputs)
        
        output = model(**inputs)
        # model = bert() 
        # print(outputs)
# 记录结束时间
end_time = time.time()
# 计算执行时间
execution_time = end_time - start_time
latency = execution_time / iterations * 1000
FPS = 1000 / latency
print(f"FPS: {FPS:.2f}")