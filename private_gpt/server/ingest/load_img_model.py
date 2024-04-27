from sentence_transformers import SentenceTransformer, util
import torch

print(f"Available CUDA devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
print('Loading img_model','-'*10,'\n')
img_model= SentenceTransformer('clip-ViT-B-32', device='cuda:0') #cuda:3
print('Loaded img_model','-'*10,'\n')
