from transformers import AutoProcessor, BlipForConditionalGeneration
import torch

print('Loading blip','-'*10,'\n')
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda:0")