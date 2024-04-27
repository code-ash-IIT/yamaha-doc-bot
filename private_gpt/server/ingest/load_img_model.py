from sentence_transformers import SentenceTransformer, util

print('Loading img_model','-'*10,'\n')
img_model= SentenceTransformer('clip-ViT-B-32', device='cpu') #cuda:3
print('Loaded img_model','-'*10,'\n')
