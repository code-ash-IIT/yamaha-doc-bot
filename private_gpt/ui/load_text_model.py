from sentence_transformers import SentenceTransformer, util

print('Loading text_model','-'*10,'\n')
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', device='cpu') #cuda:3
print('Loaded text_model','-'*10,'\n')
