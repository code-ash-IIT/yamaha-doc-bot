from sentence_transformers import SentenceTransformer, util

print('Loading img_model')
img_model= SentenceTransformer('clip-ViT-B-32', device='cpu') #cuda:3
print('Loaded img_model')

print('Loading text_model')
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', device='cpu') #cuda:3
print('Loaded text_model')
