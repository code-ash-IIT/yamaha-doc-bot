from sentence_transformers import SentenceTransformer, util

img_model= SentenceTransformer('clip-ViT-B-32', device='cpu') #cuda:3
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', device='cpu') #cuda:3
