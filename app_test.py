from fastapi import FastAPI

from pdf2image import convert_from_path
from PIL import Image

import numpy as np
import pickle

import variables

app = FastAPI()
app.x=10
@app.get("/")
async def root():
    print('Starting to create!')
    create_image_embeddings(['file_15-17.pdf','/home/vinayak/Desktop/IIT/8/dl/project/yamaha-doc-bot/pdfs_data/file_15-17.pdf'])
    print('Created!')
    return {"greeting":"Roto",'var':app.x}

# @app.get("/home")
# async def home():
#     create_image_embeddings(['file_15-17.pdf','/home/vinayak/Desktop/IIT/8/dl/project/yamaha-doc-bot/pdfs_data/file_15-17.pdf'])
#     return {"greeting":"Home",'var':app.x,'model':variables.img_model}


def create_image_embeddings(pdf_name_path):
        pdf_name,pdf_path=pdf_name_path
        def pdf_to_images(pdf_path):
            images = convert_from_path(pdf_path,dpi=100) # https://pdf2image.readthedocs.io/en/latest/reference.html#pdf2image.pdf2image.convert_from_bytes
            # dpi=200,transparent=False,first_page,last_page 
            return images

        images = pdf_to_images(pdf_path)
        print('Starting to load model!')
        # img_model = variables.img_model
        print('model loaded!')

        # img_embeddings = img_model.encode(images)
        # embeddings = np.array(img_embeddings).astype('float32')
        
        # # np.savez(f'../../../local_data/{pdf_name}.npz', embeddings)
        # data = {
        # 'pdf_name': pdf_name,
        # 'images': images,
        # 'embeddings': embeddings
        # }
        # with open(f'local_data/{pdf_name}.pkl', 'wb') as f:
        #     pickle.dump(data, f)
