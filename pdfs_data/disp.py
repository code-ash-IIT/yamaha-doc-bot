import io
import fitz
import os
import PIL.Image
import requests

def extract_images(pdf: fitz.Document, page: int, imgDir: str = 'img'):
    imageList = pdf[page].get_images()
    os.makedirs(imgDir, exist_ok=True)
    if imageList:
        print(page)
        for idx, img in enumerate(imageList, start=1):
            data = pdf.extract_image(img[0])
            with PIL.Image.open(io.BytesIO(data.get('image'))) as image:
                image.save(f'{imgDir}/{page}-{idx}.{data.get("ext")}', mode='wb')

def main():
    pdf = fitz.open('file_15-17.pdf')
    for page in range(pdf.page_count):
        extract_images(pdf, page)

if __name__ == '__main__':
    main()