__author__ = 'chris'
from PIL import Image

pil_im = Image.open('empire.jpg')
pil_im.show()

pil_im_gray = pil_im.convert('L')
pil_im_gray.show()

pil_im_gray.save("empire_gray.png")

