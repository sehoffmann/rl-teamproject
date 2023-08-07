from PIL import Image

def save_gif(filename, images_numpy, speed=1000):
    images = [Image.fromarray(img) for img in images_numpy]
    images[0].save(filename, save_all=True, append_images=images[1:], duration=(1/50)*speed, loop=0)