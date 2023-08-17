import itertools
from PIL import Image

def save_gif(filename, images_numpy, speed=1000):
    images = [Image.fromarray(img) for img in images_numpy]
    images[0].save(filename, save_all=True, append_images=images[1:], duration=(1/50)*speed, loop=0)

def save_games(filename, games, repeats=30, speed=1000):
    images = [game_imgs[0:1]*repeats + game_imgs  for game_imgs in games] # repeat last frame
    images = itertools.chain.from_iterable(images)
    save_gif(filename, images, speed=speed)
