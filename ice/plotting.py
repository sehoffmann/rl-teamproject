import itertools
from PIL import Image

def save_gif(filename, images_numpy, fps=50, duration=None):
    if duration is None:
        frame_length_ms = (1/fps) * 1000
    else:
        frame_length_ms = 1000 * duration / len(images_numpy)

    images = [Image.fromarray(img) for img in images_numpy]
    images[0].save(filename, save_all=True, append_images=images[1:], duration=frame_length_ms, loop=0)

def save_games(filename, games, repeats=30, fps=50, duration=None):
    images = [game_imgs[0:1]*repeats + game_imgs  for game_imgs in games] # repeat last frame
    images = list(itertools.chain.from_iterable(images))
    save_gif(filename, images, fps=fps, duration=duration)
