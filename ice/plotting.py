from PIL import Image

def save_gif(filename, states, speed=1000):
    images = [Image.fromarray(state) for state in states]
    images[0].save(filename, save_all=True, append_images=images[1:], duration=(1/50)*speed, loop=0)