import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_images(images, interval=20, save_path=None):
    """
    Animate a list of images using FuncAnimation
    
    Parameters:
    - images: list of numpy arrays (H, W, 3) or (H, W)
    - interval: delay between frames in milliseconds
    - save_path: if provided, save animation to this path
    """
    fig, ax = plt.subplots()
    ax.axis('off')
    
    # Display first image
    if len(images[0].shape) == 3:
        img_display = ax.imshow(images[0])
    else:
        img_display = ax.imshow(images[0], cmap='gray')
    
    def update(frame):
        img_display.set_array(images[frame])
        return [img_display]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(images), 
        interval=interval, blit=True
    )
    
    # Save if path provided
    if save_path:
        anim.save(save_path, writer='pillow', fps=1000/interval)
    
    plt.show()
    return anim