import matplotlib.pyplot as plt
import numpy as np
import gc

def display_frames_dynamic(frame_generator, width=256, height=240, pause_time=0.001):
    """
    Dynamically displays frames from a generator in a pyplot figure.

    Each frame overwrites the previous frame. Frames are discarded after displaying
    to keep memory usage constant. Uses a stable update loop suitable for long-running streams.

    Parameters:
        frame_generator: Generator yielding RGB frames of shape (height, width, 3)
        width (int): Width of each frame in pixels
        height (int): Height of each frame in pixels
        pause_time (float): Time in seconds to pause per frame (default 0.001)
    """
    plt.ion()  # enable interactive mode
    fig, ax = plt.subplots()
    img_display = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))
    ax.axis('off')
    fig.show()

    try:
        for frame in frame_generator:
            img_display.set_data(frame)        # overwrite previous frame
            fig.canvas.draw_idle()             # lightweight redraw
            plt.pause(pause_time)              # process GUI events

            del frame                          # discard frame
            gc.collect()                       # free memory
    finally:
        plt.close(fig)

