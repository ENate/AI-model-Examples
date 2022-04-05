import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
# For downloading the image.
import matplotlib.pyplot as plt
import tempfile
from six.moves.urllib.request import urlopen
from six import BytesIO

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps


# displays image
def display_image(image)
    """Image to display
    """
    fig = plt.figure(figsize=(10, 20))
    plt.grid(False)
    plt.imshow(image)
    
# load and sized image
def load_and_size_image(url, new_width=256, new_height=256, display=False):
    _, filename = tempfile.mkstemp(suffix='.jpg')
    
    
    
if __name__ == "__main__":
    print_version()
    