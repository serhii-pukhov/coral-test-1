import os
import pathlib
import sys
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
from io import BytesIO

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'model.tflite')
label_file = os.path.join(script_dir, 'labels.txt')
image_file = os.path.join(script_dir, 'test1.jpg')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# Resize the image
size = common.input_size(interpreter)
if len(sys.argv) == 2:
    image = fetch_image(sys.argv[1])
else:
    image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

# Run an inference
common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

# Print the result
labels = dataset.read_label_file(label_file)
for c in classes:
    print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

def fetch_image(url):
    # Fetch the image from the URL
    response = requests.get(url)
        # Open the image using PIL
    return Image.open(BytesIO(response.content))
