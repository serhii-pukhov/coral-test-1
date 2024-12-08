import os
import pathlib
import sys
import requests
import time
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.adapters.detect import get_objects
from PIL import Image
from io import BytesIO

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite')
label_file = os.path.join(script_dir, 'models/coco_labels.txt')

def fetch_image(url):
    # Fetch the image from the URL
    response = requests.get(url)
        # Open the image using PIL
    return Image.open(BytesIO(response.content))

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

size = common.input_size(interpreter)
labels = dataset.read_label_file(label_file)

def classify_image(image):
    start_time = round(time.time() * 1000)

    # Run an inference
    common.set_input(interpreter, image)
    interpreter.invoke()
    objs = get_objects(interpreter, 0.1)[:4]

    # Print the result
    for obj in objs:
        print('%s: %.5f' % (labels.get(obj.id, obj.id), int(100 * obj.score)))

    print(f'took {round(time.time() * 1000) - start_time} ms')

urls = [
    "https://www.capitalleasegroup.com/wp-content/uploads/2020/10/vehicle-fleet-maintenance-services.jpg",
    "https://www.saddleman.com/media/wysiwyg/Fleet-Vehicles-For-Sale.jpg",
    "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1214430325.jpg"
]

if len(sys.argv) == 2:
    image = fetch_image(sys.argv[1]).convert('RGB').resize(size, Image.ANTIALIAS)
    classify_image(image)
else:
    # run with pre-defined urls
    for url in urls:
        image = fetch_image(url).convert('RGB').resize(size, Image.ANTIALIAS)
        classify_image(image)