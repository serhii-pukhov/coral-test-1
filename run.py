import os
import pathlib
import sys
import requests
import time
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

def fetch_image(url):
    # Fetch the image from the URL
    response = requests.get(url)
        # Open the image using PIL
    return Image.open(BytesIO(response.content))

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

size = common.input_size(interpreter)

def classify(url):
    image = fetch_image(url).convert('RGB').resize(size, Image.ANTIALIAS)
    start_time = round(time.time() * 1000)

    # Run an inference
    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)

    # Print the result
    labels = dataset.read_label_file(label_file)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))

    print(f'took {round(time.time() * 1000) - start_time} ms')

urls = [
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSah1gbgT4_Wng5Kt7blk3CrCic9NMAVga7Mw&s",
    "https://www.monde-animal.fr/wp-content/uploads/2020/04/fiche-animale-monde-animal-girafe.jpg",
    "https://cdn.britannica.com/26/162626-050-3534626F/Koala.jpg",
    "https://dogsinc.org/wp-content/uploads/2021/08/extraordinary-dog.png",
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcStNw98RLUpDvKzb0evAxJVspBhoBboJSFPwQ&s",
    "https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2020/01/28150457/French-Bulldog-puppy-sitting-on-the-floor-in-the-living-room.jpg"
]

for url in urls:
    classify(url)

# if len(sys.argv) == 2:
#     image = fetch_image(sys.argv[1]).convert('RGB').resize(size, Image.ANTIALIAS)
# else:
#     image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
#
# start_time = round(time.time()*1000)
#
# # Run an inference
# common.set_input(interpreter, image)
# interpreter.invoke()
# classes = classify.get_classes(interpreter, top_k=1)
#
# # Print the result
# labels = dataset.read_label_file(label_file)
# for c in classes:
#     print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
#
# print(f'took {round(time.time()*1000) - start_time} ms')