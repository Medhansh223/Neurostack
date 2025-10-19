import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from io import BytesIO

# Try loading model and classes
try:
    model = load_model("./app/models/resnet50_best.h5")
    classes = np.load("./app/models/label_classes.npy", allow_pickle=True)
    print("✅ ResNet model and label classes loaded successfully!")
except Exception as e:
    print("⚠️ Failed to load model or labels:", e)
    model, classes = None, None


def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def predict_category(image_bytes):
    if model is None or classes is None:
        return None

    img_array = preprocess_image(image_bytes)
    preds = model.predict(img_array, verbose=0)
    pred_class_index = int(np.argmax(preds, axis=1)[0])
    pred_class_name = str(classes[pred_class_index])
    return pred_class_name
