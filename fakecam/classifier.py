#!/usr/bin/env python3
import time
import numpy as np
import tensorflow as tf
from PIL import Image

class Classifier:
    def __init__(self,
                 model_path="lite-model_deeplabv3_1_metadata_2.tflite",
                 label_array = ["background", "aeroplane", "bicycle", "bird",
                                "boat", "bottle", "bus", "car", "cat", "chair",
                                "cow", "dining table", "dog", "horse",
                                "motorbike", "person", "potted plant", "sheep",
                                "sofa", "train", "tv"]):
        self.interpreter =  tf.lite.Interpreter(model_path)
        self.label_array = label_array
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.tf_input_dim = self.input_details[0]['shape'][1:3]
        self.tf_output_dim = self.input_details[0]['shape'][1:3]

    def classify(self, input_img, selected_label):
        processed_img = (np.array(input_img.resize(self.tf_input_dim)
                                  )/255).astype('float32')
        processed_img = np.expand_dims(processed_img, 0)

        self.interpreter.set_tensor(self.input_details[0]['index'],
                                    processed_img)
        self.interpreter.invoke()
        tensor = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        label = np.argmax(tensor, 2)
        mask = (label == selected_label)
        mask_img = Image.fromarray(np.uint8(mask)).convert('L')
        return mask_img.resize(input_img.size)

if __name__ == "__main__":
    classifier = Classifier()
    input_img = Image.open("image.jpg")
    t0 = time.monotonic()
    output_img = classifier.classify(input_img, 15)
    output_img.save("output.png")
    td = time.monotonic() - t0
    print(td)
