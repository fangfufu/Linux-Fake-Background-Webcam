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

    def classify(self, input_img, label_id):
        processed_img = (np.array(input_img.resize(self.tf_input_dim)
                                  )/255).astype('float32')
        processed_img = np.expand_dims(processed_img, 0)
        self.interpreter.set_tensor(self.input_details[0]['index'],
                                    processed_img)

        self.interpreter.invoke()
        tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
        label = np.zeros((257,257)).astype(int)
        mask = np.zeros((257,257)).astype(int)

        t0 = time.monotonic()
        for x in range(257):
            for y in range(257):
                for z in range(21):
                    prob = tensor[0][x][y][z]
                    if z == 0 or prob > max_prob:
                        max_prob = prob
                        label[x][y] = z
                if label[x][y] == label_id:
                    mask[x][y] = 1
                else:
                    mask[x][y] = 0
        td = time.monotonic() - t0
        print(td)
        output_img = Image.fromarray(np.uint8(mask)).convert('L')
        return np.array(output_img.resize(input_img.size))

if __name__ == "__main__":
    classifier = Classifier()
    input_img = Image.open("image.jpg")
    output_img = Image.fromarray(classifier.classify(input_img, 15))
    output_img.save("output.png")
