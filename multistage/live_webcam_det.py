import cv2
import numpy as np
import torch
import src.asl_nn as asl_nn
import src.asl_img_process as img_proc

MODEL_FILE = "model_multistage.pt"
IMG_FILE = "q.jpg"

model = torch.load(MODEL_FILE)

img = cv2.imread(IMG_FILE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))
img = img_proc.preprocess_img(img)
img_flat = np.reshape(img, (784,))
img_flat = np.expand_dims(img_flat, 0)
in_tensor = torch.tensor(img_flat)
out_tensor = model.forward(in_tensor)
pred = torch.argmax(out_tensor)
print(pred, chr(pred + ord('a')))
print(pred.detach().numpy())



# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()

#     img_processed = img_proc.preprocess_img(frame)
#     img_tensor = torch.tensor(img_processed)
#     res = model.forward(img_tensor)

#     pred = torch.argmax(res)
#     letter = chr(pred + int('a'))
#     print(letter)

# cap.release()


