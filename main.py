# waymo_perception_demo.py

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# 1. Load pretrained vision model
model = models.detection.fasterrcnn_resnet50_fpn(
    weights="DEFAULT"
)

model.eval()


# 2. Load driving image
image_path = "driving_scene.jpg"

image = Image.open(image_path).convert("RGB")


transform = transforms.Compose([
    transforms.ToTensor()
])


tensor_image = transform(image)


# 3. Run inference
with torch.no_grad():
    prediction = model([tensor_image])


boxes = prediction[0]["boxes"]
scores = prediction[0]["scores"]
labels = prediction[0]["labels"]


# 4. Display detections
fig, ax = plt.subplots()

ax.imshow(image)


for box, score, label in zip(
    boxes,
    scores,
    labels
):

    if score > 0.7:

        x1,y1,x2,y2 = box

        width = x2-x1
        height = y2-y1


        rect = patches.Rectangle(
            (x1,y1),
            width,
            height,
            linewidth=2,
            fill=False
        )

        ax.add_patch(rect)

        ax.text(
            x1,
            y1,
            f"class {label.item()}",
        )


plt.show()