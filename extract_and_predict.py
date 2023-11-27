import numpy as np
import json
import torch
import cv2
import re

import torchvision.transforms as transforms
from classification_model import ClassificationModel


NUMBER_OF_LABELS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\ndevice : " , device," \n")

label_classes = ["Background","MaizGood", "MaizBad"]
label_colors = [(255,0,0), (0,255,0), (0,0,255)] # R G B

print("Loading in classifier ... ", end = "")
classifier = ClassificationModel()
classifier.load_state_dict(torch.load("model_loadings.pt"))
classifier.to(device).eval()
print("DONE")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128,128)),
])
    

def output_activate(pred_logits):
    return torch.nn.Softmax(dim=1)(pred_logits)


def predict(x):
    x = x / 55.0                  # 55.0 is the max value in multispectral images from Videometer
    x = transform(x)[None,:,:,:]  # transform to target size and dimensions
    x = x.to(device).cuda()       # Set type to torch.cuda.floattensor type 
    pred_logits = classifier.forward(x)
    return output_activate(pred_logits).cpu().detach().numpy().ravel()





# Get the bounding box suggestions 
with open("tmp_box_suggestions.json") as fp:
    bbox_suggestions_loaded = json.load(fp) 


bbox_suggestions_with_labels = dict()
N = len(bbox_suggestions_loaded)
for i, (filename, bbox_arr) in enumerate(bbox_suggestions_loaded.items()):
    
    print(N, " - ", i+1, " - ", filename, end="\r")

    filename = filename.replace("images_maiz_rgb","images_maiz_multispectral").replace("jpg","npy").replace("\\","/")
    image_index = re.findall(r"\d+",filename)[-1]
    bbox_suggestions_with_labels[image_index] = []

    img = np.load(filename)
    # img_gray = cv2.cvtColor(np.uint8(img[:,:,2] / 55.0 * 255.0), cv2.COLOR_GRAY2BGR)

    for i, (x,y,w,h) in enumerate(bbox_arr):
        patch = img[y:y+h, x:x+w, :]

        label_likelihood = predict(patch)
        pred_label = int(label_likelihood.argmax())

        # img_gray = cv2.rectangle(img_gray, (x,y), (x+w,y+h), color=label_colors[pred_label], thickness=2)

        bbox_suggestions_with_labels[image_index].append({"label" : pred_label, "label_confidence_score" : float(label_likelihood[pred_label]), "bbox" : [x,y,w,h]})



with open("tmp_bbox_suggestions_with_labels.json","w") as fp:
    json.dump(bbox_suggestions_with_labels, fp)



