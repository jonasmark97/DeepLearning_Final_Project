# Script step-by-step for all the RGB images: 
# 1. Find box suggestions with selective search in a "single" mode.
# 2. Filter box suggestions with min and max box lengths and sizes
# 3. Plot suggestions on image and save suggestions in a json


# Links 
# PAPER : https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib
# EXPLANATION : https://learnopencv.com/selective-search-for-object-detection-cpp-python/#:~:text=Selective%20Search%20is%20a%20region,texture%2C%20size%20and%20shape%20compatibility.

import numpy as np
import glob
import cv2
import os
import json


def get_box_suggestions_with_selective_search(img_bgr):
    # Run selective search for box suggestions
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_bgr)
    ss.switchToSingleStrategy()
    boxes_suggestions = np.array(ss.process())

    # Filter box suggestions
    height, width, _ = img_bgr.shape
    min_len = np.max([height, width]) * 0.03 # 1% is around the equavalent of one diamond in the belt
    max_len = np.max([height, width]) * 0.1 # 10%
    
    box_min_len_mask = (min_len <= boxes_suggestions[:,2]) & (min_len <= boxes_suggestions[:,3])
    box_max_len_mask = (boxes_suggestions[:,2] <= max_len) & (boxes_suggestions[:,3] <= max_len)
    box_len_mask = box_min_len_mask & box_max_len_mask
    
    return boxes_suggestions[box_len_mask]


if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    image_paths = glob.glob("./Data/images_maiz_rgb/*.jpg")

    N = len(image_paths)
    d_box_suggestions = dict()
    if not "tmp_results" in os.listdir():
        os.mkdir("tmp_results") 

    for i, filename in enumerate(image_paths):
        print(N , " - ", i, end="\r")
        filename = filename.replace("\\","/")

        # Load in image
        img_rgb = cv2.imread(filename)

        # Run selective search for box suggestions and filter
        boxes_suggestions = get_box_suggestions_with_selective_search(img_rgb)


        # Plot box suggestions and save box suggestions
        for b in boxes_suggestions:
            img = cv2.rectangle(img, tuple(b[:2]), tuple(b[:2]+b[2:]), color=(255,0,255), thickness=2)

        cv2.imwrite("./tmp_results/" + os.path.basename(filename), img)


        d_box_suggestions[filename] = boxes_suggestions.tolist()



    with open("./Data/box_suggestions.json","w") as fp:
        json.dump(d_box_suggestions, fp)


