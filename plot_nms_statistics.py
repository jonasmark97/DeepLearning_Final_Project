# Creates Precision-Recall curves for each IoU threshold and label both individually and all combined 
# 
# Saves the results into a folder called "nms_results" where this script is located

import os
import json 
import numpy as np 
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


os.chdir(os.path.dirname(os.path.abspath(__file__)))

target_folder = "./Data"

if not target_folder in os.listdir():
    os.mkdir(target_folder)

label_arr = ["Background", "MaizGood", "MaizBad"]

with open("./Data/nms_statistics.json") as fp:
    stats = json.load(fp)


color_arr = list(mcolors.TABLEAU_COLORS.values())


fig_all, axs_all = plt.subplots(1,2,figsize=(12,8))
mAP_arr = np.zeros(9, dtype=float)

nms_stats_calculated_dict = dict()
for i, iou_threshold_str in enumerate((np.arange(1,10)/10).astype(str)):
    print("iou : 0.9 - ", str(iou_threshold_str),end="\r")
    
    # fig, axs = plt.subplots(1,2,figsize=(12,8))
    avg_prec_score_arr = [0.0, 0.0]

    nms_stats_calculated_dict[str(iou_threshold_str)] = [{"recall":[],"precision":[]}, {"recall":[],"precision":[]}]
    for label_index_without_background in range(2):        
        y_pred = np.array(stats[iou_threshold_str][label_index_without_background]["y_pred"])
        y_true = np.array(stats[iou_threshold_str][label_index_without_background]["y_true"])

        # Sort by y_predicted confidence score
        sorted_index = np.argsort(y_pred)
        y_true = y_true[sorted_index]
        y_pred = y_pred[sorted_index]

        # Calculate precision-recall curve and AP
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        avg_prec = average_precision_score(y_true, y_pred)

        # Plot indivudal 
        # axs[label_index_without_background].plot(recall,precision,label="IoU: " + iou_threshold_str, color=color_arr[i])
        # axs[label_index_without_background].set_xlabel("Recall")
        # axs[label_index_without_background].set_ylabel("Precision")
        # axs[label_index_without_background].set_title("Label : "+label_arr[label_index_without_background+1]+" - AP : " + str(round(avg_prec,4)))
        avg_prec_score_arr[label_index_without_background] = avg_prec

        # Plot all together        
        axs_all[label_index_without_background].plot(recall,precision, color=color_arr[i])

        # Put in dictionary
        nms_stats_calculated_dict[str(iou_threshold_str)][label_index_without_background]["precision"] = precision.tolist()
        nms_stats_calculated_dict[str(iou_threshold_str)][label_index_without_background]["recall"] = recall.tolist()

    mAP_arr[i] = round(np.mean(avg_prec_score_arr), 4)
    
    # fig.suptitle("mAP : " + str(mAP_arr[i]))   
    # fig.legend() 
    # fig.savefig(target_folder+"/IoU_"+iou_threshold_str+"_precVsRecall.jpg")
    # plt.close(fig)

for label_index_without_background in range(2):
    axs_all[label_index_without_background].set_title("Label : " + label_arr[label_index_without_background+1])
fig_all.suptitle("Precision-Recall curves for different IoU thresholds")   
fig_all.legend(["IoU: " + str(iou) + " - mAP: " + str(mAP) for iou,mAP in zip(np.arange(1,10)/10, mAP_arr) ]) 
fig_all.savefig(target_folder+"/all_precVsRecall.jpg")
plt.close(fig_all)


nms_stats_calculated_dict["iou_final_stats"] = ["IoU: " + str(iou) + " - mAP: " + str(mAP) for iou,mAP in zip(np.arange(1,10)/10, mAP_arr) ]
with open(target_folder + "/nms_train_stats_calculated.json","w") as fp:
    json.dump(nms_stats_calculated_dict, fp)





