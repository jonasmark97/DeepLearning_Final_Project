# Two step script
# 1. Uses non-maximum suppression with different IoU thresholds and confidence threshold to filter out duplicate predictions
# 2. Gets true labels and confidence scores to create y_pred and y_true array to save in tmp_nms_statistics.json

# Precision-Recall curves and mean Average Precision for each IoU threshold are shown in plot_nms_statistics.json 

import numpy as np
import json
import cv2
import os


def calculate_iou(bbox1, bbox2):
    # Calculate the Intersect over Union for two bounding boxes
    # bbox are in the format : [x, y, w, h]

    # Coordinates of the area of intersection.
    ix1 = np.maximum(bbox1[0], bbox2[0])
    iy1 = np.maximum(bbox1[1], bbox2[1])
    ix2 = np.minimum(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    iy2 = np.minimum(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
     
    # Intersection height and width.
    i_height = iy2 - iy1 + 1
    i_width = ix2 - ix1 + 1

    # Boxes do not intersect
    if i_height < 0 or i_width < 0: 
        return 0.0

    # Get areas
    area_of_intersection = i_height * i_width

    area_of_bbox1 = bbox1[2] * bbox1[3]
    area_of_bbox2 = bbox2[2] * bbox2[3] 
     
    area_of_union = area_of_bbox1 + area_of_bbox2 - area_of_intersection
     
    # Calculate IoU
    iou = area_of_intersection / area_of_union
     
    return iou


def remove_duplicates(iou_mat, iou_mat_thresh, pred_duplicate_from_gt=True):
    # Remove the duplicates boxes within another one. 

    # pred_duplicate_from_gt
    #   TRUE - when we will remove lower IoU for duplicate prediction boxes in a ground truth boxes
    #   False - when we will remove lower IoU for duplicate ground truth boxes in a prediction boxes

    if not pred_duplicate_from_gt:
        iou_mat = np.transpose(iou_mat)
        iou_mat_thresh = np.transpose(iou_mat_thresh)

    indexes_multiple = np.where(np.sum(iou_mat_thresh, axis=1) > 1)[0]  # Get indexes of boxes who have multiple pointed to them
    for index in indexes_multiple:
        i_argmax = np.argmax(iou_mat[index]) # Max iou
        indexes_valid = np.where(iou_mat_thresh[index])[0]
        indexes_valid_without_max = indexes_valid[indexes_valid != i_argmax]

        # Remove the box with less iou
        for j in indexes_valid_without_max:
            iou_mat[index,j] = 0.0 
            iou_mat_thresh[index, j] = 0
    
    # Transpose back
    if not pred_duplicate_from_gt:
        iou_mat_thresh = np.transpose(iou_mat_thresh)

    return iou_mat_thresh



def get_ypred_ytrue(bbox_gt_masked,bbox_nms,pred_label_confidence_score_nms):
    # Get y_true labels and their respective y_pred scores. 
      
    
    # Create IoU matrix to compare all predicted boxes to the ground truth boxes
    iou_pred_threshold = 0.5 # IoU threshold which the predicted box has to fulfill to be counted as the same label

    number_of_gt_boxes = bbox_gt_masked.shape[0]
    number_of_pred_boxes = bbox_nms.shape[0]

    iou_mat = np.zeros((number_of_gt_boxes, number_of_pred_boxes), dtype=float)
        
    for j in range(number_of_gt_boxes):
        for k in range(number_of_pred_boxes):
            iou_mat[j,k] = calculate_iou(bbox_gt_masked[j], bbox_nms[k])

    iou_mat_thresh = (iou_mat > iou_pred_threshold).astype(int)

    # Correct so that only one prediction box matches one ground truth box 
    # - This is done by removing boxes who have less IoU than others 

    # Remove multiple prediction boxes within the same ground truth box
    iou_mat_thresh_pred = remove_duplicates(iou_mat, iou_mat_thresh, pred_duplicate_from_gt=True)

    # Remove multiple ground truth boxes within the same prediction box
    iou_mat_thresh_gt = remove_duplicates(iou_mat, iou_mat_thresh, pred_duplicate_from_gt=False)

    # Combine
    iou_mat_thresh2 = iou_mat_thresh_pred & iou_mat_thresh_gt 

    gt_boxes_found = np.any(iou_mat_thresh2, axis=1)   # Ground truth boxes who were found
    pred_boxes_found = np.any(iou_mat_thresh2, axis=0) # Prediction boxes who hit a ground truth

    # Get IoU between found boxes
    iou_pred_boxes_found = np.max(iou_mat * iou_mat_thresh2, axis=0)[pred_boxes_found] 

    assert np.sum(gt_boxes_found) == np.sum(pred_boxes_found), "Only one prediction per found ground truth box, gt:"+str(np.sum(gt_boxes_found))+"-pred:"+str(np.sum(pred_boxes_found)) # sanity check
    
    # Get the scores of the predicted found boxes 
    n_gt_not_found = np.sum(~gt_boxes_found)
    not_found_groundtruth_boxes_scores = ([] if n_gt_not_found == 0 else np.zeros(n_gt_not_found,dtype=float))
    found_groundtruth_boxes_scores = pred_label_confidence_score_nms[pred_boxes_found == 1]
    false_positive_pred_boxes_scores = pred_label_confidence_score_nms[pred_boxes_found == 0]
    y_pred = np.concatenate((not_found_groundtruth_boxes_scores, found_groundtruth_boxes_scores, false_positive_pred_boxes_scores))
    
    # Create the respective true label array
    all_groundtruth_boxes = np.ones(number_of_gt_boxes)
    n_false_positives = y_pred.size - number_of_gt_boxes
    false_positives = ([] if n_false_positives == 0 else np.zeros(n_false_positives))
    y_true = np.append(all_groundtruth_boxes,false_positives)

    assert len(y_pred) == len(y_true), "y_pred:"+str(len(y_pred))+"-y_true:"+str(len(y_true)) # Sanity check

    return y_pred, y_true, pred_boxes_found, gt_boxes_found, iou_pred_boxes_found




if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__))) # Set directory of file as current working directory

    NUMBER_OF_LABELS = 3

    label_dict = {"Background" : 0, "MaizGood" : 1, "MaizBad" : 2}
    label_colors = [(255,0,0), (0,255,0), (0,0,255)] # R G B


    # Get the bounding box ground truth
    with open("Data/bbox_groundtruth.json") as fp:
        bbox_dict_groundtruth = json.load(fp) 

    # Get the bounding box suggestions
    with open("Data/bbox_suggestions_with_labels.json") as fp:
        bbox_dict_suggestions = json.load(fp) 


    # Get image sizes
    img_height, img_width, _ = np.load("./Data/images_maiz_multispectral/0.npy").shape


    # Initialize statistics 
    d_statistics = dict()
    confidence_threshold = 0.8

    for iou_threshold in np.arange(1,10)/10:
        print("")
        d_statistics[str(iou_threshold)] = [{"y_true" : [], "y_pred" : [] } for _ in range(NUMBER_OF_LABELS-1)] # Skipping background

        N = len(bbox_dict_suggestions)
        for i, (image_index, bbox_arr_with_labels) in enumerate(bbox_dict_suggestions.items()):
            print("IoU threshold:", iou_threshold, "of 0.9 - image number", i+1, "of", N,  end="\r")

            # Get ground truth 
            if not image_index in bbox_dict_groundtruth:  
                bbox_groundtruth = np.array([])
                label_groundtruth = np.array([])
            else:
                bbox_groundtruth = np.array([elem["bbox"] for elem in bbox_dict_groundtruth[image_index]])
                label_groundtruth = np.array([label_dict[elem["label"]] for elem in bbox_dict_groundtruth[image_index]])

            # Get predictions 
            n = len(bbox_arr_with_labels)
            bbox_arr = np.zeros((n,4))
            pred_label_arr = -np.ones(n)
            pred_label_confidence_score = np.zeros(n)

            bbox_arr = np.array([d["bbox"] for d in bbox_arr_with_labels])
            pred_label_arr = np.array([d["label"] for d in bbox_arr_with_labels])
            pred_label_confidence_score = np.array([d["label_confidence_score"] for d in bbox_arr_with_labels])

            img_plot = cv2.imread("./Data/images_maiz_rgb/"+str(image_index)+".jpg")

            # Perform NMS and get y_true and y_pred
            for label in range(1,NUMBER_OF_LABELS):
                
                label_filter = (pred_label_arr == label)
                if not np.any(label_filter):
                    continue

                bbox_arr_filtered = np.array(bbox_arr)[label_filter].astype(int)
                pred_label_confidence_score_filtered = pred_label_confidence_score[label_filter]
                
                idxs = cv2.dnn.NMSBoxes(bbox_arr_filtered, pred_label_confidence_score_filtered, confidence_threshold, iou_threshold)   # NMS

                bbox_nms = bbox_arr_filtered[idxs]
                pred_label_confidence_score_nms = pred_label_confidence_score_filtered[idxs]
                


                # # Compare to the ground truth            
                label_gt_mask = label_groundtruth == label
                bbox_gt_masked = bbox_groundtruth[label_gt_mask]

                y_pred, y_true,_,_ = get_ypred_ytrue(bbox_gt_masked,bbox_nms,pred_label_confidence_score_nms)

                
                d_statistics[str(iou_threshold)][label-1]["y_pred"].extend(y_pred.tolist())
                d_statistics[str(iou_threshold)][label-1]["y_true"].extend(y_true.tolist())


                
                # Plot NMS suggestions
                # for (x,y,w,h), label_conf_score in zip(bbox_nms, pred_label_confidence_score_nms):
                #     img_plot = cv2.rectangle(img_plot, (x,y), (x+w,y+h), color=label_colors[label], thickness=2)
                #     img_plot = cv2.putText(img_plot, str(round(label_conf_score,3)), (x,y),  cv2.FONT_HERSHEY_SIMPLEX , fontScale=1, color=(255,0,255), thickness=2, lineType=cv2.LINE_AA)

                # cv2.imwrite("./tmp/"+str(image_index)+"_nms_box_suggestions.jpg",img_plot)
                
                # Plot ground truth and predicted label on it
                # for x,y,w,h in bbox_gt_masked:
                #     label_color = tuple([int(c * 1/2) for c in label_colors[label]])  # Little darker
                #     img_plot = cv2.rectangle(img_plot, (x,y), (x+w,y+h), color=label_color, thickness=2)
                    
                # for i in np.where(pred_boxes_found)[0]:
                #     x,y,w,h = bbox_nms[i]
                #     img_plot = cv2.rectangle(img_plot, (x,y), (x+w,y+h), color=label_colors[label], thickness=2)
                #     img_plot = cv2.putText(img_plot, "conf:" + str(round(pred_label_confidence_score_nms[i],3)) , (x,y),  cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.7, color=label_colors[label], thickness=2, lineType=cv2.LINE_AA)
                #     img_plot = cv2.putText(img_plot, "IoU:" + str(round(np.max(iou_mat[:,i]),3)) , (x,y+20),  cv2.FONT_HERSHEY_SIMPLEX , fontScale=0.7, color=label_colors[label], thickness=2, lineType=cv2.LINE_AA)

                # cv2.imwrite("./tmp/IoU_"+str(iou_threshold)+"_"+str(image_index)+"_gt_and_pred.jpg",img_plot)


    


                
    filename = "/Data/nms_statistics.json"
    with open(filename,"w") as fp:
        json.dump(d_statistics, fp)
        
    print("\n\nSaved results in " + filename)

















