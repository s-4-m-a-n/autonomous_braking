import cv2
import numpy as np
from colored import fg, bg, attr

def find_ROI(f_width, f_height, traffic_objs,
             FOCAL_LENGTH, ROI_OFFSET_DISTANCE, MY_WIDTH):
    bboxes = {}
    for obj in traffic_objs:
        for _, param in obj.items():

            distance = param["distance"] + ROI_OFFSET_DISTANCE
            roi_pix_width = FOCAL_LENGTH * MY_WIDTH / distance

            center_x = f_width//2
            roi_x1 = center_x - roi_pix_width//2
#!!!!!!!!!!
            bbox = int(roi_x1), 0, int(roi_pix_width), f_height
            bboxes[param["distance"]] = bbox

    return bboxes


def plot_ROI(bbox, closed_objects_dist, frame):
    
    if len(closed_objects_dist) > 0:
        # if there is any closed object then plot ROI of the min_dist closed obj
        min_dist = min(closed_objects_dist)
    else:  
        # else plot ROI for the min dist obj even though that is within the ROI  
        # plot only the nearest bbox
        distances = [key for key in bbox.keys()]
        print(distances)
        min_dist = min(distances)
        
    print("min distance",min_dist)
    
    
    
    x, y, w, h = bbox[min_dist][0], bbox[min_dist][1],\
                 bbox[min_dist][2], bbox[min_dist][3]
  
    shapes = np.zeros_like(frame, np.uint8)

    cv2.rectangle(shapes, (x, y), (x+w, y+h), (0, 0, 255), cv2.FILLED)
    # mask
    out = frame.copy()
    alpha = 0.3
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

    return out


def plot_bounding_boxes(frame, results, class_names, CONFIDENCE):
    
    label_index, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    label = [class_names[int(index)] for index in label_index]

    n = len(label)

    f_width, f_height = frame.shape[1], frame.shape[0]

    for obj_index in range(n):
        row = coord[obj_index]
        if row[4] >= CONFIDENCE:
            x1, y1, x2, y2 = int(row[0]*f_width), int(row[1]*f_height),\
                             int(row[2]*f_width), int(row[3]*f_height),

            color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label[obj_index], (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    return frame


def get_traffic_object_info(frame_shape, results, class_names,
                            CONFIDENCE, AVG_OBJS_HEIGHT, FOCAL_LENGTH):
    
    label_index, coord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    labels = [class_names[int(index)] for index in label_index]
    f_width, f_height = frame_shape
    traffic_objs = []

    for obj_index in range(len(labels)):
        row = coord[obj_index]
        if row[4] >= CONFIDENCE:
            x1, y1, x2, y2 = int(row[0]*f_width), int(row[1]*f_height),\
                             int(row[2]*f_width), int(row[3]*f_height)
            pix_height = float(y2-y1)
            # FOCAL_LENGTH = (pix_height * OBJECT_MEASURED_DISTANCE) / BOTTLE_HEIGHT
            # print("----------------FOCAL LENGTH:-------------",FOCAL_LENGTH)
            # coeff = 110 / pix_height
            # print("coeff",coeff)
            
            distance = (AVG_OBJS_HEIGHT[labels[obj_index]] * FOCAL_LENGTH)/pix_height
            color = fg("red")
            print(color + "-----!! traffic object detected !!------"+ attr("reset"))
            print(fg("green") + f"{labels[obj_index]} distance : {distance}"+attr("reset"))

            traffic_objs.append({labels[obj_index]: {
                                                     "distance": distance,
                                                     "x_left_pos": x1,
                                                     "x_right_pos": x2
                                                    }
                                 })
    return traffic_objs
