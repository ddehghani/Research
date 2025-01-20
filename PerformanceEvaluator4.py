from ObjectDetection import *
from Utils import *
import json
import os
import shutil
from tqdm import tqdm
import os
import random
import copy

"""
This class evaluates the performace gain through usage of YOLO heavy on the uncertain bboxes produced by YOLO light.
"""

PATH_TO_DATA_SOURCE = '/data/dehghani/EfficientVideoQueryUsingCP/coco/train2017'
PATH_TO_ANNOTATIONS = '/data/dehghani/EfficientVideoQueryUsingCP/coco/annotations/instances_train2017.json'
TEMP_DIR = 'EfficientVideoQueryUsingCP/test'
CONF_THRESHOLD = 0.125 # find dynamically or experimentally to make recall close to 1
IOU_THRESHOLD = 0.5
CALIBRATION_SIZE = 50
TEST_SIZE = 40
PADDING = 20
ALPHA = 0.15 # error rate
SHOW_DETAILS = True

def main():
    if os.path.isdir(TEMP_DIR):
        shutil.rmtree(TEMP_DIR) 
    os.mkdir(TEMP_DIR)

    
    with open(PATH_TO_ANNOTATIONS, 'r') as f:
        annotations = json.load(f)

    # Prepare calibration and test sets
    image_list = os.listdir(PATH_TO_DATA_SOURCE)
    random.shuffle(image_list)
    calibration_set = image_list[-CALIBRATION_SIZE:]
    del image_list[-CALIBRATION_SIZE:]
    test_set = image_list[-TEST_SIZE:]

    calibration_set = [os.path.join(PATH_TO_DATA_SOURCE, image) for image in calibration_set if image.endswith(".jpg")]
    test_set = [os.path.join(PATH_TO_DATA_SOURCE, image) for image in test_set if image.endswith(".jpg")]

    # Get ground truth annotations
    gt_cal_set_annotations = [filter_annotation(get_annotations(annotations, image)) for image in calibration_set]
    gt_test_set_annotations = [filter_annotation(get_annotations(annotations, image)) for image in test_set]
    for ann, image_path in zip (gt_test_set_annotations, test_set):
        annotateAndSave(image_path, ann, TEMP_DIR, f"{image_path.split('/')[-1].split('.')[0]}_gt.jpg")

    edge_cal_set_annotations = filter_annotation_bulk(detect_using_yolo_light(calibration_set, conf = CONF_THRESHOLD))
    edge_test_set_annotations = filter_annotation_bulk(detect_using_yolo_light(test_set, conf = CONF_THRESHOLD))
    edge_test_set_annotations_normal_confidence = filter_annotation_bulk(detect_using_yolo_light(test_set))
    
    for ann, image_path in zip (edge_test_set_annotations, test_set):
        annotateAndSave(image_path, ann, TEMP_DIR, f"{image_path.split('/')[-1].split('.')[0]}_edge.jpg")
    
    cloud_cal_set_annotations = filter_annotation_bulk(detect_using_yolo_heavy(calibration_set))
    cloud_test_set_annotations = filter_annotation_bulk(detect_using_yolo_heavy(test_set))


    # Calculate initial performance on calibration set
    print('\n\n**************** Initial Performance ****************\n\nCal set absolute:')
    calculate_performance([edge_cal_set_annotations, cloud_cal_set_annotations], ['Edge', 'Cloud'], gt_cal_set_annotations)

    # Calculate initial performance on calibration set
    # print('\n\nTest set absolute:')
    # calculate_performance([edge_test_set_annotations, cloud_test_set_annotations], ['Edge', 'Cloud'], gt_test_set_annotations)

    # save bounding boxes detected by edge
    for img_path, edge_annotations in zip(test_set, edge_test_set_annotations):
        image_name = img_path.split('/')[-1]
        image_file = Image.open(img_path) # open the image for processing 
        for index, annotation in enumerate(edge_annotations):
            bbox = copy.deepcopy(annotation.bbox)
            bbox[3] += bbox[1]
            bbox[2] += bbox[0]
            bbox[3] += 2 * PADDING 
            bbox[2] += 2 * PADDING 
            bbox[1] -= PADDING
            bbox[0] -= PADDING
            try:
                cropped_image = image_file.crop(bbox)
                cropped_image.save(os.path.join(TEMP_DIR, f'{image_name[:-4]}_{index}.jpg'))
            except:
                print(f'an error occured while processing image:{image_name}')
  

    # conformal prediction on edge using groud truth 
    prob_true_class = []
    for edge_annots, gt_annots in zip(edge_cal_set_annotations, gt_cal_set_annotations):
        for annot in edge_annots:
            prob_true_class.append(get_probability(annot, gt_annots))
        
    prob_true_class = np.array( prob_true_class)
    qhat = np.quantile(1 - prob_true_class, 1 - ALPHA) 
    prediction_sets = get_prediction_sets(edge_test_set_annotations, qhat)

    images_to_be_sent_to_server = []
    bbox_counter = 0
    for image_index, image_prediction_set in enumerate(prediction_sets):
        bbox_counter += len(image_prediction_set)
        for bbox_index, bbox_prediction_set in enumerate(image_prediction_set):
            if len(bbox_prediction_set) > 1:
                image_name = test_set[image_index].split('/')[-1][:-4]
                images_to_be_sent_to_server.append((image_index, bbox_index, os.path.join(TEMP_DIR, f'{image_name}_{bbox_index}.jpg')))
    
    print(f'\nPercentage of images sent to cloud = {100 * len(images_to_be_sent_to_server)/bbox_counter:.2f}')
    
    # can classify instead of detect instead
    cloud_selected_set_annotations = filter_annotation_bulk(detect_using_yolo_heavy([image[2] for image in images_to_be_sent_to_server]))
    filtered_annots = []
    for annots, image_path in zip(cloud_selected_set_annotations, [image[2] for image in images_to_be_sent_to_server]):
        img = Image.open(img_path)
        largest_acceptable_annot = None
        largest_acceptable_annot_size = 0
        for annot in annots:
            size = annot.bbox[2] * annot.bbox[3]
            if annot.bbox[0] > PADDING and \
            annot.bbox[1] and \
            annot.bbox[0] + annot.bbox[2] < img.size[0] - PADDING and \
            annot.bbox[1] + annot.bbox[3] < img.size[1] - PADDING and \
            size > largest_acceptable_annot_size:
                largest_acceptable_annot = annot
                largest_acceptable_annot_size = size
        filtered_annots.append([largest_acceptable_annot] if largest_acceptable_annot is not None else [])
                
    # print(len(cloud_selected_set_annotations))
    # print(len(cloud_selected_set_annotations[0]))
    # print(len(filtered_annots))
    # print(len(filtered_annots[0]))
    # return
    cloud_selected_set_annotations = filtered_annots
    corrected_edge_test_set_annotations = copy.deepcopy(edge_test_set_annotations)
    tobedeleted = []
    for (image_index, bbox_index, path), cloud_annot in zip(images_to_be_sent_to_server, cloud_selected_set_annotations):
        if len(cloud_annot) > 0 and corrected_edge_test_set_annotations[image_index][bbox_index].name != cloud_annot[0].name:
            if SHOW_DETAILS:
                print(f'Image Path: {path}\nEdge prediction: {edge_test_set_annotations[image_index][bbox_index].name}\nCloud prediction: {cloud_annot[0].name}\n')
            corrected_edge_test_set_annotations[image_index][bbox_index].name = cloud_annot[0].name
        elif len(cloud_annot) == 0:
            tobedeleted.append((image_index, bbox_index))
    
    test = []
    for image_index, image_annot in enumerate(corrected_edge_test_set_annotations):
        test_image = []
        for bbox_index, bbox_annot in enumerate(image_annot):
            if (image_index, bbox_index) not in tobedeleted:
                test_image.append(bbox_annot)
        test.append(test_image)
    
    print('\n\n************ After conformal prediction ************')
    print('\n\nTest set absolute:')
    calculate_performance([edge_test_set_annotations, test, edge_test_set_annotations_normal_confidence, cloud_test_set_annotations], ['Edge (low conf) Before', 'Edge (low conf) After', 'Edge normal conf', 'Cloud'], gt_test_set_annotations)
    print(f'Alpha = {ALPHA}, Padding = {PADDING}')
    # conformal prediction on edge using cloud

# pandas for seperate file
# seaborn (adjust the font to 2)

if __name__ == '__main__':
    main()