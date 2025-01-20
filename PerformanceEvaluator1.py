from mmpretrain import list_models, ImageClassificationInferencer
import os
import random
import math
import numpy as np
from imagenet_classes import i2d
from ultralytics import YOLO
from tqdm import tqdm

""" Using pretrained models to classify images. """

PATH_TO_CROPPEDIMAGES = '/home/dehghani/EfficientVideoQueryUsingCP/data/coco/cropped'
PATH_TO_IMAGENET  = '/home/dehghani/EfficientVideoQueryUsingCP/data/imagenet-mini/train'

def main():
    ### Load images

    # # list all class labels that exists in imagenet 1000
    # with open('/home/dehghani/EfficientVideoQueryUsingCP/imagenet_labels.txt') as f:
    #     imagenet_labels = [line.split("'")[1]  for line in f]

    labeled_images = []
   
    # load cropped images whose label is in the imagenet list
    for file in os.listdir(PATH_TO_CROPPEDIMAGES):
        label = file.split('_')[-1][:-4]
        if file.endswith(".png") or file.endswith(".jpg"):
            labeled_images.append((os.path.join(PATH_TO_CROPPEDIMAGES, file), label))

    print(f'{len(labeled_images)} images in the source have labels that exists in imagenet ...')

    # load imagenet sample
    # for folder in os.listdir(PATH_TO_IMAGENET):
    #     label = i2d[folder]
    #     if not label in imagenet_labels:
    #         continue
    #     for file in os.listdir(os.path.join(PATH_TO_IMAGENET, folder)):
    #         labeled_images.append((os.path.join(PATH_TO_IMAGENET, folder, file), label))
        
    # print(f'{len(labeled_images)} images in imagenet-mini were loaded ...')


    ### List all the models available in mmpretrain 
    
    # for model_name in list_models():
    #     print(model_name)
    # return

    ### Divide all images into three categories train, cal, and test

    TRAIN_PERCENTAGE = 80
    CAL_PERCENTAGE = 0.01
    
    total_size = len(labeled_images)
    train_size = math.floor(TRAIN_PERCENTAGE / 100 * total_size)
    cal_size = math.floor(CAL_PERCENTAGE / 100 * total_size) 

    random.shuffle(labeled_images)
    train_set = labeled_images[:train_size]
    cal_set = labeled_images[train_size: train_size + cal_size]
    test_set = labeled_images[train_size + cal_size:]

    print(f'Data was divided between:\n Training: {len(train_set)} items\n Calibration: {len(cal_set)} items\n Test: {len(test_set)} items')

    ### Test to see which model performs better

    # potential_models = [
    # 'xcit-large-24-p16_3rdparty-dist_in1k',
    # 'xcit-large-24-p16_3rdparty-dist_in1k-384px',
    # 'wide-resnet101_3rdparty_8xb32_in1k',
    # 'wide-resnet50_3rdparty_8xb32_in1k',
    # 'vit-large-p14_dinov2-pre_3rdparty', 
    # 'vit-base-p16_mae-400e-pre_8xb128-coslr-100e_in1k',
    # 'vgg19bn_8xb32_in1k',
    # 'van-large_3rdparty_in1k',
    # 'twins-svt-large_3rdparty_16xb64_in1k',
    # 'twins-pcpvt-large_3rdparty_16xb64_in1k',
    # 'swinv2-large-w12_3rdparty_in21k-192px',
    # 'swin-large_in21k-pre-3rdparty_in1k',
    # 'riformer-m36_in1k',
    # 'resnext50-32x4d_8xb32_in1k',
    # 'resnet101_8xb32_in1k',
    # 'hornet-base-gf_3rdparty_in1k',
    # 'efficientnetv2-xl_in21k-pre_3rdparty_in1k'
    # ]

    # models_accuracy = []
    # for model in potential_models:
    #     try:
    #         inferencer = ImageClassificationInferencer(model, pretrained=True, device='cuda:0')
    #         predictions = inferencer([img for img, _ in cal_set], batch_size=8)
    #         gt_labels = [label for _, label in cal_set]
    #         accuracy = sum([1 for pred, lbl in zip(predictions, gt_labels) if lbl in pred['pred_class']]) / len(gt_labels)
    #         # print(f'Accuracy: {accuracy}')
    #         models_accuracy.append((model, accuracy))
    #     except:
    #         print(f'An error occured for model: {model}')
    
    # sorted_models_accuracy = sorted(models_accuracy, key=lambda x: x[1])
    # print(sorted_models_accuracy)

    # [('vgg19bn_8xb32_in1k', 0.12192622950819672), 
    #  ('resnext50-32x4d_8xb32_in1k', 0.15881147540983606), 
    #  ('resnet101_8xb32_in1k', 0.1721311475409836), 
    #  ('wide-resnet50_3rdparty_8xb32_in1k', 0.17315573770491804), 
    #  ('riformer-m36_in1k', 0.1864754098360656), 
    #  ('wide-resnet101_3rdparty_8xb32_in1k', 0.19057377049180327), 
    #  ('hornet-base-gf_3rdparty_in1k', 0.19262295081967212), 
    #  ('van-large_3rdparty_in1k', 0.19979508196721313), 
    #  ('twins-svt-large_3rdparty_16xb64_in1k', 0.19979508196721313), 
    #  ('twins-pcpvt-large_3rdparty_16xb64_in1k', 0.2028688524590164), 
    #  ('xcit-large-24-p16_3rdparty-dist_in1k', 0.21413934426229508), 
    #  ('xcit-large-24-p16_3rdparty-dist_in1k-384px', 0.25), 
    #  ('efficientnetv2-xl_in21k-pre_3rdparty_in1k', 0.2612704918032787), 
    #  ('swin-large_in21k-pre-3rdparty_in1k', 0.2725409836065574)]


    ### Conformal Prediction

    # accuracy about 65%
    # for ultralytics
    # inferencer = YOLO('runs/classify/train/weights/best.pt')
    # results = inferencer(cal_set)
    # for result in results:
    #     probs = result.probs  # Probs object for classification outputs
    #     print(f'{probs.top1}')
    #     print(f'{probs.top1conf}')
    #     print(f'{probs.data[18]}')

    inferencer = YOLO('/home/dehghani/EfficientVideoQueryUsingCP/data/coco/cropped_organized/runs/classify/train/weights/best.pt')
    sample_size = 1000
    prob_true_class = []
    for index, (img, label) in enumerate(cal_set):
        pred = inferencer(img)[0].cpu()
        class_dict = pred.names
        
        key_list = list(class_dict.keys())
        val_list = list(class_dict.values())
        position = val_list.index(label)
        gt_class_id = key_list[position]
        prob_true_class.append(pred.probs.data[gt_class_id].numpy().item())

        if index == sample_size - 1:
            break

    prob_true_class = np.array( prob_true_class)

    test_set_predictions = []
    class_dict = {}
    for index, (img, label) in enumerate(test_set):
        pred = inferencer(img)[0].cpu()
        class_dict = pred.names
        test_set_predictions.append(pred.probs.data.numpy())

        if index == sample_size - 1:
            break
            
    test_set_predictions = np.stack(test_set_predictions)

    for alpha in np.linspace(0, 1, 10, endpoint=False):
        prediction_sets = getPredictionSetsUltralytics(class_dict, test_set_predictions, prob_true_class, alpha)
        average_size = sum([len(s) for s in prediction_sets])/len(prediction_sets)
        cost = sum([1 for s in prediction_sets if len(s) > 1])
        actual_coverage = sum([ 1 for (_, gt_label), prediction_set in zip(test_set, prediction_sets) if gt_label in prediction_set])/sample_size
        print(f'For alpha: {round(alpha,1)}, the average prediction set size is: {round(average_size,2)}, the actual coverage is: {round(actual_coverage,2)}')

    return

    # for mmpretrain or mmdet
    inferencer = ImageClassificationInferencer('efficientnetv2-xl_in21k-pre_3rdparty_in1k', pretrained=True, device='cuda:0')
    cal_set_predictions = inferencer([img for img, _ in cal_set], batch_size=8)
    cal_set_gt_labels = [label for _, label in cal_set]
    prob_true_class = np.array([prediction['pred_scores'][imagenet_labels.index(gt_label)] for prediction, (_, gt_label) in zip(cal_set_predictions, cal_set)])
    
    test_set_predictions = inferencer([img for img, _ in test_set], batch_size=8)

    # calculate average prediction set size and verify coverage
    for alpha in np.linspace(0, 1, 10, endpoint=False):
        prediction_sets = getPredictionSets(imagenet_labels, test_set_predictions, prob_true_class, alpha)
        average_size = sum([len(s) for s in prediction_sets])/len(prediction_sets)
        actual_coverage = sum([ 1 for (_, gt_label), prediction_set in zip(test_set, prediction_sets) if gt_label in prediction_set])/len(test_set)
        print(f'For alpha: {round(alpha,1)}, the average prediction set size is: {round(average_size,2)}, the actual coverage is: {round(actual_coverage,2)}')

    
def getPredictionSets(imagenet_labels, test_set_predictions, prob_true_class, alpha):
    qhat = np.quantile(1 - prob_true_class, alpha) 
    prediction_sets = []
    for prediction in test_set_predictions:
        prediction_sets_indexes = 1 - np.array(prediction['pred_scores']) <= qhat
        prediction_sets.append([imagenet_labels[i] for i, v in enumerate(prediction_sets_indexes) if v == True])
    return prediction_sets

def getPredictionSetsUltralytics(class_dict, test_set_predictions, prob_true_class, alpha):
    qhat = np.quantile(1 - prob_true_class, alpha) 
    prediction_sets = []
    for prediction in test_set_predictions:
        prediction_sets_indexes = 1 - prediction <= qhat
        prediction_sets.append([class_dict[i] for i, v in enumerate(prediction_sets_indexes) if v == True])
    return prediction_sets


if __name__ == '__main__':
    main()