import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import numpy as np
from torchviz import make_dot

# https://github.com/pooya-mohammadi/yolov5-gradcam/blob/master/main.py

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
    # show the model architecture.
    print(model)
    # print(type(model))


    # Define a custom model class to capture intermediate outputs
    class YOLOv7WithIntermediate(torch.nn.Module):
        def __init__(self, model, inject_layer):
            super(YOLOv7WithIntermediate, self).__init__()
            self._model = model

            self._inject_layer = inject_layer
        
            # Register a forward hook to capture the output from the specific layer
            self._register_hooks()

        def _register_hooks(self):
            def hook_fn(module, input, output):
                self.feature_map = output

            def back_hook_fn(module, grad_input, grad_output):
                self.gradients = grad_output[0]

            # Register the hook to the specified layer
            target_layer = self._model.model[self._inject_layer].conv
            target_layer.register_forward_hook(hook_fn)
            target_layer.register_backward_hook(back_hook_fn)

        def forward(self, x, augment=None):
            x = self._model(x, augment=augment)
            # return x, self.feature_map  # Return both final output and intermediate output
            return x
        
        def __setattr__(self, name, value):
            if name == 'traced':
                # Delegate attribute setting to the model object
                setattr(self._model, name, value)
            else:
                super().__setattr__(name, value)

            
        # Forward all other attributes and method calls to the original model
        def __getattr__(self, name):
            forward_attrs = set([
                'stride', 'names', 'model', 'traced'
            ])
            if name in forward_attrs:
                return getattr(self._model, name)
            else:
                return super().__getattr__(name)
    
    class ModTracedModel(TracedModel):
        def forward(self, x, augment=False, profile=False):
            out, extra = self.model(x)
            out = self.detect_layer(out)
            return out, extra

    # Register the hook to the desired layer
    layer_index = 50  # Example: layer index 50
    model = YOLOv7WithIntermediate(model, layer_index)
    # model.to(device)

    # if trace:
    #     print('trace')
    #     # model = TracedModel(model, device, opt.img_size)
    #     model = ModTracedModel(model, device, opt.img_size)
    #     print('trace done')

    if half:
        model.half()  # to FP16
    # exit()
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.requires_grad_()

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        # with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        # pred, intermediate_outputs = model(img, augment=opt.augment)[0]
        pred = model(img, augment=opt.augment)[0].requires_grad_()
        t2 = time_synchronized()

        # print(f"Intermediate output shape at layer {layer_index}: {intermediate_outputs[0].shape}")
        # print(f"Intermediate output mean values at layer {layer_index}: {intermediate_outputs[0].mean(dim=1)}")

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        for pred_bbox in pred:
            print(pred_bbox)
            # pred_bbox = _pred[0, 1, :]
            objectness_score = pred_bbox[4]
            class_scores = pred_bbox[5:]

            class_idx = torch.argmax(class_scores)
            print(class_scores[class_idx])
            print(objectness_score)

            # target = class_scores[class_idx] * objectness_score
            target = class_scores[class_idx]
            print(target)

            dot = make_dot(target, params=dict(list(model.named_parameters()) + [('img_tensor', img)]))
            dot.format = 'png'
            dot.render('computational_graph')

            model.zero_grad()
            target.backward(retain_graph=True)

            feat_map = model.feature_map.detach().numpy()
            gradients = model.gradients.detach().numpy()
            # compute
            weights = np.mean(gradients, axis=(1, 2))
            grad_cam = np.zeros(feat_map.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                grad_cam += w * feat_map[i]

            grad_cam = np.maximum(grad_cam, 0)
            # Resize to the original image size
            grad_cam = cv2.resize(grad_cam, (img.shape[2], img.shape[3]))

            # Convert original image to RGB
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)

            # Overlay heatmap on the original image
            superimposed_img = heatmap * 0.4 + img

            p = Path(path)  # to Path
            save_path = str(save_dir / 'heatmap_{}'.format(p.name))  # img.jpg
            # save the result
            cv2.imwrite(save_path, np.uint8(superimposed_img))
            # plt.imshow()
            # plt.axis('off')
            # plt.show()
            exit()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # get pred
        print(pred)
        exit()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
