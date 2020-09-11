import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *

import tensorflow as tf
# if tf.__version__.startswith('1'):
#     tf.enable_eager_execution()
from tensorflow import keras
# from keras import backend as K
import numpy as np


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    if weights[0].split('.')[-1] == 'pt':
        backend = 'pytorch'
    elif weights[0].split('.')[-1] == 'tflite':
        backend = 'tflite'
    else:
        backend = 'tensorflow'

    if backend == 'pytorch':
        model = attempt_load(weights, map_location=device)  # load FP32 model
    elif backend == 'tensorflow':
        if tf.__version__.startswith('1'):
            sess = tf.Session()
            loaded = tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING], weights[0])
            tf_input = loaded.signature_def['serving_default'].inputs['input_1']
            tf_output = loaded.signature_def['serving_default'].outputs['tf__detect']
        else:
            model = keras.models.load_model(weights[0])
    elif backend == 'tflite':
        # Load TFLite model and allocate tensors.                           
        interpreter = tf.lite.Interpreter(model_path=opt.weights[0])
        interpreter.allocate_tensors()                                      
                                            
        # Get input and output tensors.                                     
        input_details = interpreter.get_input_details()                     
        output_details = interpreter.get_output_details()                   

    if backend == 'pytorch':
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size

    if half and backend == 'pytorch':
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    if backend == 'pytorch':
        names = model.module.names if hasattr(model, 'module') else model.names
    else:
        names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    if backend == 'pytorch':
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    elif backend == 'tensorflow':
        if tf.__version__.startswith('1'):
            _ = sess.run(tf_output.name, feed_dict={tf_input.name: img.permute(0, 2, 3, 1).cpu().numpy()})
        else:
            _ = model(img.permute(0, 2, 3, 1).cpu().numpy())
    elif backend == 'tflite':
        input_data = img.permute(0, 2, 3, 1).cpu().numpy()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()                                         
        output_data = interpreter.get_tensor(output_details[0]['index'])
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        if backend == 'pytorch':
            def hook_wrapper(i):
                def pytorch_hook(model, input, output):
                    # print(model.__class__.__name__)
                    np.save('./tensors/pytorch_%d_%s.npy' % (i, model.__class__.__name__), output.cpu().numpy())
                return pytorch_hook

            # for i, m in enumerate(model.model[:-1]): 
            #     m.register_forward_hook(hook_wrapper(i))
            # pred = model(img, augment=opt.augment)[0]
            # for i, m in enumerate(model(img, augment=opt.augment)[1]):
            #     np.save('./pytorch_%d.npy' % i, m.cpu().numpy())

            # np.save('./pytorch_out.npy', pred.cpu().numpy())

        elif backend == 'tensorflow':
            if tf.__version__.startswith('1'):
                pred = sess.run(tf_output.name, feed_dict={tf_input.name: img.permute(0, 2, 3, 1).cpu().numpy()})
                pred = torch.tensor(pred)
            else:
                res = model(img.permute(0, 2, 3, 1).cpu().numpy())
                pred = res[0].numpy()
                pred = torch.tensor(pred)
                # inp = model.input
                # outputs = [layer.output for layer in model.layers]
                # layer_names = [layer.name for layer in model.layers]
                # keras.backend.set_learning_phase(0)
                # functors = [keras.backend.function([inp], out) for out in outputs]
                # layer_outs = [func(img.permute(0, 2, 3, 1).cpu().numpy()) for func in functors]
                # for l_name, l_out in zip(layer_names[:-1], layer_outs[:-1]):
                #      np.save('./tensors/' + l_name + '.npy', l_out)

                # for i, m in enumerate(res[1]):
                #     np.save('./tf_%d.npy' % i, res[1][i].numpy())

                # np.save('./tf_out.npy', res[0].numpy())

        elif backend == 'tflite':
            input_data = img.permute(0, 2, 3, 1).cpu().numpy()
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()                                         
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = torch.tensor(output_data)


        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect()
