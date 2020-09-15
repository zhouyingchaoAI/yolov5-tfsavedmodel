import tensorflow as tf
tf.enable_eager_execution()
import cv2
import numpy as np
import random
import colorsys

# if not tf.__version__.startswith('1'):
#     import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
# video_path      = "./video/IMG_9142.MP4"
video_path      = "/home/zyc/Desktop/现场分析视频/192.168.1.101_01_20200914094437397.mp4"
spp_flag = True

path = '../weights/model/1'

print(tf.test.is_built_with_cuda)

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)



def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    cv2.imwrite('show.jpg', resized)
    # return the resized image
    return resized


def draw_bbox(image, image_size, bboxes, labels, scores, thread = 0.3):

    num_classes = 80
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    fontScale = 0.5

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for k, score in enumerate(scores):
        if score > thread:
            box = bboxes[k]
            class_ind = int(labels[k])
            bbox_color = colors[(class_ind)]
            # if(class_ind == 0):
            #     bbox_color = [255, 0, 0]
            # else:
            #     bbox_color = [0, 255, 0]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (int(box[1]*image_w), int(box[0]*image_h)), (int(box[3]*image_w), int(box[2]*image_h))
            cv2.rectangle(image, c1, c2, bbox_color, 2)


            if 1:
                bbox_mess = '%d: %.2f' % (class_ind, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

                cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image

with tf.device('/GPU:0'):

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], path)
        graph = tf.get_default_graph()

        vid = cv2.VideoCapture(video_path)
        out = cv2.VideoWriter((video_path.split('.', -1)[0] + '_detect.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (960, 540))
        while True:
            # frame = cv2.imread("data/samples/test.jpg")
            # return_value = True

            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = frame[0:640, 0:640]
                # frame = cv2.resize(frame, (416, 416))
            else:
                raise ValueError("No image!")
            frame_size = frame.shape[:2]
            print(frame_size)
            data_set = []
            data_set.clear()
            frame_data = np.asarray(frame)
            # frame_data = np.expand_dims(frame_data, 0)

            image_jp = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1]
            image_jp = np.squeeze(image_jp, 1).tostring()
            data_set.append(image_jp)
            x = sess.graph.get_tensor_by_name('encoded_image_tensor:0')
            y1 = sess.graph.get_tensor_by_name('strided_slice_12:0')
            y2 = sess.graph.get_tensor_by_name('strided_slice_13:0')
            y3 = sess.graph.get_tensor_by_name('strided_slice_14:0')
            y4 = sess.graph.get_tensor_by_name('Tile:0')
            # y1 = sess.graph.get_tensor_by_name('strided_slice_18:0')
            # y2 = sess.graph.get_tensor_by_name('strided_slice_19:0')
            # y3 = sess.graph.get_tensor_by_name('strided_slice_20:0')
            # y4 = sess.graph.get_tensor_by_name('Tile:0')


            data_set = np.expand_dims(np.asarray(data_set), axis=0)

            scores, classes, boxes, m4 = sess.run([y1, y2, y3, y4],
                              feed_dict={x: data_set})
            image = draw_bbox(frame, frame_size, boxes[0], classes[0], scores[0], 0.4)
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("output/2020_7_9_tools_001.jpg", result)
            out.write(result)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    out.release()



