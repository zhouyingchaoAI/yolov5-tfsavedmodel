# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7
"""Export inception model given existing training checkpoints.
The model is exported as SavedModel with proper signatures that can be loaded by
standard tensorflow_model_server.
"""

from __future__ import print_function

import os.path
import tensorflow as tf
import argparse
if not tf.__version__.startswith('1'):
    import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

IMAGE_SIZE = 640
path = 'weights/yolov5s.pb'


parser = argparse.ArgumentParser(description='Generate a saved model.')

parser.add_argument("--model_version", default="1", help="Version number of the model.", type=str)
parser.add_argument("--output_dir", default="../weights/model", help="export model directory", type=str)
parser.add_argument("--pb_dir", default="../weights/helmet_opti.pb", help="Directory where to read training checkpoints.", type=str)
parser.add_argument("--input_tensor", default="encoded_image_tensor:0", help="input tensor", type=str)
parser.add_argument("--output_tensor", default="Identity:0", help="Concat_383:0", type=str)
parser.add_argument("--class_num", default=3, help="class num", type=int)


args = parser.parse_args()

def export():
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    serialized_tf_example = tf.placeholder(tf.string, shape=[None, 1], name='encoded_image_tensor')
    images, dh_b, dw_b, resized_height, resized_width = tf.map_fn(preprocess_image, serialized_tf_example,
                                      (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
    # serialized_tf_example = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='encoded_image_tensor')
    # images = tf.map_fn(preprocess_image, serialized_tf_example, tf.float32)

    with tf.gfile.FastGFile(args.pb_dir, 'rb') as f:  # 加载冻结图模型文件
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, input_map={'x':images}, name='')  # 导入图定义
    sess.run(tf.global_variables_initializer())



    with tf.Session() as sess:

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        output_path = os.path.join(
        tf.compat.as_bytes(args.output_dir),
        tf.compat.as_bytes(str(args.model_version)))
        print('Exporting trained model to', output_path)
        builder = tf.saved_model.builder.SavedModelBuilder(output_path)
        print([n.name for n in tf.get_default_graph().as_graph_def().node])
        input_tensor = tf.get_default_graph().get_tensor_by_name(args.input_tensor)
        output_tensor = tf.get_default_graph().get_tensor_by_name(args.output_tensor)
        class_num = args.class_num

        batch_size = tf.shape(input_tensor)[0]
        output_tensor = tf.reshape(tf.convert_to_tensor(output_tensor), shape=[batch_size, -1, 5 + class_num])

        num_tensor = tf.cast(tf.tile([[100]], multiples=[batch_size, 1]), dtype=tf.int32)

        classes_tensor = tf.argmax(output_tensor[:, :, 5:], axis=2)
        scores_tensor = output_tensor[:, :, 4]
        raw_boxs_tensor = output_tensor[:, :, 0:4]
        print(raw_boxs_tensor.shape)
        boxs_tensor_com = raw_boxs_tensor
        print(boxs_tensor_com.shape)

        dh_b = tf.reshape(dh_b, [-1, 1])
        dw_b = tf.reshape(dw_b, [-1, 1])

        boxs_tensor_minx = raw_boxs_tensor[:, :, 0] - raw_boxs_tensor[:, :, 2] * 0.5
        boxs_tensor_miny = raw_boxs_tensor[:, :, 1] - raw_boxs_tensor[:, :, 3] * 0.5
        boxs_tensor_maxx = raw_boxs_tensor[:, :, 0] + raw_boxs_tensor[:, :, 2] * 0.5
        boxs_tensor_maxy = raw_boxs_tensor[:, :, 1] + raw_boxs_tensor[:, :, 3] * 0.5

        boxs_tensor_minx = (boxs_tensor_minx - dw_b) / resized_width
        boxs_tensor_miny = (boxs_tensor_miny - dh_b) / resized_height
        boxs_tensor_maxx = (boxs_tensor_maxx - dw_b) / resized_width
        boxs_tensor_maxy = (boxs_tensor_maxy - dh_b) / resized_height

        boxs_tensor_minx = tf.expand_dims(boxs_tensor_minx, 2)
        boxs_tensor_miny = tf.expand_dims(boxs_tensor_miny, 2)
        boxs_tensor_maxx = tf.expand_dims(boxs_tensor_maxx, 2)
        boxs_tensor_maxy = tf.expand_dims(boxs_tensor_maxy, 2)

        boxs_tensor_com = tf.concat([boxs_tensor_miny, boxs_tensor_minx, boxs_tensor_maxy, boxs_tensor_maxx], 2)
        scores_tensor_map = tf.expand_dims(scores_tensor, 2)
        classes_tensor_map = tf.expand_dims(classes_tensor, 2)
        classes_tensor_map = tf.cast(classes_tensor_map, dtype=tf.float32)
        nms_tensor_map = tf.concat([boxs_tensor_com, scores_tensor_map, classes_tensor_map], 2)
        tensor_result = tf.map_fn(preprocess_nms, nms_tensor_map, tf.float32, name="map2")

        # with tf.control_dependencies([tf.print(tensor_result[0, :, 5])]):
        tmpt = tensor_result[:, :, 5]
        scores_tensor_info = tf.saved_model.utils.build_tensor_info(tmpt)
        classes_tensor_info = tf.saved_model.utils.build_tensor_info(tf.cast(tensor_result[:, :, 4], dtype=tf.float32))
        boxes_tensor_info = tf.saved_model.utils.build_tensor_info(tensor_result[:, :, 0:4])
        raw_boxes_tensor_info = tf.saved_model.utils.build_tensor_info(raw_boxs_tensor)
        num_tensor_info = tf.saved_model.utils.build_tensor_info(num_tensor)


        inputs_tensor_info = tf.saved_model.utils.build_tensor_info(
            input_tensor)

        tensor_info_inputs = {
            'inputs': inputs_tensor_info
        }
        print(scores_tensor_info, inputs_tensor_info, classes_tensor_info, boxes_tensor_info, num_tensor_info)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs=tensor_info_inputs,
                # outputs=tensor_info_outputs,
                outputs={
                    'detection_scores': scores_tensor_info,
                    'detection_classes': classes_tensor_info,
                    'detection_boxes': boxes_tensor_info,
                    # 'raw_boxes': raw_boxes_tensor_info,
                    'num_detections': num_tensor_info,
                },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            ))

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants
                    .DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    prediction_signature,
            },
        )

        builder.save()
        print('Successfully exported model to %s' % args.output_dir)


def preprocess_image(image_buffer):
  """Preprocess JPEG encoded bytes to 3D float Tensor."""

  # Decode the string as an RGB JPEG.
  # Note that the resulting image contains an unknown height and width
  # that is set dynamically by decode_jpeg. In other words, the height
  # and width of image is unknown at compile-time.
  image = tf.image.decode_jpeg(image_buffer[0], channels=3)
  # image_buffer = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
  image_resized = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], preserve_aspect_ratio=True)
  resized_height = tf.shape(image_resized)[0]
  resized_width = tf.shape(image_resized)[1]
  dh = IMAGE_SIZE - tf.shape(image_resized)[0]
  dw = IMAGE_SIZE - tf.shape(image_resized)[1]
  # r = tf.shape(image_resized)[0] / tf.shape(image)[0]
  dh_b = dh // 2
  dh_e = dh - dh_b
  dw_b = dw // 2
  dw_e = dw - dw_b
  # print(dh_b, dh_e, dw_b, dw_e)
  image_paded = tf.pad(image_resized, paddings=[[dh_b, dh_e], [dw_b, dw_e], [0, 0]])
  # image_paded = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method=0)
  # image_paded = tf.transpose(image_paded, perm=[2, 0, 1])
  return (image_paded/255, tf.cast(dh_b, dtype=tf.float32), tf.cast(dw_b, dtype=tf.float32),
        tf.cast(resized_height, dtype=tf.float32), tf.cast(resized_width, dtype=tf.float32))

def preprocess_nms(nms_tensor_map):
  """Preprocess nms float Tensor."""
  boxs_tensor_box = nms_tensor_map[:, 0:4]
  scores_tensor = nms_tensor_map[:, 4]
  classes_tensor = nms_tensor_map[:, 5]
  boxs_tensor_indices = tf.image.non_max_suppression(boxs_tensor_box, scores_tensor, 100, 0.5)
  boxs_tensor_indices = tf.reshape(boxs_tensor_indices, shape=[100])

  boxs_tensor_com = tf.gather(boxs_tensor_box, boxs_tensor_indices)
  # boxs_tensor_com = tf.reshape(boxs_tensor_com, shape=[-1, 100, 4])
  classes_tensor = tf.gather(classes_tensor, boxs_tensor_indices)
  classes_tensor = tf.expand_dims(classes_tensor, 1)
  scores_tensor = tf.gather(scores_tensor, boxs_tensor_indices)
  scores_tensor = tf.expand_dims(scores_tensor, 1)
  result = tf.concat([boxs_tensor_com, classes_tensor, scores_tensor], 1)

  return result


def main(unused_argv=None):
  # test()
  export()


if __name__ == '__main__':
  tf.app.run()


