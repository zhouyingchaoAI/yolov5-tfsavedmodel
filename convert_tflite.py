import argparse
import tensorflow as tf
# if not tf.__version__.startswith('1'):
#     import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from tensorflow import keras
# Path to the frozen graph file

model = keras.models.load_model('./weights/tf_model/')
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.allow_custom_ops = False
converter.experimental_new_converter = True
tflite_model = converter.convert()
# Write the converted model to disk
open("weights/yolov5s_tf.tflite", "wb").write(tflite_model)



