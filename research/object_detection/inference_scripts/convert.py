
import tensorflow as tf

# Path to the frozen graph file
graph_def_file = '../tor_models/tier_2/tier_2_faster_rcnn_inception_v2_coco_2018_01_28/inference_graph/frozen_inference_graph.pb'
# A list of the names of the model's input tensors
input_arrays = ['Cast']
# A list of the names of the model's output tensors
output_arrays = ['raw_detection_scores']
# Load and convert the frozen graph
converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays) #, input_shapes={"image_tensor":[1,600,600,3]}
tflite_model = converter.convert()
# Write the converted model to disk
open("converted_model.tflite", "wb").write(tflite_model)

# Graph = tf.GraphDef()
# File = open("../tor_models/tier_2/tier_2_faster_rcnn_inception_v2_coco_2018_01_28/inference_graph/frozen_inference_graph.pb","rb")
# Graph.ParseFromString(File.read())

# # for Layer in Graph.node:
# #     print(Layer.name)
# print(Graph.node[0].name)
# print(Graph.node[-1].name)
