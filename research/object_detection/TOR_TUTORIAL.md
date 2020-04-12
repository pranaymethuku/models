# TOR Tutorial to train + run a model

**Platform**: Linux (tested on CentOS 7 and Ubuntu 18.04)

**NOTE**: You do not need to have root permissions to do anything, as long as you are able to get Anaconda installed on the machine you’re working on (that is, as long as you get the conda setup script on your machine somehow)

Assuming Anaconda 3 is already installed, first create a new conda environment called `tor` to hold all dependencies of this project.

```bash
conda create --name tor python=3.6
```

Everything you do will be on this environment, so to activate it you will need to run:

```bash
conda activate tor
```

Using conda, install the dependencies by running the following commands:

```bash
conda install protobuf
conda install lxml
conda install Cython
conda install jupyter
conda install matplotlib
conda install pandas
pip install opencv-python
conda install tensorflow=1.15
```

Download the Tensorflow Object Detection API from TOR’s fork [pranaymethuku/models](https://github.com/pranaymethuku/models) of  [tensorflow/models](https://github.com/tensorflow/models).

Although we need the entire models folder, our main focus is the `object_detection` directory.
The next step is to compile the protobuf files within the `models/research` directory:

```bash
cd <path_to_your_tensorflow_installation>/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

While still in the models/research folder, set the PYTHONPATH variable.

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

You can test your setup using the following command:

```bash
python object_detection/builders/model_builder_test.py
```

Which should show something like the following output:

```bash
...............
----------------------------------------------------------------------
Ran 15 tests in 0.123sOK
```

Now you have everything set up!

The next step is to get some kind of data to use. For this tutorial, we’ll skip the data-collection/post-processing/labeling step and assume you have a `train` and `test` folder already created with correspondingly labeled `.xml` files for your .jpgs. As well as an appropriately configured `labelmap.pbtxt` file.

Use `xml_to_csv.py` to generate labeled .csv files for `train` and `test`

```bash
python xml_to_csv.py --source=<path_to_train> --csv-file=<path_to_train_labels.csv>
```

Use `generate_tfrecord.py` to generate .tfrecord files from the .csv files for `train` and `test`

```bash
python generate_tfrecord.py -c=<path_to_train_labels.csv> -r=<path_to_train.tfrecord> -i=<path_to_train> -l=<path_to_labelmap.pbtxt>
```

Download a pre-trained model file (from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)). For the purposes of this tutorial we’ll use [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz).

Copy the corresponding .config file (faster_rcnn_inception_v2_coco.config) from `object_detection/samples/config/` into the directory for your specific tier.

Modify the num_classes to the number of classes you’re using
Modify the path to the checkpoint to be the checkpoint from the downloaded model folder (just model.ckpt - don’t worry about the .index or .meta shit) 
Modify the paths to the train.record and test.record and the corresponding labelmap.pbtxt file
https://towardsdatascience.com/custom-object-detection-using-tensorflow-from-scratch-e61da2e10087
