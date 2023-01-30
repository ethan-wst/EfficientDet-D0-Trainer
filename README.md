# EfficientDet-D0-Trainer

&ensp;&thinsp;This installation is recomended to completed within an Anaconda virtual enviroment for Python 3.9

	https://www.anaconda.com/products/individual
	
## Install TensorFlow PIP package

  - Install TensorFlow package

		pip install --ignore-installed --upgrade tensorflow==2.5.0
		
  - Verify installation
	
		python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
		
  - The above code should return a print-out ending with (the first number will vary):
 
	`tf.Tensor(-54.834015, shape=(), dtype=float32)`
 
## Install TensorFlow Object Detection API 
	
 ### Protobuf Installation
 
 - Download the latest realease from (e.g. protoc-all-#.##.#.tar.gz)
 	
		https://github.com/protocolbuffers/protobuf/releases
		
 - Extract contents into the of the `tar.gz` file into a directory `<PATH_TO_PB>` of your choice

 - Add `<PATH_TO_PB>\bin` to your `PATH`

   - Open `~/ .bashrc` file in a text editor
	
	
   - Insert export syntax to the end of the file 
	
		 export PATH="<PATH_TO_PB>/bin:$PATH"
		
   - Restart the terminal, `cd` into `models/research/` and run
	
		 protoc object_detection/protos/*.proto --python_out=.
		 
 ### COCO API Installation

 Although `pycocotools` should get installed along with the Object Detection API, this install can fail for various reasons and is simpler to install before hand

 - Download COCOAPI into a directory of your choice

	 	git clone https://github.com/cocodataset/cocoapi.git
	 
 - Make the `pycocotools` directory
	
		cd cocoapi/PythonAPI
		make
		
 - Copy the `pycocotools` directory into the `models/reseach/` directory
		
		cp -r pycocotools <PATH_TO_MODELS>/models/research/

 ### Install Object Detection API

 - From within `<PATH_TO_MODELS>/models/research/` run

		cp object_detection/packages/tf2/setup.py .
		python -m pip install --use-feature=2020-resolver .
		
 ### Test Installation

 - From within `<PATH_TO_MODELS>/models/research/` run

		python object_detection/builders/model_builder_tf2_test.py
		
 - Once the program is finished you should observe a print-out similiar to
 &ensp;&thinsp; <Details>
	```
	[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
	[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
	INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update): 0.0s
	I0608 18:49:13.183754 29296 test_util.py:2102] time(__main__.ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update): 0.0s
	[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
	[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
	INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold): 0.0s
	I0608 18:49:13.186750 29296 test_util.py:2102] time(__main__.ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold): 0.0s
	[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
	[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
	INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_model_config_proto): 0.0s
	I0608 18:49:13.188250 29296 test_util.py:2102] time(__main__.ModelBuilderTF2Test.test_invalid_model_config_proto): 0.0s
	[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
	[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
	INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_second_stage_batch_size): 0.0s
	I0608 18:49:13.190746 29296 test_util.py:2102] time(__main__.ModelBuilderTF2Test.test_invalid_second_stage_batch_size): 0.0s
	[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
	[ RUN      ] ModelBuilderTF2Test.test_session
	[  SKIPPED ] ModelBuilderTF2Test.test_session
	[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
	INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor): 0.0s
	I0608 18:49:13.193742 29296 test_util.py:2102] time(__main__.ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor): 0.0s
	[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
	[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
	INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_meta_architecture): 0.0s
	I0608 18:49:13.195241 29296 test_util.py:2102] time(__main__.ModelBuilderTF2Test.test_unknown_meta_architecture): 0.0s
	[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
	[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
	INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_ssd_feature_extractor): 0.0s
	I0608 18:49:13.197239 29296 test_util.py:2102] time(__main__.ModelBuilderTF2Test.test_unknown_ssd_feature_extractor):	0.0s
	[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
	----------------------------------------------------------------------
	Ran 24 tests in 29.980s
	
	OK (skipped=1)
	```
 </Details>

## Preparing for Training Job

 ### Create TensorFlow Records
 
 - Install `pandas` package
 
	- Anaconda

			conda install pandas
	- Pip
	
			pip install pandas
 
 - Using the `generate_tfrecord.py` in `preprocessing/` run
 
 		# Creates train data:
 		python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record
		
		#Creates test data:
		python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

  - There should now be a `test.record` and `train.record` in  `trainer/annotaions/`

## Training Model

### Begin Training
This is to start the training job, a training job may take several hours depending on the number of images to train on

- `cd` into the `trainer/` directory and run 

		python model_main_tf2.py --model_dir=models/my_ssd_effdet_d0 --pipeline_config_path=models/my_ssd_effdet_d0/pipeline.config
		
### Monitor Training

This is only if you would like to use TensorBoard to moniter training progress (not required)

- Open a seperate command terminal from the terminal training the model

- Activate Anaconda virtual enviroment (if you are using one to train the model)

- `cd` into the `trainer/` directory

- Then run 

		tensorboard --logdir=models/my_ssd_effdet_d0

- This command should return a print out with a URL on the last line, the URL should be the same as below but can differ

		http://localhost:6006/
		
## Export Trained Model

The training software can be automatically stopped by adjusting the **Need to FInish Pipeline stuff**




 
