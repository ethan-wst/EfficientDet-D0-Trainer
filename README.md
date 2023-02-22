# EfficientDet-D0-Trainer
## Installation and Training Through Buff-Code

- Ensure buffpy is added to your enviroment, from within the `buff-code/` directory run
	
		source buffpy/buff.bash
		
 ### Clone Repository and Install Dependencies
 
 - `cd` into `<PATH_TO_BUFFCODE>/buffpy/scripts` and run
		
		bash effdet_trainer.bash
		
 ### Append ProtoBuf to PATH
 
 This step is not required but is a nice QOL change as you will not need to re-export protobuf to path in every new terminal.
 
 - Add `<PATH_TO_PB>\bin` to your `PATH`

   - Open `~/ .bashrc` file in a text editor
	
	
   - Append export syntax to the end of the file 
	
		 export PATH="$PATH:<PATH_TO_BUFFCODE>/src/efficientdet-d0-trainer/software/protobuf-21.12/bin"

 ### Test Dependecies Installation
 
  #### Test TensorFlow Installation
  
  - Verify installation
	
		python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
		
  - The above code should return a print-out ending with (the first number will vary):
 
	`tf.Tensor(-54.834015, shape=(), dtype=float32)`

  #### Test Object Detection API Installation
  
  - From within `efficientdet-d0-trainer/models/research/` run

		python3 object_detection/builders/model_builder_tf2_test.py
		
 - Once the program is finished you should observe a print-out similiar to
 &ensp;&thinsp;<Details>
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

### Preparing for Training Job

#### Adjusting Training Job

  - `cd` into the `efficientdet-d0-trainer/trainer/models/my_ssd_effdet_do/`
  - Open the `pipeline.config` file in a text editor
  - Ensure all directory paths are correct for your machine and all variables are adjusted
&ensp;&thinsp; <Details>
	```python
	model {
	  ssd {
		num_classes: 4
		image_resizer {
		  keep_aspect_ratio_resizer {
			min_dimension: 512
			max_dimension: 512
			pad_to_max_dimension: true
		  }
		}
		feature_extractor {
		  type: "ssd_efficientnet-b0_bifpn_keras"
		  conv_hyperparams {
			regularizer {
			  l2_regularizer {
				weight: 3.9999998989515007e-05
			  }
			}
			initializer {
			  truncated_normal_initializer {
				mean: 0.0
				stddev: 0.029999999329447746
			  }
			}
			activation: SWISH
			batch_norm {
			  decay: 0.9900000095367432
			  scale: true
			  epsilon: 0.0010000000474974513
			}
			force_use_bias: true
		  }
		  bifpn {
			min_level: 3
			max_level: 7
			num_iterations: 3
			num_filters: 64
		  }
		}
		box_coder {
		  faster_rcnn_box_coder {
			y_scale: 1.0
			x_scale: 1.0
			height_scale: 1.0
			width_scale: 1.0
		  }
		}
		matcher {
		  argmax_matcher {
			matched_threshold: 0.5
			unmatched_threshold: 0.5
			ignore_thresholds: false
			negatives_lower_than_unmatched: true
			force_match_for_each_row: true
			use_matmul_gather: true
		  }
		}
		similarity_calculator {
		  iou_similarity {
		  }
		}
		box_predictor {
		  weight_shared_convolutional_box_predictor {
			conv_hyperparams {
			  regularizer {
				l2_regularizer {
				  weight: 3.9999998989515007e-05
				}
			  }
			  initializer {
				random_normal_initializer {
				  mean: 0.0
				  stddev: 0.009999999776482582
				}
			  }
			  activation: SWISH
			  batch_norm {
				decay: 0.9900000095367432
				scale: true
				epsilon: 0.0010000000474974513
			  }
			  force_use_bias: true
			}
			depth: 64
			num_layers_before_predictor: 3
			kernel_size: 3
			class_prediction_bias_init: -4.599999904632568
			use_depthwise: true
		  }
		}
		anchor_generator {
		  multiscale_anchor_generator {
			min_level: 3
			max_level: 7
			anchor_scale: 4.0
			aspect_ratios: 1.0
			aspect_ratios: 2.0
			aspect_ratios: 0.5
			scales_per_octave: 3
		  }
		}
		post_processing {
		  batch_non_max_suppression {
			score_threshold: 9.99999993922529e-09
			iou_threshold: 0.5
			max_detections_per_class: 100
			max_total_detections: 100
		  }
		  score_converter: SIGMOID
		}
		normalize_loss_by_num_matches: true
		loss {
		  localization_loss {
			weighted_smooth_l1 {
			}
		  }
		  classification_loss {
			weighted_sigmoid_focal {
			  gamma: 1.5
			  alpha: 0.25
			}
		  }
		  classification_weight: 1.0
		  localization_weight: 1.0
		}
		encode_background_as_zeros: true
		normalize_loc_loss_by_codesize: true
		inplace_batchnorm_update: true
		freeze_batchnorm: false
		add_background_class: false
	  }
	}
	train_config {
	
	  # Batch_Size will depend on available memory, lower than 8 can decrease model accuarcy
	  batch_size: 8
	  
	  data_augmentation_options {
		random_horizontal_flip {
		}
	  }
	  data_augmentation_options {
		random_scale_crop_and_pad_to_square {
		  output_size: 512
		  scale_min: 0.10000000149011612
		  scale_max: 2.0
		}
	  }
	  sync_replicas: true
	  optimizer {
		momentum_optimizer {
		  learning_rate {
			cosine_decay_learning_rate {
			  learning_rate_base: 0.07999999821186066
			  
			  # Set equal to Num_Steps
			  total_steps: 300000
			  
			  warmup_learning_rate: 0.0010000000474974513
			  
			  # Set equal to Num_Steps/100
			  warmup_steps: 3000
			  
			}
		  }
		  momentum_optimizer_value: 0.8999999761581421
		}
		use_moving_average: false
	  }
	  
	  # Insert path to `efficientdet-d0-trainer/` directory
	  fine_tune_checkpoint: "<PATH_TO_EFFDET_DIR>/efficientdet-d0-trainer/trainer/pre-trained-models/efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0.index"
	  
	  # EfficientDet Models are recommended to be trained for 20 Epochs. Num_Steps = (#_of_Images/Batch_Size)*Epochs (eg. Num_Steps = (800/8)*20)
	  num_steps: 300000
	  
	  startup_delay_steps: 0.0
	  replicas_to_aggregate: 8
	  max_number_of_boxes: 100
	  unpad_groundtruth_tensors: false
	  fine_tune_checkpoint_type: "detection"
	  use_bfloat16: false
	  fine_tune_checkpoint_version: V2
	}
	train_input_reader: {
	
	  # Insert path to `efficientdet-d0-trainer/` directory
	  label_map_path: "<PATH_TO_EFFDET_DIR>/efficientdet-d0-trainer/trainer/annotations/label_map.pbtxt"
	  
	  tf_record_input_reader {
	    # Insert path to `efficientdet-d0-trainer/` directory
		input_path: "<PATH_TO_EFFDET_DIR>/efficientdet-d0-trainer/trainer/annotations/train.record"
		
	  }
	  
	  # Helps control memory uses in tandom with batch size
	  queue_capacity: 500
	  min_after_dequeue: 250
	  
	}

	eval_config: {
	  metrics_set: "coco_detection_metrics"
	  use_moving_averages: false
	  
	  # Set Equal to Batch_Size in Train_Config
	  batch_size: 8;
	  
	}

	eval_input_reader: {
	
	  # Insert path to `efficientdet-d0-trainer/` directory
	  label_map_path: "<PATH_TO_EFFDET_DIR>/efficientdet-d0-trainer/trainer/annotations/label_map.txt"
	  
	  shuffle: false
	  num_epochs: 1
	  tf_record_input_reader {
	  
	    # Insert path to `efficientdet-d0-trainer/` directory
	    input_path: "<PATH_TO_EFFDET_DIR>/efficientdet-d0-trainer/trainer/annotations/test.record"
		
	  }
	}
	```
</Details>

### Training Model

#### Begin Training
	
This is to start the training job, a training job may take several hours depending on the number of images to train on
	
- `cd` into the `buff-code/` directory and run
	
		buffpy -b efficientdet-d0-trainer
	
- Warnings will appear, and after some time an information print out as the one below should appear (this can take around a minute to appear)

		INFO:tensorflow:Step 100 per-step time 1.153s loss=0.761
		I0716 05:26:55.879558  1364 model_lib_v2.py:632] Step 100 per-step time 1.153s loss=0.761
		...
	
	
#### Monitor Training

This is only if you would like to use TensorBoard to moniter training progress (not required)

- Open a seperate command terminal from the terminal training the model

- Activate Anaconda virtual enviroment (if you are using one to train the model)

- `cd` into the `trainer/` directory

- Then run 

		tensorboard --logdir=models/my_ssd_effdet_d0

- This command should return a print out with a URL on the last line, the URL should be the same as below but can differ

		http://localhost:6006/
		
#### Warning

If while training a `Loss/Total Loss` of `nan` appears the training model is ruined because the training job ran out of memory. This can be troubleshooted by lowering the batch_size and/or lowering the queue_capacity and min_after_dequeue variables in the `pipeline` file. If possible the easiest solution would be to dedicate more memory to the training job.
	
### Export Trained Model

The training job can be stopped by `ctrl-c` if the job needs to be stopped before the set num_steps (it can take around a minute to actually stop the job)

- To export the trained model `cd` into the `trainer/` directory and run
		
		python3 .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_effdet_d0\pipeline.config --trained_checkpoint_dir .\models\my_ssd_effdet_d0\ --output_directory .\exported-models\my_model
		
- After export the `exported-models/my_model/` directory should contain

		trainer/
		|- exported-models/
		   |- my_model/
		      |- checkpoint/
		      |- saved_model/
		      |- pipeline.config
		
- This model can be used for inferencing

<br>
	
## Manual Installation and Training
	
 ### Install TensorFlow PIP package

  - Install TensorFlow package

		pip install --ignore-installed --upgrade tensorflow==2.5.0
		
  - Verify installation
	
		python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
		
  - The above code should return a print-out ending with (the first number will vary):
 
	`tf.Tensor(-54.834015, shape=(), dtype=float32)`
 
### Install TensorFlow Object Detection API 
 #### Protobuf Installation
 
 - Download the latest realease from (e.g. protoc-all-#.##.#.tar.gz)
 	
		https://github.com/protocolbuffers/protobuf/releases
		
 - Extract contents into the of the `tar.gz` file into a directory `<PATH_TO_PB>` of your choice

 - Add `<PATH_TO_PB>\bin` to your `PATH`

   - Open `~/ .bashrc` file in a text editor
	
	
   - Append export syntax to the end of the file 
	
		 export PATH="<PATH_TO_PB>/bin:$PATH"
		
   - Restart the terminal, `cd` into `models/research/` and run
	
		 protoc object_detection/protos/*.proto --python_out=.
		 
 #### COCO API Installation

 Although `pycocotools` should get installed along with the Object Detection API, this install can fail for various reasons and is simpler to install before hand

 - Download COCOAPI into a directory of your choice

	 	git clone https://github.com/cocodataset/cocoapi.git
	 
 - Make the `pycocotools` directory
	
		cd cocoapi/PythonAPI
		pip install cython
		make
		
 - Copy the `pycocotools` directory into the `models/reseach/` directory
		
		cp -r pycocotools <PATH_TO_MODELS>/models/research/

 #### Install Object Detection API

 - From within `<PATH_TO_MODELS>/models/research/` run

		cp object_detection/packages/tf2/setup.py .
		python -m pip install .
		
 #### Test Installation

 - From within `<PATH_TO_MODELS>/models/research/` run

		python3 object_detection/builders/model_builder_tf2_test.py
		
 - Once the program is finished you should observe a print-out similiar to
 &ensp;&thinsp;<Details>
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

### Preparing for Training Job

 #### Create TensorFlow Records
 
 - Install `pandas` package
 
	- Anaconda

			conda install pandas
	- Pip
	
			pip install pandas
 
 - Using the `generate_tfrecord.py` in `preprocessing/` run
 
 		# Creates train data:
 		python3 generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record
		
		#Creates test data:
		python3 generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record

  - There should now be a `test.record` and `train.record` in  `trainer/annotaions/`

#### Adjusting Training Job

  - `cd` into the `trainer/models/my_ssd_effdet_do/`
  - Open the `pipeline.config` file in a text editor
  - Ensure all directory paths are correct for your machines and all variables are adjusted
&ensp;&thinsp; <Details>
	```python
	model {
	  ssd {
		num_classes: 4
		image_resizer {
		  keep_aspect_ratio_resizer {
			min_dimension: 512
			max_dimension: 512
			pad_to_max_dimension: true
		  }
		}
		feature_extractor {
		  type: "ssd_efficientnet-b0_bifpn_keras"
		  conv_hyperparams {
			regularizer {
			  l2_regularizer {
				weight: 3.9999998989515007e-05
			  }
			}
			initializer {
			  truncated_normal_initializer {
				mean: 0.0
				stddev: 0.029999999329447746
			  }
			}
			activation: SWISH
			batch_norm {
			  decay: 0.9900000095367432
			  scale: true
			  epsilon: 0.0010000000474974513
			}
			force_use_bias: true
		  }
		  bifpn {
			min_level: 3
			max_level: 7
			num_iterations: 3
			num_filters: 64
		  }
		}
		box_coder {
		  faster_rcnn_box_coder {
			y_scale: 1.0
			x_scale: 1.0
			height_scale: 1.0
			width_scale: 1.0
		  }
		}
		matcher {
		  argmax_matcher {
			matched_threshold: 0.5
			unmatched_threshold: 0.5
			ignore_thresholds: false
			negatives_lower_than_unmatched: true
			force_match_for_each_row: true
			use_matmul_gather: true
		  }
		}
		similarity_calculator {
		  iou_similarity {
		  }
		}
		box_predictor {
		  weight_shared_convolutional_box_predictor {
			conv_hyperparams {
			  regularizer {
				l2_regularizer {
				  weight: 3.9999998989515007e-05
				}
			  }
			  initializer {
				random_normal_initializer {
				  mean: 0.0
				  stddev: 0.009999999776482582
				}
			  }
			  activation: SWISH
			  batch_norm {
				decay: 0.9900000095367432
				scale: true
				epsilon: 0.0010000000474974513
			  }
			  force_use_bias: true
			}
			depth: 64
			num_layers_before_predictor: 3
			kernel_size: 3
			class_prediction_bias_init: -4.599999904632568
			use_depthwise: true
		  }
		}
		anchor_generator {
		  multiscale_anchor_generator {
			min_level: 3
			max_level: 7
			anchor_scale: 4.0
			aspect_ratios: 1.0
			aspect_ratios: 2.0
			aspect_ratios: 0.5
			scales_per_octave: 3
		  }
		}
		post_processing {
		  batch_non_max_suppression {
			score_threshold: 9.99999993922529e-09
			iou_threshold: 0.5
			max_detections_per_class: 100
			max_total_detections: 100
		  }
		  score_converter: SIGMOID
		}
		normalize_loss_by_num_matches: true
		loss {
		  localization_loss {
			weighted_smooth_l1 {
			}
		  }
		  classification_loss {
			weighted_sigmoid_focal {
			  gamma: 1.5
			  alpha: 0.25
			}
		  }
		  classification_weight: 1.0
		  localization_weight: 1.0
		}
		encode_background_as_zeros: true
		normalize_loc_loss_by_codesize: true
		inplace_batchnorm_update: true
		freeze_batchnorm: false
		add_background_class: false
	  }
	}
	train_config {
	
	  # Batch_Size will depend on available memory, lower than 8 can decrease model accuarcy
	  batch_size: 8
	  
	  data_augmentation_options {
		random_horizontal_flip {
		}
	  }
	  data_augmentation_options {
		random_scale_crop_and_pad_to_square {
		  output_size: 512
		  scale_min: 0.10000000149011612
		  scale_max: 2.0
		}
	  }
	  sync_replicas: true
	  optimizer {
		momentum_optimizer {
		  learning_rate {
			cosine_decay_learning_rate {
			  learning_rate_base: 0.07999999821186066
			  
			  # Set equal to Num_Steps
			  total_steps: 300000
			  
			  warmup_learning_rate: 0.0010000000474974513
			  
			  # Set equal to Num_Steps/100
			  warmup_steps: 3000
			  
			}
		  }
		  momentum_optimizer_value: 0.8999999761581421
		}
		use_moving_average: false
	  }
	  
	  # Insert path to `efficientdet-d0-trainer/` directory
	  fine_tune_checkpoint: "<PATH_TO_EFFDET_DIR>/efficientdet-d0-trainer/trainer/pre-trained-models/efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0.index"
	  
	  # EfficientDet Models are recommended to be trained for 20 Epochs. Num_Steps = (#_of_Images/Batch_Size)*Epochs (eg. Num_Steps = (800/8)*20)
	  num_steps: 300000
	  
	  startup_delay_steps: 0.0
	  replicas_to_aggregate: 8
	  max_number_of_boxes: 100
	  unpad_groundtruth_tensors: false
	  fine_tune_checkpoint_type: "detection"
	  use_bfloat16: false
	  fine_tune_checkpoint_version: V2
	}
	train_input_reader: {
	
	  # Insert path to `efficientdet-d0-trainer/` directory
	  label_map_path: "<PATH_TO_EFFDET_DIR>/efficientdet-d0-trainer/trainer/annotations/label_map.pbtxt"
	  
	  tf_record_input_reader {
	    # Insert path to `efficientdet-d0-trainer/` directory
		input_path: "<PATH_TO_EFFDET_DIR>/efficientdet-d0-trainer/trainer/annotations/train.record"
		
	  }
	  
	  # Helps control memory uses in tandom with batch size
	  queue_capacity: 500
	  min_after_dequeue: 250
	  
	}

	eval_config: {
	  metrics_set: "coco_detection_metrics"
	  use_moving_averages: false
	  
	  # Set Equal to Batch_Size in Train_Config
	  batch_size: 8;
	  
	}

	eval_input_reader: {
	
	  # Insert path to `efficientdet-d0-trainer/` directory
	  label_map_path: "<PATH_TO_EFFDET_DIR>/efficientdet-d0-trainer/trainer/annotations/label_map.txt"
	  
	  shuffle: false
	  num_epochs: 1
	  tf_record_input_reader {
	  
	    # Insert path to `efficientdet-d0-trainer/` directory
	    input_path: "<PATH_TO_EFFDET_DIR>/efficientdet-d0-trainer/trainer/annotations/test.record"
		
	  }
	}
	```
</Details>

### Training Model

#### Begin Training
This is to start the training job, a training job may take several hours depending on the number of images to train on

- `cd` into the `trainer/` directory and run 

		python3 model_main_tf2.py --model_dir=models/my_ssd_effdet_d0 --pipeline_config_path=models/my_ssd_effdet_d0/pipeline.config
		
- Warnings will appear, and after some time an information print out as the one below should appear (this may take a handful of minutes to appear)

		INFO:tensorflow:Step 100 per-step time 1.153s loss=0.761
		I0716 05:26:55.879558  1364 model_lib_v2.py:632] Step 100 per-step time 1.153s loss=0.761
		...
		
#### Monitor Training

This is only if you would like to use TensorBoard to moniter training progress (not required)

- Open a seperate command terminal from the terminal training the model

- Activate Anaconda virtual enviroment (if you are using one to train the model)

- `cd` into the `trainer/` directory

- Then run 

		tensorboard --logdir=models/my_ssd_effdet_d0

- This command should return a print out with a URL on the last line, the URL should be the same as below but can differ

		http://localhost:6006/
		
#### Warning

If while training a `Loss/Total Loss` of `nan` appears the training model is ruined because the training job ran out of memory. This can be troubleshooted by lowering the batch_size and/or lowering the queue_capacity and min_after_dequeue variables in the `pipeline` file. If possible the easiest solution would be to dedicate more memory to the training job.
		
### Export Trained Model

The training job can be stopped by `ctrl-c` if the job needs to be stopped before the set num_steps (it can take around a minute to actually stop the job)

- To export the trained model `cd` into the `trainer/` directory and run
		
		python3 .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_effdet_d0\pipeline.config --trained_checkpoint_dir .\models\my_ssd_effdet_d0\ --output_directory .\exported-models\my_model
		
- After export the `exported-models/my_model/` directory should contain

		trainer/
		|- exported-models/
		   |- my_model/
		      |- checkpoint/
		      |- saved_model/
		      |- pipeline.config
		
- This model can be used for inferencing


 
