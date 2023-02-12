#!/bin/bash

cd ${PROJECT_ROOT}
cd buff-code/src/efficientdet-d0-trainer/trainer/images/
imagesDir=$(pwd)

cd ${PROJECT_ROOT}
cd buff-code/src/efficientdet-d0-trainer/trainer/annotations/
annoDir=$(pwd)

cd ${PROJECT_ROOT}
cd buff-code/src/efficientdet-d0-trainer/preprocessing/



# Creates train.record
python generate_tfrecord.py -x ${imagesDir}/train -l ${annoDir}/label_map.pbtxt -o ${imagesDir}/train.record

# Creates test.record
python generate_tfrecord.py -x ${imagesDir}/test -l ${annoDir}/label_map.pbtxt -o ${imagesDir}/test.record

cd ${PROJECT_ROOT}
cd buff-code/src/efficientdet-d0-trainer/trainer/

#Begins training 
python model_main_tf2.py --model_dir=models/my_ssd_effdet_d0 --pipeline_config_path=models/my_ssd_effdet_d0/pipeline.config
echo -e "\n"
echo "training should begin, after a little while and some warning messages a information print out with"
echo "per-step time and loss statistics should appear, to monitor training using TensorBoard go to the"
echo "github https://github.com/ethan-wst/efficientdet-d0-trainer.git and follow the instrucitons in the"
echo "\'Monito Training\' section"
echo -e "\n"


