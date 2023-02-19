#!/bin/bash

EFFDET_SRC="${PROJECT_ROOT}/src/efficientdet-d0-trainer"
IMAGE_DIR="${EFFDET_SRC}/trainer/images"
ANNO_DIR="${EFFDET_SRC}/trainer/annotations"

# Creates .record files
cd "${EFFDET_SRC}/preprocessing" && \
  python3 generate_tfrecord.py -x ${IMAGE_DIR}/train -l ${ANNO_DIR}/label_map.pbtxt -o ${ANNO_DIR}/train.record && \
  python3 generate_tfrecord.py -x ${IMAGE_DIR}/test -l ${ANNO_DIR}/label_map.pbtxt -o ${ANNO_DIR}/test.record
  
# Begins model training 
cd "${EFFDET_SRC}/trainer" && \
  python3 model_main_tf2.py --model_dir=models/my_ssd_effdet_d0 --pipeline_config_path=models/my_ssd_effdet_d0/pipeline.config
  
echo -e "\n"
echo "training should begin, after a little while and some warning messages a information print out with"
echo "per-step time and loss statistics should appear, to monitor training using TensorBoard go to the"
echo "github https://github.com/ethan-wst/efficientdet-d0-trainer.git and follow the instrucitons in the"
echo "Monitor Training section"
echo -e "\n"
