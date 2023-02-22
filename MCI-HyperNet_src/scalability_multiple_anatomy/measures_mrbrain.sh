MODEL='<modelname>' 
BASE_PATH='<base path>'
DATASET_TYPE='mrbrain_t1320x320'
MASK_TYPE='gaussian'
ACC_FACTOR='7x'
TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}

PREDICTIONS_PATH='/<path to store the model file>/'${MODEL}'/results/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'
REPORT_PATH='/<path to store the model file>/'${MODEL}'/results/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'

echo python measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTOR} --mask-type ${MASK_TYPE} --dataset-type ${DATASET_TYPE}
 
python measures_csv.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTOR} --mask-type ${MASK_TYPE} --dataset-type ${DATASET_TYPE}
 



