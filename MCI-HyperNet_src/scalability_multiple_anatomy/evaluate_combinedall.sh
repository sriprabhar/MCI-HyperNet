MODEL='<modelname>' 
BASE_PATH='<base path>'
for DATASET_TYPE in 'mrbrain_t1320x320' 'kneeMRI320x320' 
    do
    for MASK_TYPE in 'cartesian' 'gaussian' 
        do 
        for ACC_FACTOR in '4x' '5x' '8x'
            do 
            echo ${MASK_TYPE}','${ACC_FACTOR} 
            TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
            PREDICTIONS_PATH='/<path to store the model file>/'${MODEL}'/results/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'
            REPORT_PATH='/<path to store the model file>/'${MODEL}'/'
            python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTOR} --mask-type ${MASK_TYPE} --dataset-type ${DATASET_TYPE}
            done 
        done 
    done
