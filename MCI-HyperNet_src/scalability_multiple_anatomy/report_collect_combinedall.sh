MODEL='<modelname>' 
BASE_PATH='<base path>'
echo ${MODEL}

for DATASET_TYPE in 'mrbrain_t1320x320' 'kneeMRI320x320' 
    do
    for MASK_TYPE in 'cartesian' 'gaussian'
        do 
        for ACC_FACTOR in '4x' '5x' '8x'
            do 
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR} 
            REPORT_PATH='/<path to store the model file>/'${MODEL}'/report_'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'.txt'
            echo ${REPORT_PATH}
            cat ${REPORT_PATH}
            echo "\n"
            done 
        done 
    done 

