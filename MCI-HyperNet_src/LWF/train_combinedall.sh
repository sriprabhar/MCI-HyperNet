MODEL='<modelname>' 
BASE_PATH='<base path>'
# this training aims to combines three contrasts T1, FLAIR and IR from MrbrainS dataset 
DATASET_TYPE='mrbrain_t1','mrbrain_flair','mrbrain_ir' # list of all contrast type or anatomy type to be considered
CURRENT_DATASET_TYPE='mrbrain_ir' # list of new contrast to be learned. In this case, new is IR while the old contexts are T1 and FLAIR
# the model needs to learn IR without forgetting T1 and FLAIR which is learned previously
MASK_TYPE='cartesian','gaussian' # list of mask types that the model has to learn
ACC_FACTORS='4x','5x','8x' # list of acceleration factors that the model needs to learn
BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
EXP_DIR='/<path to store the model file>/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/us_masks/' #path to undersampling masks

echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --current_dataset_type ${CURRENT_DATASET_TYPE}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --current_dataset_type ${CURRENT_DATASET_TYPE}
