MODEL='<modelname>' 
BASE_PATH='<base path>'
mODEL='macj-dc-unetpyr_multiple_anatomy_ontheflymask_onlymaskaspartofcontext_mulp_dataset_type_withgaussian' 
# this training aims to combines brain T1, and coronal pd knee from Hammernik dataset 
DATASET_TYPE='mrbrain_t1320x320','kneeMRI320x320'
MASK_TYPE='cartesian','gaussian'
ACC_FACTORS='4x','5x','8x'
BATCH_SIZE=4
NUM_EPOCHS=150
DEVICE='cuda:0'
EXP_DIR='/<path to store the model file>/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/us_masks/' #path to undersampling masks
echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE}
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE}
