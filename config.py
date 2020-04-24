class Config:
    # data set config
    CSV_DIR = "data_list"
    LOG_DIR = "logs"
    IMAGE_SHAPE = (3, 1710, 3384)
    CROP_SIZE = 699
    RESIZE_SCALE = 3  # train_img_size->(3,337,1128)
    
    # model config
    CLASS_NUM = 8
    MODEL = 'deeplabv3+'  # 'unet' or 'deeplabv3+'
    NORMAL = 'gn'
    NORMAL_GROUP_NUM = 8

    # train config
    MULTI_GPU = False
    DEVICE_LIST = [0]
    TRAIN_NUMBER = 10  # 1,2,3,4,5,6,7,8,9,10
    EPOCHS = 5  # total->50
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 4
    WEIGHT_DECAY = 1.0e-4
    LR_LIST = [2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6, 5e-7, 2e-7]
    BASE_LR = LR_LIST[TRAIN_NUMBER-1] if TRAIN_NUMBER <= 10 else 1e-7
    PRE_TRAINED = False
    PRE_TRAIN_WEIGHTS = 'laneNet_{0}_{1}th.pt'.format(MODEL, TRAIN_NUMBER-1)

    # inference config
    INFER_MODE = 'eval'  # eval or test
    # in 'eval' mode: 
    #                input: the csv file which contain both src_img_path and label_path
    #                         csv file should put in 'data_list'
    #                 output:
    #                         'inference_result.txt': contain the metric and src_img_path
    #                         '***_bin.png': predict_label
    # in 'test' mode:
    #                 input: src_img_path
    #                 output:
    #                         'src_img_path.csv'
    #                         'predict_label.csv'
    #                         '***_bin.png': predict_label

    SRC_IMG = ''
    # in 'eval' mode: SRC_IMG = '****.csv'
    # in 'test' mode: SRC_IMG = '/***/***'
    FINAL_WEIGHTS = ''
    LABEL_STYLE = 'color'  # 'color' or 'gray' or 'both'
    OUTPUT_DIR = 'output'

    IMAGE_DIR = "output/2020-01-25 11:47:08"
    SRC_CSV_DIR = "data_list/test_mode_2020-01-25 11:47:08.csv"

