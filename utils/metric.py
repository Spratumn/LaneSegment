import numpy as np


def update_confusion_matrix(prediction, label, confus_matrix):
    """
    prediction : [N, H, W]
    label: [N, H, W]
    """
    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()
    for i in range(8):
        # label: Positive or Negative
        label_p = label == i
        label_n = label != i

        # prediction: Positive or Negative
        pred_p = prediction == i
        pred_n = prediction != i

        # label == Positive and prediction == Positive
        tp = np.sum(label_p * pred_p)
        # label == Negative and prediction == Negative 
        tn = np.sum(label_n * pred_n)
        # label == Negative and prediction == Positive 
        fp = np.sum(label_n * pred_p)
        # label == Positive and prediction == Negative 
        fn = np.sum(label_p * pred_n)
        
        confus_matrix["TP"][i] += int(tp)
        confus_matrix["TN"][i] += int(tn)
        confus_matrix["FP"][i] += int(fp)
        confus_matrix["FN"][i] += int(fn)

    return confus_matrix


def compute_iou(confusion_matrix):
    class_num = len(confusion_matrix["TP"].keys())
    ious = [0.0]*class_num

    for i in range(class_num):
        tp = confusion_matrix["TP"][i]
        fp = confusion_matrix["FP"][i]
        fn = confusion_matrix["FN"][i]
        ious[i] = tp/(tp+fp+fn) if tp+fp+fn != 0 else 0
    return ious


def compute_precision(confusion_matrix):
    class_num = len(confusion_matrix["TP"].keys())
    precisions = [0.0]*class_num

    for i in range(class_num):
        tp = confusion_matrix["TP"][i]
        fp = confusion_matrix["FP"][i]

        precisions[i] = tp/(tp+fp) if tp+fp != 0 else 0
    return precisions


def compute_recall(confusion_matrix):
    class_num = len(confusion_matrix["TP"].keys())
    recalls = [0.0]*class_num

    for i in range(class_num):
        tp = confusion_matrix["TP"][i]
        fn = confusion_matrix["FN"][i]
        recalls[i] = tp/(tp+fn) if tp+fn != 0 else 0
    return recalls


def compute_F1_score(precisions, recalls):
    class_num = len(precisions)
    f1_scores = [0.0]*class_num

    for i in range(class_num):
        f1_scores[i] = 2*precisions[i]*recalls[i]/(precisions[i]+recalls[i])
    return f1_scores


def compute_mean(score_list):
    return np.mean(score_list[1:])
