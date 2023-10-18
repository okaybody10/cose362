import torch


def quadrant(class_index, predict, truth):
    c = class_index
    true_p = torch.where((c == truth) & (c == predict), 1, 0).sum()  # true positive for c
    false_p = torch.where((c == predict) & (c != truth), 1, 0).sum()  # false positive for c
    false_n = torch.where((c == truth) & (c != predict), 1, 0).sum()  # false negative for c
    true_n = torch.where((c != truth) & (c != predict), 1, 0).sum()  # true negative for c
    return {"true_p": true_p, "false_p": false_p, "false_n": false_n, "true_n": true_n}

def precision_micro(num_classes, predict, truth):
    true_p_of_all_c = 0
    false_p_of_all_c = 0
    for c in range(num_classes):
        quad = quadrant(class_index=c, predict=predict, truth=truth)
        true_p_of_all_c += quad["true_p"]
        false_p_of_all_c += quad["false_p"]
    return true_p_of_all_c / (true_p_of_all_c + false_p_of_all_c)


def precision_macro(num_classes, predict, truth):
    sum_of_precision_of_class_c = 0
    for c in range(num_classes):
        quad = quadrant(class_index=c, predict=predict, truth=truth)
        precision_of_class_c = quad["true_p"] / (quad["true_p"] + quad["false_p"])
        sum_of_precision_of_class_c += precision_of_class_c if (
                    precision_of_class_c == precision_of_class_c) else 0  # NaN Check
    return sum_of_precision_of_class_c / num_classes


def recall_micro(num_classes, predict, truth):
    true_p_of_all_c = 0
    false_n_of_all_c = 0
    for c in range(num_classes):
        quad = quadrant(class_index=c, predict=predict, truth=truth)
        true_p_of_all_c += quad["true_p"]
        false_n_of_all_c += quad["false_n"]
    return true_p_of_all_c / (true_p_of_all_c + false_n_of_all_c)


def recall_macro(num_classes, predict, truth):
    sum_of_recall_of_class_c = 0
    for c in range(num_classes):
        quad = quadrant(class_index=c, predict=predict, truth=truth)
        recall_of_class_c = quad["true_p"] / (quad["true_p"] + quad["false_n"])
        sum_of_recall_of_class_c += recall_of_class_c if (recall_of_class_c == recall_of_class_c) else 0  # NaN check
    return sum_of_recall_of_class_c / num_classes


# it gets "one-hot vector" version of "truth" and "predict"
def f1score(num_classes, predict, truth, precision, recall):
    predict_index = torch.tensor([torch.argmax(predict[i]) for i in range(predict.size()[0])])
    truth_index = torch.tensor([torch.argmax(truth[i]) for i in range(truth.size()[0])])
    print("predict_index", predict_index)
    pval = precision(num_classes=num_classes, predict=predict_index, truth=truth_index)
    rval = recall(num_classes=num_classes, predict=predict_index, truth=truth_index)
    return 2 * (pval * rval) / (pval + rval)
