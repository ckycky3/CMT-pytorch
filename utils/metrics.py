def cal_metrics(out, target, metrics, mode, name=None, **kwargs):
    if not metrics[0]:
        return {}

    result = {}
    for metric in metrics:
        if name == "pitch" and metric == "accuracy":
            metric = "accuracy_pitch"
        func = _select_func(metric)
        if name is None:
            key = metric + "/" + mode
        else:
            if metric == "accuracy_pitch":
                metric = "accuracy"
            key = metric + "_" + name + "/" + mode
        result[key] = func(out, target, **kwargs)

    return result


def _select_func(metric_name):
    if metric_name == "accuracy":
        func = accuracy
    elif metric_name == "accuracy_pitch":
        func = accuracy_pitch
    elif metric_name == "confusion_matrix":
        func = confusion_matrix
    elif metric_name == "overall_accuracy":
        func = overall_accuracy
    else:
        raise RuntimeError("%s is not implemented yet" % metric_name)

    return func


# for classification
def accuracy(out, target):
    predicted = out.argmax(dim=1)

    return (predicted == target).float().mean().item()


def accuracy_pitch(out, target, ignore_index=88):
    predicted = out.argmax(dim=1)

    same = (predicted == target)
    # non_ignore = (target != ignore_index)
    non_ignore = (target != 48) * (target != 49)

    return (same * non_ignore).float().sum().item() / non_ignore.float().sum().item()

def confusion_matrix(out, target):
    predicted = out.argmax(dim=1)
    same = (predicted == target).float()
    diff = (predicted != target).float()

    TP = (predicted.float() * same).sum()
    TN = ((1 - predicted.float()) * same).sum()
    FP = (predicted.float() * diff).sum()
    FN = ((1 - predicted.float()) * diff).sum()

    return TP, TN, FP, FN


def overall_accuracy(outs, labels):
    out_pitch, out_voice = outs
    target_pitch, target_voice = labels

    predicted_pitch = out_pitch.argmax(dim=1)
    predicted_voice = out_voice.argmax(dim=1)

    correct_pitch = (predicted_pitch == target_pitch).float()
    correct_voice = (predicted_voice == target_voice).float()

    return (correct_pitch * correct_voice).mean().item()
