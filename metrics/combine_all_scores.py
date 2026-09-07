# from f1_score_f1_pa import *
# from fc_score import *
# from precision_at_k import *
# from customizable_f1_score import *
# from AUC import *
# from Matthews_correlation_coefficient import *
# from affiliation.generics import convert_vector_to_events
# from affiliation.metrics import pr_from_events
# from vus.models.feature import Window
# from vus.metrics import get_range_vus_roc
from metrics.f1_score_f1_pa import *
from metrics.fc_score import *
from metrics.precision_at_k import *
from metrics.customizable_f1_score import *
from metrics.AUC import *
from metrics.Matthews_correlation_coefficient import *
from metrics.affiliation.generics import convert_vector_to_events
from metrics.affiliation.metrics import pr_from_events
# from vus.models.feature import Window
from metrics.vus.metrics import get_range_vus_roc


def _to_float(value):
    return float(value)


def _safe_set(score_list, warnings, key, fn):
    try:
        score_list[key] = _to_float(fn())
    except Exception as exc:
        warnings[key] = str(exc)


def combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores):
    """Return every metric that can be computed.

    The call signature follows the legacy evaluation scripts:
        y_test: binary prediction sequence
        pred_labels: ground-truth label sequence
        anomaly_scores: continuous anomaly scores, used by precision@k

    Range-AUC/VUS intentionally follow the legacy script convention, where
    the binary prediction sequence is passed as the score input.
    """
    score_list = {}
    warnings = {}

    true_events = None
    _safe_set(score_list, warnings, "f1_score_ori", lambda: get_accuracy_precision_recall_fscore(y_test, pred_labels)[3])
    _safe_set(score_list, warnings, "f05_score_ori", lambda: get_accuracy_precision_recall_fscore(y_test, pred_labels)[4])

    try:
        true_events = get_events(y_test)
    except Exception as exc:
        warnings["true_events"] = str(exc)

    if true_events is not None:
        _safe_set(score_list, warnings, "f1_score_pa", lambda: get_point_adjust_scores(y_test, pred_labels, true_events)[5])
        _safe_set(
            score_list,
            warnings,
            "f1_score_c",
            lambda: get_composite_fscore_raw(y_test, pred_labels, true_events, return_prec_rec=True)[2],
        )

    def _pa_scores():
        pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(y_test, pred_labels)
        return pa_accuracy, pa_precision, pa_recall, pa_f_score

    try:
        pa_accuracy, pa_precision, pa_recall, pa_f_score = _pa_scores()
        score_list["pa_accuracy"] = _to_float(pa_accuracy)
        score_list["pa_precision"] = _to_float(pa_precision)
        score_list["pa_recall"] = _to_float(pa_recall)
        score_list["pa_f_score"] = _to_float(pa_f_score)
    except Exception as exc:
        warnings["point_adjust_scores"] = str(exc)

    _safe_set(score_list, warnings, "range_f_score", lambda: customizable_f1_score(y_test, pred_labels))
    _safe_set(score_list, warnings, "precision_k", lambda: precision_at_k(y_test, anomaly_scores, pred_labels))
    _safe_set(score_list, warnings, "point_auc", lambda: point_wise_AUC(pred_labels, y_test))
    _safe_set(score_list, warnings, "range_auc", lambda: Range_AUC(pred_labels, y_test))
    _safe_set(score_list, warnings, "MCC_score", lambda: MCC(y_test, pred_labels))

    try:
        events_pred = convert_vector_to_events(y_test)
        events_gt = convert_vector_to_events(pred_labels)
        affiliation = pr_from_events(events_pred, events_gt, (0, len(y_test)))
        aff_precision = float(affiliation["precision"])
        aff_recall = float(affiliation["recall"])
        score_list["Affiliation precision"] = aff_precision
        score_list["Affiliation recall"] = aff_recall
        denom = aff_precision + aff_recall
        score_list["Affiliation F1"] = 0.0 if denom == 0 else 2 * aff_recall * aff_precision / denom
    except Exception as exc:
        warnings["Affiliation"] = str(exc)

    try:
        results = get_range_vus_roc(y_test, pred_labels, 100)  # slidingWindow = 100 default
        score_list["R_AUC_ROC"] = _to_float(results["R_AUC_ROC"])
        score_list["R_AUC_PR"] = _to_float(results["R_AUC_PR"])
        score_list["VUS_ROC"] = _to_float(results["VUS_ROC"])
        score_list["VUS_PR"] = _to_float(results["VUS_PR"])
    except Exception as exc:
        warnings["RangeAUC/VUS"] = str(exc)

    if warnings:
        score_list["_warnings"] = warnings

    return score_list


def main():
    y_test = np.zeros(100)
    y_test[10:20] = 1
    y_test[50:60] = 1
    pred_labels = np.zeros(100)
    pred_labels[15:17] = 1
    pred_labels[55:62] = 1
    anomaly_scores = np.zeros(100)
    anomaly_scores[15:17] = 0.7
    anomaly_scores[55:62] = 0.6
    pred_labels[51:55] = 1
    true_events = get_events(y_test)
    scores = combine_all_evaluation_scores(y_test, pred_labels, anomaly_scores)
    # scores = test(y_test, pred_labels)
    for key,value in scores.items():
        print(key,' : ',value)

    
if __name__ == "__main__":
    main()
