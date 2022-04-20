from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, \
                     precision_score, recall_score, confusion_matrix, \
                     roc_auc_score, roc_curve
from sklearn.preprocessing import Binarizer
import matplotlib.pyplot as plt

def get_clf_eval(y_test, pred, pred_proba=None) :
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    # F1 스코어 추가
    f1 = f1_score(y_test, pred)
    #AUC 추가
    AUC = roc_auc_score(y_test, pred)
    lr_probs = pred_proba[:,1]
    ns_probs = [0 for _ in range(len(y_test))]
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    plt.plot(ns_fpr, ns_tpr, linestyle = "--", label="No Skill")
    plt.plot(lr_fpr, lr_tpr, marker = ".", label="XGBOOST")
    plt.xlabel("False Positive Rate")
    plt.ylabel("Ture Positive Rate")
    plt.legend()
    plt.show()
    print("Confusion Matrix")
    print(confusion)
    # f1 score print 추가
    print(f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, Recall : {recall:.4f}, F1 :{f1:.4f}, AUC : {AUC:.4f}")


def get_eval_by_threshold(y_test, pred_proba_c1, thresholds) :
    # thresholds list 객체 내의 값을 차례로 iteration 하면서 Evaluation 수행
    for custom_threshold in thresholds :
        binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print(f"임곗값 : {custom_threshold}")
        get_clf_eval(y_test, custom_predict)


def precision_recall_curve_plot(y_test, pred_proba_c1) :
    # threshold ndarray와 이 threshold에 따른 정밀도, 재현율 ndarray 추출
    precisions, recalls, thresholds = precision_recall_curve(y_test, pred_proba_c1)

    # X축을 threshold값으로, Y축은 정밀도, 재현율 값으로 각각 plot 수행.
    # 정밀도는 점선으로 표시
    plt.figure(figsize=(8,6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(thresholds, precisions[0:threshold_boundary], linestyle="--", label="precision")
    plt.plot(thresholds, recalls[0:threshold_boundary], label="recall")

    # threshold값 X 축의 Scale을 0.1 단위로 변경
    start, end = plt.xlim()
    # thresholds의 값을 0.10394, 0.2~ 이런식으로 배열하고 소숫점 2자리 반올림
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    # x축, y축 label과 legend, 그리고 grid 설정
    plt.xlabel("Threshold value"); plt.ylabel("Precision and Recall value")
    plt.legend(); plt.grid()
    plt.show()