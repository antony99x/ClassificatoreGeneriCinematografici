from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score

train_etc_acc, valid_etc_acc = 0


def classificazione(y_train, y_valid, contatore_generi, costruzione_modello):
    etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
    model_etc, train_etc, valid_etc = costruzione_modello(etc)
    print("REPORT CLASSIFICAZIONE : EXTRA TREES")
    print("Training:\n", classification_report(y_true=y_train, y_pred=train_etc, target_names=contatore_generi))
    print("Validation:\n", classification_report(y_true=y_valid, y_pred=valid_etc, target_names=contatore_generi))

    print("Accuracy")
    global train_etc_acc
    global valid_etc_acc
    train_etc_acc = accuracy_score(y_true=y_train, y_pred=train_etc)
    valid_etc_acc = accuracy_score(y_true=y_valid, y_pred=valid_etc)
    print("Training: ", train_etc_acc)
    print("Validation: ", valid_etc_acc)
    print("\n\n\n")


def getAccTrain():
    return train_etc_acc


def getAccValid():
    return valid_etc_acc
