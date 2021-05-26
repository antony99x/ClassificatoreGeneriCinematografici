from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

train_rfc_acc, valid_rfc_acc = 0


def classificazione(y_train, y_valid, contatore_generi, costruzione_modello):
    rfc = RandomForestClassifier()
    model_rfc, train_rfc, valid_rfc = costruzione_modello(rfc)
    print("REPORT CLASSIFICAZIONE : RANDOM FOREST")
    print("Training:\n", classification_report(y_true=y_train, y_pred=train_rfc, target_names=contatore_generi))
    print("Validation:\n", classification_report(y_true=y_valid, y_pred=valid_rfc, target_names=contatore_generi))

    print("Accuracy")
    global train_rfc_acc
    global valid_rfc_acc
    train_rfc_acc = accuracy_score(y_true=y_train, y_pred=train_rfc)
    valid_rfc_acc = accuracy_score(y_true=y_valid, y_pred=valid_rfc)
    print("Training: ", train_rfc_acc)
    print("Validation: ", valid_rfc_acc)
    print("\n\n\n")


def getAccTrain():
    return train_rfc_acc


def getAccValid():
    return valid_rfc_acc
