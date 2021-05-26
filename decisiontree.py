from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

train_dtr_acc, valid_dtr_acc = 0


def classificazione(y_train, y_valid, contatore_generi, costruzione_modello):
    dtr = DecisionTreeClassifier()
    model_dtr, train_dtr, valid_dtr = costruzione_modello(dtr)
    print("REPORT CLASSIFICAZIONE : DECISION TREE")
    print("Training:\n", classification_report(y_true=y_train, y_pred=train_dtr, target_names=contatore_generi))
    print("Validation:\n", classification_report(y_true=y_valid, y_pred=valid_dtr, target_names=contatore_generi))

    print("Accuracy")
    global train_dtr_acc
    global valid_dtr_acc
    train_dtr_acc = accuracy_score(y_true=y_train, y_pred=train_dtr)
    valid_dtr_acc = accuracy_score(y_true=y_valid, y_pred=valid_dtr)
    print("Training: ", train_dtr_acc)
    print("Validation: ", valid_dtr_acc)
    print("\n\n\n")


def getAccTrain():
    return train_dtr_acc


def getAccValid():
    return valid_dtr_acc
