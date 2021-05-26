from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

train_mlp_acc, valid_mlp_acc = 0


def classificazione(y_train, y_valid, contatore_generi, costruzione_modello):
    mlp = MLPClassifier(verbose=True, max_iter=100, hidden_layer_sizes=100)
    model_mlp, train_mlp, valid_mlp = costruzione_modello(mlp, cv=10)
    print("REPORT CLASSIFICAZIONE : MLP")
    print("Training:\n", classification_report(y_true=y_train, y_pred=train_mlp, target_names=contatore_generi))
    print("Validation:\n", classification_report(y_true=y_valid, y_pred=valid_mlp, target_names=contatore_generi))

    print("Accuracy")
    global train_mlp_acc
    global valid_mlp_acc
    train_mlp_acc = accuracy_score(y_true=y_train, y_pred=train_mlp)
    valid_mlp_acc = accuracy_score(y_true=y_valid, y_pred=valid_mlp)
    print("Training: ", train_mlp_acc)
    print("Validation: ", valid_mlp_acc)
    print("\n\n\n")


def getAccTrain():
    return train_mlp_acc


def getAccValid():
    return valid_mlp_acc
