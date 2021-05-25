import pandas as pd
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

# Importo il dataset contenente tutti i dati dei film
dataframeOne = pd.read_csv('movies_metadata.csv', low_memory=False)
# print(dataframeOne.head()) per visualizzare i primi 5 film del dataset

# Creo un nuovo dataframe contenente solo le colonne che mi servono
dataframe = dataframeOne[['title', 'tagline', 'original_title', 'overview', 'genres']]

# Setto il titolo come indice
dataframe.set_index('title', inplace=True)

# Rilevo i dati Not Available dal dataset (verifico che esiste almeno un film senza overview)
# print(dataframe.isna().sum())  # (ci sono film con overview e tagline N.A.)

# Elimino dal dataset i film senza overview (panoramica), dato che userò questa per predire i generi
dataframe.dropna(subset=['overview'], inplace=True)
# print(dataframe)

# Estrazione del genere di ogni film presente nella lista dei dizionari sotto la chiave nome
dataframe['genres'] = dataframe['genres'].apply(literal_eval).apply(
    lambda gen: [] if not isinstance(gen, list) else [i['name'] for i in gen])

# Escludo ogni film in cui il genere è assente o è []
# Seleziono solo quelle righe (film) che hanno un genere
genre_present = dataframe['genres'] != '[]'  # Verifica che per ogni film il genere esiste (true)

# Serie dei generi presenti in movies_metadata
genres = dataframe['genres'][genre_present]
# print(genres)

# Separo e seleziono i generi
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(genres)
label_classes = mlb.classes_
# print(label_classes)

label_data = pd.DataFrame(labels, columns=label_classes)
val = {}
for x in label_classes:
    val.update({x: label_data[x].value_counts()[1]})

# Ordino i generi dei film in base al loro numero di istanze in ordine decrescente
sorted_val = sorted(val.items(), key=lambda kv: kv[1], reverse=True)
val_pd = pd.DataFrame.from_dict(sorted_val, orient='columns')
val_pd.rename(columns={0: "Genre", 1: "Count"}, inplace=True)
# print(val_pd)

# In val_pd ci sono generi che sono case cinematografiche o canali tv. Per eliminarli, seleziono soltanto i primi
# 20 generi del dataset utilizzato per la predizione
dummy_counts = sorted(val.items(), key=lambda kv: kv[1], reverse=True)[0:20]
# print(dummy_counts)
genre_counts = [i[0] for i in dummy_counts]  # Visualizzo i soli generi
# print(genre_counts)

# genre_counts è l'elenco finale dei generi che verranno utilizzati per l' addestramento del modello
final_genres = MultiLabelBinarizer(classes=genre_counts)
top = final_genres.fit(genres)
# Separo la variabile dipendente
y = final_genres.transform(genres)  # I generi esclusi in genre_counts verranno ignorati durante l'implementazione
# del MultiLabelBinarizer
# print(final_genres.classes_)

# Separo la variabile indipendente. La feature 'overview' verrà utilizzata per predire i generi dei film
X = dataframe['overview']
# Se nella lista dei film è presente l'overview ma non il genere, ciò non allena correttamente il nostro modello
# predittivo. Pertanto includerò solo le righe che contengono i generi effettivi e non "[]". Uno dei modi più
# semplici per eseguire questa azione è controllare la somma di ogni riga nei generi dopo aver eseguito
# MultiLabelBinarizer. Se la somma è uguale a 0, ciò dimostra che il film in particolare non ha generi.
# Quindi, non li includeremo nel training set.
no_label_classes = y.sum(axis=1) == 0
X_train, X_valid, y_train, y_valid = train_test_split(X[~no_label_classes], y[~no_label_classes], test_size=0.3,
                                                      random_state=1234)
# print(X_train.shape, y_train.shape)
# print(X_valid.shape, y_valid.shape)

# Per predire i generi, seguirò questi passaggi:
# 1. Converto le righe overview in features TF-IDF utilizzando TfidfVectorizer
# 2. Addestro e costruisco un modello di classificazione multi-classe
# 3. Effettuo la predizione dei generi in base all'overview data.

vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_valid_vec = vectorizer.transform(X_valid)


# print(X_train_vec)
# print(X_valid_vec)

# Nel modello di classificazione, implementerò i seguenti classificatori:
# 1 - Decision Tree Classifier
# 2 - Random Forest Classifier
# 3 - Extra Trees Classifier
# 4 - Multilayer perceptron Classifier (MLP): modello di rete neurale
# Il vantaggio di MLP Classifier è che questa implementazione funziona con i valori in virgola mobile rappresentati
# come array numpy densi o array scipy sparsi. Dato che i dati di training sono matrici sparse e array numpy, l'MLP ci
# aiuterebbe a costruire un modello di classificazione migliore.

def model_building(modello, parameters=None, cv=10):
    if parameters is None:
        modello.fit(X_train_vec, y_train)
        return modello, modello.predict(X_train_vec), modello.predict(X_valid_vec)
    else:
        model_cv = GridSearchCV(estimator=modello, param_grid=parameters, cv=cv)
        model_cv.fit(X_train_vec, y_train)
        modello = model_cv.best_estimator_

        return model_cv, modello, modello.predict(X_train_vec), modello.predict(X_valid_vec)


# Decision Tree Classifier
dtr = DecisionTreeClassifier()
model_dtr, train_dtr, valid_dtr = model_building(dtr)
print("CLASSIFICATION REPORT : DECISION TREE")
print("Training:\n", classification_report(y_true=y_train, y_pred=train_dtr, target_names=genre_counts))
print("Validation:\n", classification_report(y_true=y_valid, y_pred=valid_dtr, target_names=genre_counts))

print("Accuracy")
train_dtr_acc = accuracy_score(y_true=y_train, y_pred=train_dtr)
valid_dtr_acc = accuracy_score(y_true=y_valid, y_pred=valid_dtr)
print("Training: ", train_dtr_acc)
print("Validation: ", valid_dtr_acc)
print("\n\n\n")

# Random Forest Classifier
rfc = RandomForestClassifier()
model_rfc, train_rfc, valid_rfc = model_building(rfc)
print("CLASSIFICATION REPORT : RANDOM FOREST")
print("Training:\n", classification_report(y_true=y_train, y_pred=train_rfc, target_names=genre_counts))
print("Validation:\n", classification_report(y_true=y_valid, y_pred=valid_rfc, target_names=genre_counts))

print("Accuracy")
train_rfc_acc = accuracy_score(y_true=y_train, y_pred=train_rfc)
valid_rfc_acc = accuracy_score(y_true=y_valid, y_pred=valid_rfc)
print("Training: ", train_rfc_acc)
print("Validation: ", valid_rfc_acc)
print("\n\n\n")

# Extra Trees Classifier
etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
model_etc, train_etc, valid_etc = model_building(etc)
print("CLASSIFICATION REPORT : EXTRA TREES CLASSIFIER")
print("Training:\n", classification_report(y_true=y_train, y_pred=train_etc, target_names=genre_counts))
print("Validation:\n", classification_report(y_true=y_valid, y_pred=valid_etc, target_names=genre_counts))

print("Accuracy")
train_etc_acc = accuracy_score(y_true=y_train, y_pred=train_etc)
valid_etc_acc = accuracy_score(y_true=y_valid, y_pred=valid_etc)
print("Training: ", train_etc_acc)
print("Validation: ", valid_etc_acc)
print("\n\n\n")

# MLP Classifier
mlp = MLPClassifier(verbose=True, max_iter=100, hidden_layer_sizes=100)
model_mlp, train_mlp, valid_mlp = model_building(mlp, cv=10)
print("CLASSIFICATION REPORT : MLP")
print("Training:\n", classification_report(y_true=y_train, y_pred=train_mlp, target_names=genre_counts))
print("Validation:\n", classification_report(y_true=y_valid, y_pred=valid_mlp, target_names=genre_counts))

print("Accuracy")
train_mlp_acc = accuracy_score(y_true=y_train, y_pred=train_mlp)
valid_mlp_acc = accuracy_score(y_true=y_valid, y_pred=valid_mlp)
print("Training: ", train_mlp_acc)
print("Validation: ", valid_mlp_acc)
print("\n\n\n")

# Valutazione delle metriche del dataframe
print("VALUTAZIONE FINALE DELLE PRECISIONI DEI CLASSIFICATORI\n")
train_acc = [train_dtr_acc, train_rfc_acc, train_etc_acc, train_mlp_acc]
valid_acc = [valid_dtr_acc, valid_rfc_acc, valid_etc_acc, valid_mlp_acc]
eval_mat = pd.DataFrame([train_acc, valid_acc], index=['Training', 'Validation'],
                        columns=['Decision Tree Classifier', 'Random Forest Classifier', 'Extra Trees Classifier',
                                 'MLP Classifier'])
print(eval_mat.T)

# Osservazioni:
#
# Gli alberi di decisione tendono a sovradimensionare (overfit) il set di dati di training (come si può anche vedere
# sopra). Possiamo vedere chiaramente che il classificatore dell'albero di decisione non riesce a funzionare bene sul
# set di dati di validazione (dato che le prestazioni si valutano su questo set).

# Random Forest è un algoritmo di bagging e ha un migliore controllo sull'over-fitting. Qui, possiamo vedere che il
# classificatore Random Forest ha prestazioni migliori rispetto all'albero decisionale.

# Il Multi-layer Perceptron (MLP) si basa su reti neurali e utilizza una tecnica di apprendimento supervisionato
# chiamata backpropagation per l'addestramento. Nel dataframe sopra, possiamo vedere chiaramente che MLP Classifier
# offre il meglio tra i 3.
