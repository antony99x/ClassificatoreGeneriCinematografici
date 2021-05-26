import pandas as pd
import decisiontree
import randomforest
import extratrees
import mlp

from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

# Importo il dataset contenente tutti i dati dei film
dataframeCompleto = pd.read_csv('movies_metadata.csv', low_memory=False)
# print(dataframeCompleto.head()) per visualizzare i primi 5 film del dataset

# Creo un nuovo dataframe contenente solo le colonne che mi servono
dataframe = dataframeCompleto[['title', 'tagline', 'original_title', 'overview', 'genres']]

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
generipresenti = dataframe['genres'] != '[]'  # Verifica che per ogni film il genere esiste (true)

# Serie dei generi presenti in movies_metadata
generi = dataframe['genres'][generipresenti]
# print(generi)

# Separo e seleziono i generi
mlb = MultiLabelBinarizer()
etichette = mlb.fit_transform(generi)
classi_etichette = mlb.classes_
# print(classi_etichette)

dati_etichette = pd.DataFrame(etichette, columns=classi_etichette)
val = {}
for x in classi_etichette:
    val.update({x: dati_etichette[x].value_counts()[1]})

# Ordino i generi dei film in base al loro numero di istanze in ordine decrescente
val_ordinati = sorted(val.items(), key=lambda kv: kv[1], reverse=True)
val_pd = pd.DataFrame.from_dict(val_ordinati, orient='columns')
val_pd.rename(columns={0: "Genre", 1: "Count"}, inplace=True)
# print(val_pd)

# In val_pd ci sono generi che sono case cinematografiche o canali tv. Per eliminarli, seleziono soltanto i primi
# 20 generi del dataset utilizzato per la predizione
contatore_fittizio = sorted(val.items(), key=lambda kv: kv[1], reverse=True)[0:20]
# print(contatore_fittizio)
contatore_generi = [i[0] for i in contatore_fittizio]  # Visualizzo i soli generi
# print(contatore_generi)

# genre_counts è l'elenco finale dei generi che verranno utilizzati per l' addestramento del modello
generi_finali = MultiLabelBinarizer(classes=contatore_generi)
top = generi_finali.fit(generi)
# Separo la variabile dipendente
y = generi_finali.transform(generi)  # I generi esclusi in genre_counts verranno ignorati durante l'implementazione
# del MultiLabelBinarizer
# print(generi_finali.classes_)

# Separo la variabile indipendente. La feature 'overview' verrà utilizzata per predire i generi dei film
X = dataframe['overview']
# Se nella lista dei film è presente l'overview ma non il genere, ciò non allena correttamente il nostro modello
# predittivo. Pertanto includerò solo le righe che contengono i generi effettivi e non "[]". Uno dei modi più
# semplici per eseguire questa azione è controllare la somma di ogni riga nei generi dopo aver eseguito
# MultiLabelBinarizer. Se la somma è uguale a 0, ciò dimostra che il film in particolare non ha generi.
# Quindi, non li includeremo nel training set.
film_senza_genere = y.sum(axis=1) == 0
X_train, X_valid, y_train, y_valid = train_test_split(X[~film_senza_genere], y[~film_senza_genere], test_size=0.3,
                                                      random_state=1234)
# print(X_train.shape, y_train.shape)
# print(X_valid.shape, y_valid.shape)

# Per predire i generi, seguirò questi passaggi:
# 1. Converto le righe overview in features TF-IDF utilizzando TfidfVectorizer
# 2. Addestro e costruisco un modello di classificazione multi-classe
# 3. Effettuo la predizione dei generi in base all'overview data.

vettorizzatore = TfidfVectorizer(max_features=1000, stop_words='english', lowercase=True)
X_train_vec = vettorizzatore.fit_transform(X_train)
X_valid_vec = vettorizzatore.transform(X_valid)


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

def costruzione_modello(modello, parameters=None, cv=10):
    if parameters is None:
        modello.fit(X_train_vec, y_train)
        return modello, modello.predict(X_train_vec), modello.predict(X_valid_vec)
    else:
        model_cv = GridSearchCV(estimator=modello, param_grid=parameters, cv=cv)
        model_cv.fit(X_train_vec, y_train)
        modello = model_cv.best_estimator_

        return model_cv, modello, modello.predict(X_train_vec), modello.predict(X_valid_vec)


# Decision Tree Classifier
decisiontree.classificazione(y_train, y_valid, contatore_generi, costruzione_modello)

# Random Forest Classifier
randomforest.classificazione(y_train, y_valid, contatore_generi, costruzione_modello)

# Extra Trees Classifier
extratrees.classificazione(y_train, y_valid, contatore_generi, costruzione_modello)

# MLP Classifier
mlp.classificazione(y_train, y_valid, contatore_generi, costruzione_modello)

# Valutazione delle metriche del dataframe
print("VALUTAZIONE FINALE DELLE PRECISIONI DEI CLASSIFICATORI\n")
train_acc = [decisiontree.getAccTrain(), randomforest.getAccTrain(), extratrees.getAccTrain(), mlp.getAccTrain()]
valid_acc = [decisiontree.getAccValid(), randomforest.getAccValid(), extratrees.getAccValid(), mlp.getAccValid()]
eval_mat = pd.DataFrame([train_acc, valid_acc], index=['Training', 'Validation'],
                        columns=['Classificatore Decision Tree', 'Classificatore Random Forest', 'Classificatore Extra Trees',
                                 'Classificatore MLP'])
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
