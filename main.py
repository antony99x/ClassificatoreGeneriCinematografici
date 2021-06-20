import pandas as pd
import decisiontree
import randomforest
import extratrees
import mlp

from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from google_trans_new import google_translator
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'

# Importo il dataset contenente tutti i dati dei film
dataframeCompleto = pd.read_csv('movies_metadata.csv', low_memory=False)
# print(dataframeCompleto.head()) per visualizzare i primi 5 film del dataset

# Creo un nuovo dataframe contenente solo le colonne che mi servono
dataframe = dataframeCompleto[['title', 'overview', 'genres']]

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
# print(etichette)
classi_etichette = mlb.classes_
# print(classi_etichette)

dati_etichette = pd.DataFrame(etichette, columns=classi_etichette)
# print(dati_etichette)
val = {}
for x in classi_etichette:
    val.update({x: dati_etichette[x].value_counts()[1]})
# print(val)

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

# contatore_generi è l'elenco finale dei generi che verranno utilizzati per l' addestramento del modello
generi_finali = MultiLabelBinarizer(classes=contatore_generi)
# Separo la variabile dipendente
y = generi_finali.fit_transform(generi)
# I generi esclusi in contatore_generi verranno ignorati durante l'implementazione del MultiLabelBinarizer
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
                                                      random_state=42)
# print(X_train.shape, y_train.shape)
# print(X_valid.shape, y_valid.shape)

# Per predire i generi, seguirò questi passaggi:
# 1. Converto le righe overview in features TF-IDF utilizzando TfidfVectorizer
# 2. Addestro e costruisco un modello di classificazione multi-classe
# 3. Effettuo la predizione dei generi in base all'overview data.

vettorizzatore = TfidfVectorizer(max_features=1000, stop_words='english', lowercase=True)
X_train_vec = vettorizzatore.fit_transform(X_train)
X_valid_vec = vettorizzatore.transform(X_valid)


# print(X_train_vec, X_valid_vec)


# Nel modello di classificazione, implementerò i seguenti classificatori:
# 1 - Decision Tree Classifier
# 2 - Random Forest Classifier
# 3 - Extra Trees Classifier
# 4 - Multilayer perceptron Classifier (MLP): modello di rete neurale

def costruzione_modello(modello, parameters=None, cv=10):
    if parameters is None:
        modello.fit(X_train_vec, y_train)
        return modello, modello.predict(X_train_vec), modello.predict(X_valid_vec)
    else:
        model_cv = RandomizedSearchCV(estimator=modello, param_distributions=parameters, cv=cv, verbose=3)
        model_cv.fit(X_train_vec, y_train)
        modello = model_cv.best_estimator_
        # print('Best parameters found: ', model_cv.best_params_, '\nBest score found: ', model_cv.best_score_)
        return model_cv, modello, modello.predict(X_train_vec), modello.predict(X_valid_vec)


# Decision Tree Classifier
model_dtr = decisiontree.classificazione(y_train, y_valid, contatore_generi, costruzione_modello)

# Random Forest Classifier
model_rfc = randomforest.classificazione(y_train, y_valid, contatore_generi, costruzione_modello)

# Extra Trees Classifier
model_etc = extratrees.classificazione(y_train, y_valid, contatore_generi, costruzione_modello)

# MLP Classifier
model_mlp = mlp.classificazione(y_train, y_valid, contatore_generi, costruzione_modello)

# Valutazione delle metriche del dataframe
print("VALUTAZIONE FINALE DELLE PRECISIONI DEI CLASSIFICATORI\n")
train_acc = [decisiontree.getAccTrain(), randomforest.getAccTrain(), extratrees.getAccTrain(), mlp.getAccTrain()]
valid_acc = [decisiontree.getAccValid(), randomforest.getAccValid(), extratrees.getAccValid(), mlp.getAccValid()]
eval_mat = pd.DataFrame([train_acc, valid_acc], index=['Training', 'Validation'],
                        columns=['Classificatore Decision Tree', 'Classificatore Random Forest',
                                 'Classificatore Extra Trees',
                                 'Classificatore MLP'])
print(eval_mat.T)
print("\n\n\n")

# Osservazioni:

# Gli alberi di decisione tendono a sovradimensionare (overfit) il set di dati di training (come si può anche vedere
# sopra). Possiamo vedere chiaramente che il classificatore dell'albero di decisione non riesce a funzionare bene sul
# set di dati di validazione (dato che le prestazioni si valutano su questo set).

# Random Forest è un algoritmo di bagging e ha un migliore controllo sull'over-fitting. Qui, possiamo vedere che il
# classificatore Random Forest ha prestazioni migliori rispetto all'albero decisionale.

# Extremely Randomized Trees (Extra Trees) è un algoritmo di apprendimento automatico, il quale crea un gran numero di
# alberi di decisione dal training set e la classificazione viene fatta tramite un voto di maggioranza. A differenza
# dell'algoritmo Random Forest, Extra Trees seleziona il miglior split randomicamente.

# Il Multi-Layer Perceptron (MLP) si basa su reti neurali e utilizza una tecnica di apprendimento supervisionato
# chiamata backpropagation per l'addestramento.


# Fase di scelta del classificatore da utilizzare

valido = False
while not valido:
    dec_classifier = input('In base alla valutazione delle precisioni dei vari classificatori appena '
                           'analizzati, digita:\n'
                           '1 se vuoi usare il Decision Tree Classifier;\n'
                           '2 se vuoi usare il Random Forest Classifier;\n'
                           '3 se vuoi usare l\'Extra Trees Classifier;\n'
                           '4 se vuoi usare il Multi-Layer Perceptron (MLP).\n')
    if dec_classifier == '1':
        valido = True
        classificatore_scelto = model_dtr
    elif dec_classifier == '2':
        valido = True
        classificatore_scelto = model_rfc
    elif dec_classifier == '3':
        valido = True
        classificatore_scelto = model_etc
    elif dec_classifier == '4':
        valido = True
        classificatore_scelto = model_mlp
    else:
        print('Digita una scelta valida.')


# Fase di input del set di film da cui ricavare il genere (applicazione del modello)

valido = False
while not valido:
    decisione = input('Importa un set di film da cui ricavare il genere.\n'
                      'Digita 1 se vuoi importarne uno preesistente da file. Digita 2 se vuoi crearne uno.\n')
    if decisione == '1':
        valido = True
        set_input = pd.read_excel('set_prova.xlsx', index_col='title')
        # print(set_input)

    elif decisione == '2':
        valido = True
        set_input = pd.DataFrame(columns=['title', 'overview'])
        scelta_input = '1'
        while scelta_input == '1':
            titolo = input('Inserisci il titolo del film: ')
            trama = input('Inserisci la trama del film: ')
            translator = google_translator()
            tramaEN = translator.translate(trama, lang_src='it', lang_tgt='en')
            # print(tramaEN)
            set_input.loc[len(set_input.index)] = [titolo, tramaEN]
            scelta_input = input('Vuoi aggiungere un altro film al set? Digita 1 se sì, 0 altrimenti: ')
        set_input.set_index('title', inplace=True)
        # print(set_input)

    else:
        print('Digita una scelta valida.')

X_test = set_input['overview']
# print(X_test)
X_test_vec = vettorizzatore.transform(X_test)
# print(X_test_vec)
risultati_finali = pd.DataFrame(classificatore_scelto.predict(X_test_vec), columns=contatore_generi,
                                index=set_input.index)
# print(risultati_finali)
generi_previsti = risultati_finali.apply(lambda p: p.index[p.astype(bool)].tolist(), 1)
print(generi_previsti)
