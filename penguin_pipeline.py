import pandas as pd
import numpy as np 

# Costanti sono i valori che non cambiano mai durante il programma
# Per convenzione si scrivono in maiuscolo e si lasciano nelle prime righe di codice del modulo
# Così se un giorno vuoi aggiungere una features o modificare le features presenti, le trovi tutte in unico posto

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
LABELS = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}

# Inserisco nelle funzioni che lo richiedono i type hints per anticipare a chi richiama la funzione che tipo di dato passare come argomento
def load_data(path: str):
    """Legge il CSV dei pinguini e restituisce un DataFrame."""
# path è il percorso del file: lo riceviamo come parametro
# invece di scriverlo fisso, così la funzione funziona con qualsiasi file
    return pd.read_csv(path)

def build_features(df: pd.DataFrame): 
    # 1. df_model: df senza le righe a cui manca una delle FEATURES o il sesso
    df_model = df.dropna(subset=FEATURES + ["sex"])
    # 2. le due tabelle one-hot, per island e per sex
    island_onehot = pd.get_dummies(df_model["island"], prefix="island", dtype=int)
    sex_onehot = pd.get_dummies(df_model["sex"], prefix="sex", dtype=int)
    # 3. df_features: le tre tabelle affiancate
    df_features = pd.concat([df_model[FEATURES], island_onehot, sex_onehot], axis=1)
    # 4. X come array NumPy
    X = df_features.to_numpy()
    # 5. y: la specie tradotta con LABELS, come array NumPy
    species = df_model["species"].map(LABELS)
    y = species.to_numpy()
    # 6. return X, y
    return X, y

def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float=0.8, seed: int=42):
    """Divide X e y in train e test in modo casuale e riproducibile.
       Restituisce X_train, X_test, y_train, y_test, in quest'ordine.
       """
    # Il generatore con il seme: stessa divisione a ogni esecuzione
    rng = np.random.default_rng(seed)
    # Le posizioni da 0 a len(X)-1 mescolate
    indices = rng.permutation(len(X))
    # Dati di allenamento, l'80% del totale
    n_train = int(train_ratio*len(X))
    # In train ci vanno tutti i dati da posizione : ad n_train escluso
    train_idx = indices[:n_train]
    # In test ci vanno tutti i dati da posizione n_train incluso alla fine dell'array
    test_idx = indices[n_train:]
    # Faccio si che il dato in riga 0 di X coincida con il dato in posizione 0 di y
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    """Standardizza X_train e X_test con media e deviazione standard del train.

       Restituisce X_train_scaled, X_test_scaled, mean, std, in quest'ordine.
       """
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_scaled = (X_train - mu)/std
    X_test_scaled = (X_test - mu)/std
    return(X_train_scaled, X_test_scaled, mu, std)

       