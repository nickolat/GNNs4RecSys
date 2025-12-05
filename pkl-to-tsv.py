import pickle
import pandas as pd

# Percorsi file
input_file = r"C:\Users\Nicola\DataspellProjects\LightGCL-main\data\amazon\tstMat.pkl"
output_file = 'test.tsv'

# Carica la matrice COO dal file .pkl
with open(input_file, 'rb') as f:
    coo = pickle.load(f)

# Assicurati che sia una coo_matrix
from scipy.sparse import coo_matrix
if not isinstance(coo, coo_matrix):
    raise ValueError("Il file non contiene una coo_matrix")

# Crea un DataFrame con le colonne UserID e ItemID
df = pd.DataFrame({
    'UserID': coo.row,
    'ItemID': coo.col
})

# Ordina
df.sort_values(['UserID', 'ItemID'], inplace=True)

# Salva in formato TSV senza header e senza index
df.to_csv(output_file, sep='\t', header=False, index=False)
