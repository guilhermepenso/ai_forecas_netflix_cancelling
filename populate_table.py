import pandas as pd
import numpy as np

# Carregar o arquivo Excel
file_path = './db/netflix_userbase.xlsx'  # Substitua pelo caminho correto do arquivo Excel
df = pd.read_excel(file_path)

# Definir a proporção para "Cancelled" (15%) e "Active" (85%)
statuses = ["Cancelled"] * 15 + ["Active"] * 85
np.random.shuffle(statuses)  # Embaralhar os status

# Garantir que temos status suficientes para todas as linhas do DataFrame
if len(statuses) < len(df):
    statuses *= (len(df) // len(statuses)) + 1  # Repetir a lista para garantir que cubra todas as linhas

# Cortar a lista para o tamanho exato e adicionar a nova coluna "Status"
df['Status'] = statuses[:len(df)]

# Salvar o resultado em um novo arquivo Excel
output_path = './db/netflix_userbase_new.xlsx'
df.to_excel(output_path, index=False)

output_path
