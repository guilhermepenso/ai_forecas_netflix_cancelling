import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tkinter as tk
from tkinter import ttk, messagebox

# Configurações do matplotlib para o Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Carregar os dados
df = pd.read_excel("./db/netflix_userbase.xlsx")

# Codificação das variáveis categóricas
label_encoder_subscription = LabelEncoder()
label_encoder_device = LabelEncoder()
label_encoder_country = LabelEncoder()
label_encoder_gender = LabelEncoder()
label_encoder_status = LabelEncoder()

df["Subscription_encoded"] = label_encoder_subscription.fit_transform(df["Subscription Type"])
df["Device_encoded"] = label_encoder_device.fit_transform(df["Device"])
df["Country_encoded"] = label_encoder_country.fit_transform(df["Country"])
df["Gender_encoded"] = label_encoder_gender.fit_transform(df["Gender"])
df["Status_encoded"] = label_encoder_status.fit_transform(df["Status"])  # Status como variável alvo

# Selecionar as variáveis de entrada (features) e a variável alvo (target)
X = df[["Age", "Device_encoded", "Country_encoded", "Gender_encoded", "Subscription_encoded"]]
y = df["Status_encoded"]

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar os dados para normalizar as entradas, mas manter os valores originais da idade
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construção e treinamento do modelo de rede neural
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Saída sigmoide para prever probabilidade de cancelamento
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test), verbose=1)

# Fazer previsões no conjunto de teste e calcular a probabilidade de cancelamento
y_pred_prob = model.predict(X_test_scaled)

# Adicionar probabilidade de cancelamento ao conjunto de teste (em porcentagem)
X_test_df = X_test.copy()  # Usar os dados originais não normalizados para visualização
X_test_df['Cancel_Probability'] = (y_pred_prob * 100).flatten()  # Converter para porcentagem
X_test_df['Actual_Status'] = y_test.values
X_test_df['Predicted_Status'] = (y_pred_prob > 0.5).astype(int)

# Remapear as variáveis codificadas para os valores originais para visualização
X_test_df['Device'] = label_encoder_device.inverse_transform(X_test_df['Device_encoded'].astype(int))
X_test_df['Country'] = label_encoder_country.inverse_transform(X_test_df['Country_encoded'].astype(int))
X_test_df['Subscription'] = label_encoder_subscription.inverse_transform(X_test_df['Subscription_encoded'].astype(int))
X_test_df['Gender'] = label_encoder_gender.inverse_transform(X_test_df['Gender_encoded'].astype(int))

# Filtrar usuários com alta probabilidade de cancelamento
high_risk_users = X_test_df[X_test_df["Cancel_Probability"] > 30]

# Função para exibir os top 10 usuários "Active" com maior probabilidade de cancelamento baseado em um filtro
def top_cancel_risk_with_filter():
    column_name = column_choice.get()
    column_value = filter_entry.get()
    
    if column_name and column_value:  # Se ambos foram selecionados, aplica o filtro
        if column_name not in X_test_df.columns:
            messagebox.showerror("Erro", f"A coluna '{column_name}' não existe no conjunto de dados.")
            return

        # Se o valor do filtro for um número (como idade), converte para inteiro
        if column_name == "Age" and column_value.isdigit():
            filtered_users = X_test_df[(X_test_df["Actual_Status"] == 0) & (X_test_df["Age"] == int(column_value))]
        else:
            filtered_users = X_test_df[(X_test_df["Actual_Status"] == 0) & (X_test_df[column_name] == column_value)]
    else:
        # Se não houver filtro selecionado, apenas exibe os top 10 usuários "Active"
        filtered_users = X_test_df[X_test_df["Actual_Status"] == 0]

    # Ordenar os usuários pela probabilidade de cancelamento e pegar os 10 com maior probabilidade
    top_cancel_risk_users = filtered_users.sort_values(by="Cancel_Probability", ascending=False).head(10)
    
    # Exibir o resultado na tabela
    for row in result_table.get_children():
        result_table.delete(row)
    for _, row in top_cancel_risk_users.iterrows():
        row["Cancel_Probability"] = f"{row['Cancel_Probability']:.2f}%"  # Formatar com 2 casas decimais e símbolo de %
        result_table.insert("", "end", values=[row.name] + row[["Age", "Device", "Country", "Gender", "Subscription", "Cancel_Probability", "Actual_Status", "Predicted_Status"]].tolist())


# Função para exibir gráficos lado a lado no Tkinter
def show_graphs():
    fig, axes = plt.subplots(2, 3, figsize=(18, 6))  # Diminuição da altura dos gráficos

    sns.barplot(data=high_risk_users, x="Age", y="Cancel_Probability", color="blue", ax=axes[0, 0], errorbar=None)
    axes[0, 0].set_title("Por Idade")
    axes[0, 0].set_xlabel("Idade")
    axes[0, 0].set_ylabel("Porcentagem (%)")

    sns.barplot(data=high_risk_users, x="Device", y="Cancel_Probability", palette="viridis", ax=axes[0, 1], errorbar=None)
    axes[0, 1].set_title("Por Dispositivo")
    axes[0, 1].set_xlabel("Dispositivo")
    axes[0, 1].set_ylabel("Porcentagem (%)")

    sns.barplot(data=high_risk_users, x="Country", y="Cancel_Probability", palette="viridis", ax=axes[0, 2], errorbar=None)
    axes[0, 2].set_title("Por País")
    axes[0, 2].set_xlabel("País", fontsize=10)  # Diminuir tamanho do xlabel para o gráfico de País
    axes[0, 2].set_ylabel("Porcentagem (%)")

    sns.barplot(data=high_risk_users, x="Subscription", y="Cancel_Probability", palette="viridis", ax=axes[1, 0], errorbar=None)
    axes[1, 0].set_title("Por Tipo de Assinatura")
    axes[1, 0].set_xlabel("Tipo de Assinatura")
    axes[1, 0].set_ylabel("Porcentagem (%)")

    sns.barplot(data=high_risk_users, x="Gender", y="Cancel_Probability", palette="viridis", ax=axes[1, 1], errorbar=None)
    axes[1, 1].set_title("Por Gênero")
    axes[1, 1].set_xlabel("Gênero")
    axes[1, 1].set_ylabel("Porcentagem (%)")

    # Gráfico de distribuição geral de usuários Active vs Cancelled
    status_counts = X_test_df["Actual_Status"].value_counts(normalize=True) * 100
    sns.barplot(x=["Active", "Cancelled"], y=status_counts, palette="viridis", ax=axes[1, 2])
    axes[1, 2].set_title("Distribuição de Status")
    axes[1, 2].set_ylabel("Porcentagem (%)")

    fig.tight_layout()
    
    # Inserir gráfico no Tkinter
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Função para limpar o filtro
def clear_filter():
    column_choice.set("")  # Limpar o valor da coluna selecionada
    filter_entry.set("")  # Limpar o valor do filtro
    # Exibir os top 10 usuários "Active" sem nenhum filtro
    top_cancel_risk_with_filter()

# Configuração do Tkinter
root = tk.Tk()
root.title("Análise de Probabilidade de Cancelamento")

# Frame para gráficos
graph_frame = tk.Frame(root)
graph_frame.pack()

# Exibir gráficos na inicialização
show_graphs()

# Seção de filtro
filter_frame = tk.Frame(root)
filter_frame.pack(pady=10)

# Seleção da coluna para o filtro (incluindo Age)
tk.Label(filter_frame, text="Selecionar coluna para filtro:").grid(row=0, column=0)
column_choice = ttk.Combobox(filter_frame, values=["Age", "Device", "Country", "Gender", "Subscription"], state="readonly")
column_choice.grid(row=0, column=1)

# Campo de entrada para valor do filtro
tk.Label(filter_frame, text="Valor do filtro:").grid(row=1, column=0)
filter_entry = ttk.Combobox(filter_frame)
filter_entry.grid(row=1, column=1)

# Atualizar valores do filtro conforme a coluna selecionada
def update_filter_entry(*args):
    column_name = column_choice.get()
    if column_name in X_test_df.columns:
        filter_entry["values"] = list(X_test_df[column_name].unique())
        filter_entry.set("")  # Limpar o valor anterior

column_choice.bind("<<ComboboxSelected>>", update_filter_entry)

# Botão para exibir os resultados do filtro
btn_filter = tk.Button(filter_frame, text="Filtrar e Exibir Top 10", command=top_cancel_risk_with_filter)
btn_filter.grid(row=4, column=0, columnspan=2, pady=5)

# Botão para limpar o filtro
btn_clear_filter = tk.Button(filter_frame, text="Limpar Filtro", command=clear_filter)
btn_clear_filter.grid(row=4, column=2, pady=5)  # Botão ao lado do filtro

# Tabela para exibir os resultados
result_table = ttk.Treeview(root, columns=("ID", "Age", "Device", "Country", "Gender", "Subscription", "Cancel_Probability"), show="headings", height=10)
result_table.pack(padx=10, pady=10, fill="x")

# Definir cabeçalhos para a tabela
for col in result_table["columns"]:
    result_table.heading(col, text=col, anchor="w")
    result_table.column(col, width=120, anchor="w", minwidth=100)

root.mainloop()
