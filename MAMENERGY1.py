import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Definición del modelo SimpleMambaSSM
class SimpleMambaSSM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMambaSSM, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        _, h_n = self.rnn(x)
        out = self.fc(h_n.squeeze(0))
        return out

# Función para cargar y preparar los datos
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('Energyp.csv', sep=';', encoding='latin1')
    df.columns = ['Date', 'Energy']
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Energy'] = df['Energy'].str.replace(',', '.').astype(float)
    df = df.sort_values('Date')
    df['DiaSemana'] = df['Date'].dt.dayofweek
    df['MesAno'] = df['Date'].dt.month
    return df

# Función para inicializar el modelo
def initialize_model():
    st.session_state.model = SimpleMambaSSM(input_dim=2, hidden_dim=64, output_dim=1)
    st.session_state.optimizer = torch.optim.Adam(st.session_state.model.parameters(), lr=0.001)
    st.session_state.criterion = nn.MSELoss()
    st.session_state.losses = []

# Función para crear y entrenar el modelo
def create_and_train_model(X, y):
    losses = []
    for epoch in range(100):
        st.session_state.optimizer.zero_grad()
        outputs = st.session_state.model(X)
        loss = st.session_state.criterion(outputs, y)
        loss.backward()
        st.session_state.optimizer.step()
        losses.append(loss.item())
    return losses

# Función para hacer predicciones
def predict_energy(model, scaler_X, scaler_y, date):
    input_data = torch.tensor([[date.weekday(), date.month]], dtype=torch.float32)
    input_scaled = torch.tensor(scaler_X.transform(input_data.numpy()), dtype=torch.float32)
    with torch.no_grad():
        prediction_scaled = model(input_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled.numpy().reshape(-1, 1))
    return prediction[0][0]

# Interfaz de usuario con Streamlit
st.title('Predictor de Consumo Energético con Mamba SSM')

# Cargar datos
df = load_and_prepare_data()

# Preparar datos para el entrenamiento
X = df[['DiaSemana', 'MesAno']].values
y = df['Energy'].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Inicializar variables de estado
if 'model' not in st.session_state:
    initialize_model()
    st.session_state.training_enabled = True

# Botón para entrenar el modelo
if st.button('Entrenar Modelo') and st.session_state.training_enabled:
    with st.spinner('Entrenando el modelo...'):
        st.session_state.losses = create_and_train_model(X_tensor, y_tensor)
    st.success('Modelo entrenado exitosamente')

    # Visualizar la pérdida durante el entrenamiento
    fig, ax = plt.subplots()
    ax.plot(st.session_state.losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('Pérdida durante el entrenamiento')
    st.pyplot(fig)

# Botón para desactivar el entrenamiento
if st.button('Desactivar Entrenamiento'):
    st.session_state.training_enabled = False
    st.success('Entrenamiento desactivado. El modelo actual se usará para predicciones.')

# Botón para desentrenar el modelo
if st.button('Desentrenar Modelo'):
    initialize_model()
    st.success('Modelo reiniciado a su estado inicial.')

# Mostrar el estado del entrenamiento
st.write(f"Estado del entrenamiento: {'Activado' if st.session_state.training_enabled else 'Desactivado'}")

# Entrada de datos del usuario para predicción
prediction_date = st.date_input("Fecha para predicción", datetime.now() + timedelta(days=7))

if st.button('Generar Predicción'):
    if len(st.session_state.losses) == 0:
        st.error("El modelo no ha sido entrenado aún. Por favor, entrena el modelo primero.")
    else:
        predicted_energy = predict_energy(st.session_state.model, scaler_X, scaler_y, prediction_date)
        st.success(f"Predicción de consumo de energía para {prediction_date}: {predicted_energy:.2f}")
        
        # Visualizar la predicción en el contexto de los datos históricos
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df['Date'], df['Energy'], label='Datos históricos')
        ax.scatter(prediction_date, predicted_energy, color='red', s=100, label='Predicción')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Consumo de Energía')
        ax.set_title('Predicción de Consumo de Energía')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(fig)

st.info('Desarrollado por ARMY para un proyecto de IA')