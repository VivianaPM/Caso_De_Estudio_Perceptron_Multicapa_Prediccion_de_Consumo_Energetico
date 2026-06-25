# Tratamiento de datos
import os
import pandas as pd
import numpy as np
# -------------------------------------------------
# Graficos 
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
# -------------------------------------------------
# Reducir el nivel de registro de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Desactivar operaciones personalizadas oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Modelado
import tensorflow as tf
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):
        # Verificar si el archivo existe antes de cargarlo
        if not os.path.exists(self.filepath):
            print(f"El archivo {self.filepath} no se encuentra.")
            return None, None, None, None, None, None
        
        # Intentar leer el archivo
        try:
            table = pd.read_csv(self.filepath, sep=';', header=None)
            print("Archivo cargado correctamente.")
        except Exception as e:
            print(f"Error al leer el archivo: {e}")
            return None, None, None, None, None, None

        # Normalización Min-Max solo en la última columna
        table[table.columns[-1]] = pd.to_numeric(table[table.columns[-1]], errors='coerce')
        def min_max_scaling(column):
            return (column - column.min()) / (column.max() - column.min())
        
        table[table.columns[-1]] = min_max_scaling(table[table.columns[-1]])

        # Separar en características y objetivo, considerando que la primera columna indica el día de la semana
        X = table.iloc[:, :-1]
        y = table.iloc[:, -1]

        # División de los datos
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test

class NeuralNetwork:

    def Model():
        inputSize = 2
        outputSize = 1
        C_hiddenlayer = int(input("Ingrese el numero de capas ocultas: "))
        learning_rate = 0.01

        hiddenLayer = []
        
        for i in range(C_hiddenlayer):
            neurons = int(input(f"Ingrese la cantidad de neuronas para la capa oculta {i}: "))
            hiddenLayer.append((neurons))
        return inputSize, outputSize, hiddenLayer, learning_rate


    def build_model(input_size, output_size, hidden_layers):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_size,)))

        for neurons in hidden_layers:
            model.add(tf.keras.layers.Dense(neurons, activation='relu'))

        model.add(tf.keras.layers.Dense(output_size, activation='sigmoid'))
        return model
    
    def train(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs, learning_rate):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])
        
        # Añadir datos de validación
        historial = model.fit(X_train, y_train, 
                            epochs=epochs, 
                            validation_data=(X_val, y_val),
                            steps_per_epoch=len(X_train) // epochs if len(X_train) >= epochs else 1,
                            verbose=1)
        # Errores
        test_loss, test_mae = model.evaluate(X_test, y_test)
        print(f'Error cuadrático medio: {test_loss}')
        print(f'Error absoluto medio: {test_mae}')

        # Graficar el error acumulado
        grahips.ErrorAcum(historial)
        return model

    
    def generar_predicciones_diarias(model, X_test, y_test):
        dias_semana = {1: "Lunes", 2: "Martes", 3: "Miércoles", 4: "Jueves", 5: "Viernes", 6: "Sábado", 7: "Domingo"}
        valores_reales = []
        predicciones = []

        # Agrupamos los datos en X_test y y_test por los días de la semana en la primera columna
        for dia, nombre_dia in dias_semana.items():
            X_dia = X_test[X_test.iloc[:, 0] == dia]
            y_dia = y_test[X_test.iloc[:, 0] == dia]

            if X_dia.empty:
                print(f"Advertencia: No hay datos para el día {nombre_dia}")
                continue

            # Realizar la predicción para los datos del día actual
            prediccion_dia = model.predict(X_dia)
            valores_reales.append(y_dia.values)
            predicciones.append(prediccion_dia.flatten())

        # Graficar días con datos
        grahips.preduction(list(dias_semana.values()), valores_reales, predicciones)

class grahips():
    def ErrorAcum(historial):
        plt.figure(figsize=(10, 5))
        plt.plot(historial.history['loss'], label='Pérdida en el entrenamiento')
        plt.plot(historial.history['val_loss'], label='Pérdida en la validación')
        plt.title('Error Acumulado')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()
        
    def preduction(dias, valores_reales, predicciones):
        ventana = tk.Tk()
        ventana.title("Comparación de Valores Reales y Predicciones por Día")

        fig, axs = plt.subplots(2, 4, figsize=(15, 8))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for i, dia in enumerate(dias):
            if i < len(valores_reales):  # Solo graficar días con datos
                ax = axs[i // 4, i % 4]
                ax.plot(range(1, len(valores_reales[i]) + 1), valores_reales[i], label="Valor Real", marker='o', color="blue")
                ax.plot(range(1, len(predicciones[i]) + 1), predicciones[i], label="Predicción", marker='x', color="orange")
                ax.set_title(f"Día {dia}")
                ax.set_xticks(range(1, len(valores_reales[i]) + 1))
                ax.set_xlabel("Horas")
                ax.set_ylabel("Valores")
                ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=ventana)
        canvas.draw()
        canvas.get_tk_widget().pack()

        ventana.mainloop()