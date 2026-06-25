import numpy as np
import neuralNetwork as n
from neuralNetwork import NeuralNetwork

def main():
    data_loader = n.DataLoader('datosEntrenamiento.csv')
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_data()

    if X_train is not None:
        input_size, output_size, hidden_layers, learning_rate = NeuralNetwork.Model()
        model = NeuralNetwork.build_model(input_size, output_size, hidden_layers)
        model = NeuralNetwork.train(model, X_train.values, y_train.values, X_val.values, y_val.values, X_test.values, y_test.values, epochs=9000, learning_rate=learning_rate)
        print("Entrenamiento completado.")

        # Pedir al usuario nuevos datos para predicciones
        print("Nuevos datos de entrada para obtener predicciones:")
        new_data = []
        day = float(input("Dia en [1 - 7]: "))
        new_data.append(day)
        hours = float(input("Horas: "))
        new_data.append(hours)
        new_data = np.array(new_data).reshape(1, -1)

        # Hacer una predicción con los nuevos datos
        predictions = model.predict(new_data)
        print("Predicción:")
        print(predictions[0])

        print(f"X_test: {X_test.shape}")
        print(f"Y_test: {y_test.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"Y_val {y_val.shape}")
        print(f"X_train: {X_train.shape}")
        print(f"Y_train: {y_train.shape}")

        # Graficas de valor con dia y hora
        NeuralNetwork.generar_predicciones_diarias(model, X_train, y_train)


    else:
        print("Error al cargar los datos")

if __name__ == "__main__":
    main()
