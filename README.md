# Predicción de Consumo Eléctrico con Red Neuronal Multicapa

## Contexto del Problema
Este proyecto aborda la predicción del consumo máximo de energía eléctrica (en Mega Watts) en una población, basándose en datos horarios de una semana completa. El objetivo es desarrollar un modelo que capture los patrones temporales (diarios y semanales) del consumo eléctrico.

### Características del Dataset
- **Período:** 1 semana (Lunes a Domingo)
- **Frecuencia:** Cada hora (24 registros por día)
- **Total:** 168 observaciones
- **Variable objetivo:** Consumo máximo de energía (MW)
- **Variables predictoras:**
  - Día de la semana (1=Lunes, 7=Domingo)
  - Hora del día (1=1:00 AM, 24=12:00 AM)

## Enfoque del Modelo
Se implementa una **Red Neuronal Feed-Forward Multicapa (MLP)** desde cero con TensorFlow/Keras, utilizando un enfoque de **Backpropagation** para el aprendizaje supervisado. El modelo permite una **arquitectura dinámica** definida por el usuario en tiempo de ejecución.

### Características Principales

- **Arquitectura dinámica:** El usuario define el número de capas ocultas y neuronas por capa interactivamente
- **Backpropagation:** Algoritmo de retropropagación implementado mediante TensorFlow
- **Función de activación:** ReLU en capas ocultas, Sigmoid en capa de salida (clasificación binaria)
- **Optimización:** Adam con learning rate configurable (default: 0.01)
- **Función de pérdida:** Binary Crossentropy (para clasificación binaria del consumo)
- **Métrica de evaluación:** Accuracy y Error Absoluto Medio (MAE)
