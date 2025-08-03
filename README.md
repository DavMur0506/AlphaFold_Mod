# 🧬 Optimización del primer bloque de AlphaFold mediante el uso de redes neuronales profundas

Este proyecto implementa un modelo basado en **Transformers** para predecir directamente las características de entrada (`features.pkl`) utilizadas por **AlphaFold2**, con el objetivo de eliminar la dependencia de herramientas externas como JackHMMER o MMseqs2 para la generación de alineamientos múltiples de secuencias (MSA).

## 📌 Descripción del proyecto

- Se desarrolló un modelo **Transformer multicapa** (`EnhancedTransformer`) capaz de generar representaciones equivalentes a las obtenidas por AlphaFold2 tras el procesamiento del MSA.  
- Se creó un conjunto de datos mejorado (`EnhancedSeqDataset`) a partir de archivos `features.pkl` generados por AlphaFold2.  
- Todas las features fueron **truncadas a una longitud máxima de 1024** para reducir el consumo de memoria.  
- Se implementaron funciones para **comparar las predicciones del modelo con las features reales de AlphaFold2**.

## 📂 Estructura del proyecto

