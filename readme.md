# Security Monitor Based on Computer Vision

[![C++](https://img.shields.io/badge/Language-C%2B%2B17-00599C.svg?logo=cplusplus)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Language-Python%203.10+-3776AB.svg?logo=python)](https://python.org/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV%204.x-5C3EE8.svg?logo=opencv)](https://opencv.org/)
[![Deep Learning](https://img.shields.io/badge/Inference-PyTorch%20%7C%20Keypoint%20R--CNN-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Alerts](https://img.shields.io/badge/Notifications-Telegram%20Bot%20API-2CA5E0.svg?logo=telegram)](https://core.telegram.org/bots/api)
[![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)](#)

> **Plataforma híbrida de videovigilancia activa y análisis postural en tiempo real.**  
> Desacoplamiento de detección rápida en hardware nativo (**C++17 / OpenCV**) e inferencia semántica profunda (**Python / PyTorch**).

---

## 1. Resumen de Ingeniería

Los sistemas de videovigilancia convencionales son eminentemente pasivos: almacenan cientos de horas de video continuo sin valor predictivo y sobrecargan a los operadores humanos. Por otro lado, ejecutar redes neuronales profundas (CNN/Transformers) cuadro a cuadro sobre streams continuos de 1080p es computacionalmente prohibitivo para implementaciones en hardware de borde (*edge devices*).

**Security Monitor** resuelve este dilema mediante un **diseño arquitectónico híbrido de dos niveles**:

1. **Nivel 1 ("El Ojo" - C++17 / OpenCV):** Ejecuta continuamente sobre el stream de video a máxima tasa de cuadros (30+ FPS). Utiliza un descriptor clásico **HOG (Histograms of Oriented Gradients)** acoplado a un clasificador **Linear SVM** entrenado mediante **Hard Negative Mining** para detectar presencia humana con consumo mínimo de CPU y memoria RSS. Al confirmar una detección sostenida, captura un buffer de video circular de 5 segundos.
2. **Nivel 2 ("El Cerebro" - Python / PyTorch / Flask):** Se activa únicamente bajo demanda al recibir el clip de 5 segundos. Ejecuta un modelo de **Keypoint R-CNN (ResNet-50 FPN)** para estimar las coordenadas anatómicas de 17 puntos clave del sujeto, clasificando su postura (*de pie, agachado, caída crítica*) y despachando alertas multimedia inmediatas vía **Telegram** y registrando la telemetría en un **Dashboard Web**.

---

## 2. Diagrama de Arquitectura y Flujo de Datos

```
[ Cámara RTSP / USB ]
         | (Video Stream raw)
         v
+-------------------------------------------------------------+
|              SUBSISTEMA C++ (Detección Rápida)              |
|                                                             |
|   1. Captura Frame & Preprocesamiento (Downscaling)         |
|   2. Extractor de Características HOG (64x128 píxeles)      |
|   3. Clasificador Linear SVM (Optimizado con Hard Negatives)|
|   4. Non-Maximum Suppression (NMS) & Buffer de Decisión     |
|   5. Grabación Circular (5s MP4 con VideoWriter)            |
+------------------------------+------------------------------+
                               |
                   HTTP POST / IPC Multipart
                   (Clip 5 segundos solo si hay evento)
                               |
                               v
+-------------------------------------------------------------+
|              SUBSISTEMA PYTHON (Análisis Semántico)         |
|                                                             |
|   1. API Gateway Flask (Recepción asíncrona)                |
|   2. Decodificación de Frames & Muestreo                    |
|   3. Inferencia PyTorch (Keypoint R-CNN)                    |
|   4. Clasificación de Postura:                              |
|      - Normal (De pie)                                      |
|      - Sospechoso (Agachado / Oculto)                       |
|      - Emergencia (Caída detectada)                         |
+------------------------------+------------------------------+
                               |
             +-----------------+-----------------+
             |                                   |
             v                                   v
+-------------------------+         +-------------------------+
|     Telegram Bot API    |         |   Dashboard Web UI      |
|  - Alerta Push en <2s   |         |  - Flask + TailwindCSS  |
|  - Video MP4 adjunto    |         |  - Bitácora de Eventos  |
|  - Diagnóstico Postural |         |  - Stream de Monitoreo  |
+-------------------------+         +-------------------------+
```

---

## 3. Aspectos Técnicos Destacados

### 3.1 Hard Negative Mining para Mitigación de Falsos Positivos
Un detector HOG estándar suele disparar falsas alarmas ante patrones de fondo complejos (sombras, ramas, muebles). Se implementó un pipeline de entrenamiento en dos etapas:
* **Etapa 1:** Entrenamiento inicial de Linear SVM con positivos etiquetados y negativos aleatorios.
* **Etapa 2 (Mining):** Evaluación del detector sobre miles de imágenes sin personas; los falsos positivos detectados (*Hard Negatives*) se recolectan automáticamente y se inyectan en una segunda ronda de reentrenamiento, logrando una reducción drástica de falsos disparos en el módulo C++.

### 3.2 Eficiencia de Memoria y Métricas en Tiempo Real
El ejecutable C++ implementa monitorización nativa de memoria residente (**RSS**) leyendo directamente el descriptor del kernel Linux (`/proc/self/statm`), garantizando que no existan fugas de memoria (*memory leaks*) en el bucle principal de adquisición de video de OpenCV.

---

## 4. Estructura del Repositorio

```
Security_Monitor_Based_on_Computer_Vision/
├── CMakeLists.txt              # Configuración de compilación CMake (C++17)
├── real_time_detector.cpp      # Módulo principal C++ de detección en vivo y buffer
├── train_hog_svm.cpp           # Pipeline de entrenamiento HOG + Linear SVM
├── train_hog_fast.cpp          # Versión optimizada con multithreading OpenMP
├── prepare_dataset.py          # Scripts de preparación y aumento de datos
├── evaluation.py               # Métricas de precisión, exhaustividad y curva ROC
├── telegram_pose_bot.py        # Servidor Flask + Inferencia PyTorch + Bot Telegram
├── test_api.py                 # Pruebas automatizadas de comunicación HTTP
├── requirements.txt            # Dependencias del entorno Python
└── templates/
    └── index.html              # Interfaz de monitoreo web en tiempo real
```

---

## 5. Guía de Compilación y Ejecución

### Prerrequisitos
* **Linux:** Ubuntu 20.04 / 22.04 LTS o Kubuntu.
* **Compilador:** GCC/G++ con soporte C++17 y CMake $\ge$ 3.16.
* **Bibliotecas:** `libopencv-dev` (OpenCV 4.x), `libcurl4-openssl-dev`.
* **Python:** 3.10+ con soporte CUDA (opcional para acelerar PyTorch).

### Paso 1: Configurar el Entorno Python
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Paso 2: Compilar el Núcleo C++ con CMake
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```
Generará el ejecutable binario de alto rendimiento: `real_time_detector`.

### Paso 3: Iniciar el Servidor de Inferencia y Notificaciones
```bash
# Exportar el token de Telegram provisto por BotFather
export TELEGRAM_BOT_TOKEN="TU_TOKEN_AQUI"
export TELEGRAM_CHAT_ID="TU_CHAT_ID_AQUI"

python3 telegram_pose_bot.py
```

### Paso 4: Lanzar el Detector de Borde en C++
```bash
./build/real_time_detector
```
El detector procesará la señal de cámara en tiempo real; ante cualquier evento, empaquetará el clip y lo despachará al servidor Python de forma no bloqueante.
