# vIgilancia: Sistema de Videovigilancia Inteligente Híbrido

![Estado](https://img.shields.io/badge/Estado-Finalizado-green)
![Lenguaje](https://img.shields.io/badge/C++-17-blue)
![Lenguaje](https://img.shields.io/badge/Python-3.8+-yellow)
![Vision](https://img.shields.io/badge/OpenCV-4.x-red)
![AI](https://img.shields.io/badge/PyTorch-Deep_Learning-orange)

## Resumen Ejecutivo

**Security Monitor Based on Computer Vision** es una plataforma de seguridad activa diseñada para transformar cámaras de videovigilancia pasivas en agentes inteligentes de detección. Utilizando una **arquitectura híbrida** de alto rendimiento, el sistema combina la velocidad de **C++** para la detección de intrusos en tiempo real con la potencia de **Python y Deep Learning** para el análisis semántico de comportamientos.

A diferencia de los sistemas tradicionales que solo graban evidencia, vIgilancia detecta, analiza y alerta en el momento exacto en que ocurre un evento, enviando notificaciones multimedia a los usuarios y registrando la actividad en un **Dashboard Web** de monitoreo.

---

## Objetivos del Proyecto

### Objetivo General
Desarrollar un sistema de seguridad inteligente, escalable y de bajo costo, capaz de detectar presencia humana en tiempo real y analizar posturas clave (de pie, caído, agachado) para generar alertas tempranas y reducir la dependencia de la supervisión humana constante.

### Objetivos Específicos
1.  **Detección en Tiempo Real:** Implementar un detector de personas robusto utilizando **HOG (Histograms of Oriented Gradients) + SVM Lineal** en C++, optimizado con técnicas de *downscaling* y gestión eficiente de memoria.
2.  **Reducción de Falsos Positivos:** Entrenar un modelo personalizado utilizando **Hard Negative Mining** (minería de negativos difíciles) para minimizar las falsas alarmas en entornos complejos.
3.  **Análisis de Comportamiento:** Integrar un módulo de Inteligencia Artificial en Python (Keypoint R-CNN) para estimar la postura de los sujetos detectados.
4.  **Monitoreo Centralizado:** Desarrollar un **Dashboard Web** (Flask + TailwindCSS) con estética *Cyberpunk* para la visualización de cámaras, estado del sistema y logs de detección.
5.  **Alertas Inmediatas:** Implementar un bot de **Telegram** que reciba clips de video y notifique al usuario segundos después de una detección confirmada.

---

## Arquitectura del Sistema

El sistema opera mediante dos subsistemas desacoplados que se comunican vía HTTP:

### 1. El "Ojo" (C++ / OpenCV)
*   **Rol:** Captura, Detección y Filtrado.
*   **Tecnología:** C++17, OpenCV 4.
*   **Funcionamiento:** Procesa el feed de video cuadro a cuadro. Utiliza un descriptor HOG personalizado y un SVM entrenado para detectar siluetas humanas.
*   **Lógica de Grabación:** Al detectar movimiento humano, inicia una grabación bufferizada de **5 segundos**. Si la detección persiste o es válida, envía el clip de video al servidor Python.
*   **Ventaja:** Extremadamente rápido y ligero, filtra el 99% del video "vacío" antes de usar recursos de IA pesados.

### 2. El "Cerebro" (Python / Flask / PyTorch)
*   **Rol:** Análisis, Interfaz y Notificación.
*   **Tecnología:** Python, Flask, PyTorch (TorchVision), Telegram API.
*   **Funcionamiento:**
    *   Actúa como servidor web para recibir los videos del módulo C++.
    *   Procesa los clips recibidos con **Keypoint R-CNN** para extraer el esqueleto y determinar la postura.
    *   Actualiza el Dashboard Web en tiempo real.
    *   Envía las alertas finales al usuario vía Telegram.

---

## Stack Tecnológico

*   **Lenguajes:** C++17, Python 3.10+, HTML5, CSS3, JavaScript.
*   **Visión Artificial:** OpenCV 4.6 (C++ y Python).
*   **Deep Learning:** PyTorch, TorchVision (Modelo Keypoint-RCNN ResNet-50).
*   **Backend Web:** Flask, Gunicorn.
*   **Frontend y UI:** TailwindCSS, FontAwesome, Diseño responsive.
*   **Mensajería:** python-telegram-bot.
*   **Herramientas de Desarrollo:** CMake, GCC, Git.

---

## Guía de Instalación

### Prerrequisitos
*   Linux (Ubuntu 20.04/22.04 recomendado).
*   Compilador C++ (g++) y CMake.
*   Python 3.8 o superior.
*   OpenCV 4 instalado en el sistema (`libopencv-dev`).

### 1. Clonar el Repositorio
```bash
git clone https://github.com/David5Uzhca/Security_Monitor_Based_on_Computer_Vision.git
cd Security_Monitor_Based_on_Computer_Vision
```

### 2. Configurar el Entorno Python
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Compilar el Módulo C++
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```
Esto generará dos ejecutables:
*   `train_hog_svm`: Para entrenar el modelo (si se requiere re-entrenar).
*   `real_time_detector`: El detector principal.

### 4. Configuración
Crea un archivo `.env` en la raíz con tus credenciales:
```env
TELEGRAM_TOKEN=tu_token_aqui
TELEGRAM_CHAT_ID=tu_chat_id
```

---

## Uso del Sistema

Para iniciar el sistema completo, necesitas dos terminales:

**Terminal 1: El Servidor (Dashboard y Bot)**
```bash
source venv/bin/activate
python telegram_pose_bot.py
```
*   Accede al Dashboard en: `http://localhost:5000`

**Terminal 2: El Detector (Cámara)**
```bash
cd build
./real_time_detector
```
*   El sistema comenzará a buscar personas. Al detectar una durante 5 segundos, enviará el video automáticamente al Dashboard y a Telegram.

---

## Mejoras Futuras
*   Integración de reconocimiento facial.
*   Soporte para múltiples cámaras IP simultáneas.
*   Contenedorización con Docker para despliegue fácil.
*   Análisis de trayectoria y mapas de calor.

---
**Autor:**\
David Uzhca\
Tania Lojano
