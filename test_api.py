#!/usr/bin/env python3
"""
Script para probar la API Flask
Verifica que el endpoint /process_video funciona correctamente.
"""

import requests
import cv2
import os
import numpy as np

print("=" * 60)
print("  PRUEBA DE LA API FLASK")
print("=" * 60)
print()

# 1. Verificar que la API esté corriendo
print("1️⃣  Verificando que la API esté activa...")
try:
    response = requests.get("http://localhost:5000/health", timeout=5)
    if response.status_code == 200:
        print("   ✅ API Flask está corriendo")
        data = response.json()
        print(f"      Status: {data.get('status')}")
        print(f"      Modelo: {data.get('model', 'N/A')}")
        print(f"      PyTorch: {data.get('pytorch_version', 'N/A')}")
        print(f"      Device: {data.get('device', 'N/A')}")
    else:
        print(f"   ❌ API respondió con código: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("   ❌ No se puede conectar a la API")
    print()
    print("   Solución:")
    print("   1. Asegúrate de que el bot esté corriendo:")
    print("      python telegram_pose_bot.py")
    print()
    print("   2. Espera a que veas el mensaje:")
    print("      '🌐 API FLASK INICIADA'")
    exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

print()

# 2. Crear un video de prueba
print("2️⃣  Creando video de prueba...")
os.makedirs("temp", exist_ok=True)
test_video_path = "temp/test_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(test_video_path, fourcc, 30.0, (640, 480))

# Generar 90 frames (3 segundos a 30fps)
for i in range(90):
    # Crear un frame con degradado y texto
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Agregar algo de color para que no sea todo negro
    frame[:, :, 0] = 50   # Canal azul
    frame[:, :, 1] = 30   # Canal verde
    frame[:, :, 2] = 20   # Canal rojo
    
    cv2.putText(frame, f"Frame de prueba {i}", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Test API", (220, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    out.write(frame)

out.release()

file_size = os.path.getsize(test_video_path) / 1024
print(f"   ✅ Video de prueba creado: {test_video_path} ({file_size:.1f} KB)")
print()

# 3. Enviar video a la API
print("3️⃣  Enviando video a la API...")
try:
    with open(test_video_path, 'rb') as f:
        files = {'video': ('test_video.mp4', f, 'video/mp4')}
        response = requests.post(
            "http://localhost:5000/process_video",
            files=files,
            timeout=120
        )

    if response.status_code == 200:
        print("   ✅ Video procesado correctamente")
        result = response.json()
        print(f"      Status: {result.get('status')}")
        print(f"      Output: {result.get('output')}")
        print(f"      Archivos enviados a Telegram: {result.get('files_sent', 'N/A')}")

        output_path = result.get('output')
        if output_path and os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"   ✅ Video de salida existe: {output_path} ({size_mb:.2f} MB)")
        else:
            print("   ⚠️  Video de salida no encontrado localmente (puede haberse limpiado)")
    else:
        print(f"   ❌ Error del servidor: {response.status_code}")
        print(f"      Respuesta: {response.text[:500]}")

except requests.exceptions.Timeout:
    print("   ❌ Timeout: El servidor tardó demasiado")
    print("      El procesamiento con PyTorch puede ser lento en CPU")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# 4. Limpiar
print("4️⃣  Limpiando archivos de prueba...")
try:
    if os.path.exists(test_video_path):
        os.remove(test_video_path)
    print("   ✅ Archivos limpiados")
except Exception:
    pass

print()
print("=" * 60)
print("✅ PRUEBA COMPLETADA")
print("=" * 60)
print()
print("Si todo funciona aquí pero no desde C++:")
print("  - Verifica la URL en real_time_detector.cpp")
print("  - Debe ser: http://localhost:5000/process_video")
print("  - Verifica que libcurl esté instalado")
print()