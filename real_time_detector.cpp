/**
 * real_time_detector.cpp
 * 
 * Detector de personas en tiempo real usando HOG + SVM
 * Integrado con Bot de Telegram para análisis de postura
 * 
 * Características:
 * - Detección HOG+SVM en tiempo real
 * - Captura de video ≥5 segundos
 * - Envío automático a Bot de Telegram
 * - Métricas: FPS, Memoria RSS real, Confianza
 * 
 * Correcciones:
 * - Memoria RSS real del proceso via /proc/self/statm
 * - detectMultiScale con parámetros completos (NMS)
 * - mkdir -p temp antes de guardar video
 */

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <chrono>
#include <curl/curl.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <unistd.h>  // Para sysconf(_SC_PAGESIZE)

using namespace cv;
using namespace std;
using namespace chrono;

class RealtimePersonDetector {
private:
    VideoCapture camera;
    HOGDescriptor hog;
    Ptr<ml::SVM> svm;

    vector<Mat> videoBuffer;
    const int bufferMaxFrames = 150;  // 5 segundos a 30 FPS
    const int bufferMinFrames = 150;  // Mínimo 5 segundos (requisito del proyecto)

    double fps = 0.0;
    double memoryUsage = 0.0;

    // Log file
    ofstream logFile;

public:
    RealtimePersonDetector() {
        // Crear directorios necesarios
        system("mkdir -p logs");
        system("mkdir -p temp");
        
        // Abrir archivo de log
        logFile.open("logs/detector.log", ios::app);
        log("=== INICIANDO DETECTOR ===");
        
        // Abrir cámara
        camera.open(0);
        if (!camera.isOpened()) {
            log("ERROR: No se pudo abrir la cámara");
            cerr << "❌ No se pudo abrir la cámara\n";
            exit(1);
        }
        log("Cámara abierta correctamente");

        // Configurar resolución
        camera.set(CAP_PROP_FRAME_WIDTH, 640);
        camera.set(CAP_PROP_FRAME_HEIGHT, 480);
        camera.set(CAP_PROP_FPS, 30);

        // Cargar SVM
        log("Cargando modelo SVM...");
        svm = ml::SVM::load("person_detector_svm.yml");
        if (svm.empty() || !svm->isTrained()) {
            log("ERROR: SVM no cargado o no entrenado");
            cerr << "❌ Error: SVM no cargado o no entrenado\n";
            cerr << "   Ejecuta primero: ./build/train_hog_svm\n";
            exit(1);
        }
        log("SVM cargado correctamente");

        // Configurar HOG
        hog = HOGDescriptor(
            Size(64, 128),  // winSize
            Size(16, 16),   // blockSize
            Size(8, 8),     // blockStride
            Size(8, 8),     // cellSize
            9               // nbins
        );

        // Conversión SVM → HOG (para kernel LINEAR)
        log("Configurando detector HOG...");
        try {
            // Intentar cargar vector detector exportado directamente
            bool loadedFromFile = false;
            string detectorFile = "models/hog_detector.yml";
            
            ifstream checkFile(detectorFile);
            if (checkFile.good()) {
                checkFile.close();
                FileStorage fs(detectorFile, FileStorage::READ);
                if (fs.isOpened()) {
                    vector<float> detector;
                    fs["detector"] >> detector;
                    fs.release();
                    
                    size_t expectedSize = hog.getDescriptorSize() + 1;
                    if (detector.size() == expectedSize) {
                        hog.setSVMDetector(detector);
                        loadedFromFile = true;
                        log("Detector cargado desde archivo: " + detectorFile);
                        cout << "✅ Detector HOG cargado desde " << detectorFile << "\n";
                    }
                }
            }
            
            // Si no se pudo cargar del archivo, reconstruir desde el SVM
            if (!loadedFromFile) {
                log("Reconstruyendo detector desde modelo SVM...");
                
                Mat sv = svm->getSupportVectors();
                Mat alpha, svidx;
                double rho = svm->getDecisionFunction(0, alpha, svidx);

                log("Support Vectors: " + to_string(sv.rows) + " x " + to_string(sv.cols));
                log("Rho: " + to_string(rho));

                // Calcular w = sum(alpha_i * sv_i)
                Mat w = Mat::zeros(1, sv.cols, CV_32F);
                for (int i = 0; i < sv.rows; i++) {
                    float alphaVal = (float)alpha.at<double>(i);
                    for (int j = 0; j < sv.cols; j++) {
                        w.at<float>(0, j) += alphaVal * sv.at<float>(i, j);
                    }
                }

                // Crear detector: [w, -rho]
                vector<float> detector;
                detector.resize(sv.cols + 1);
                for (int i = 0; i < sv.cols; i++) {
                    detector[i] = w.at<float>(0, i);
                }
                detector[sv.cols] = (float)-rho;

                // Verificar dimensiones
                size_t expectedSize = hog.getDescriptorSize() + 1;
                if (detector.size() != expectedSize) {
                    log("ERROR: Dimensiones incorrectas del detector");
                    cerr << "❌ Error dimensional:\n";
                    cerr << "   HOG espera: " << expectedSize << "\n";
                    cerr << "   Detector tiene: " << detector.size() << "\n";
                    cerr << "\n   Sugerencia: Reentrena el modelo con:\n";
                    cerr << "   ./build/train_hog_svm\n";
                    exit(1);
                }

                hog.setSVMDetector(detector);
                log("Detector HOG reconstruido desde SVM");
                cout << "✅ Detector HOG+SVM reconstruido correctamente\n";
            }
            
            cout << "   Dimensiones: " << hog.getDescriptorSize() + 1 << " features\n";

        } catch (const Exception& e) {
            log(string("ERROR al configurar detector: ") + e.what());
            cerr << "❌ Error al configurar detector: " << e.what() << endl;
            cerr << "\n   Solución:\n";
            cerr << "   1. Elimina el modelo anterior: rm person_detector_svm.yml\n";
            cerr << "   2. Reentrena: ./build/train_hog_svm\n";
            exit(1);
        }
        
        log("Inicialización completada");
    }

    ~RealtimePersonDetector() {
        log("=== DETECTOR DETENIDO ===");
        logFile.close();
    }

    void log(const string& message) {
        auto now = system_clock::now();
        auto time = system_clock::to_time_t(now);
        string timeStr = ctime(&time);
        timeStr.pop_back();
        
        logFile << "[" << timeStr << "] " << message << endl;
        logFile.flush();
    }

    void updateFPS(time_point<high_resolution_clock>& lastTime) {
        auto now = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(now - lastTime).count();
        if (duration > 0) {
            fps = 1000.0 / duration;
        }
        lastTime = now;
    }

    // ============================================
    // MEDICIÓN DE MEMORIA RSS REAL DEL PROCESO
    // ============================================
    void updateMemoryUsage() {
        // Leer memoria RSS real del proceso desde /proc/self/statm (Linux)
        ifstream statm("/proc/self/statm");
        if (statm.is_open()) {
            long size, resident, shared, text, lib, data, dt;
            statm >> size >> resident >> shared >> text >> lib >> data >> dt;
            statm.close();
            
            // resident está en páginas, convertir a MB
            long pageSize = sysconf(_SC_PAGESIZE);
            memoryUsage = (resident * pageSize) / (1024.0 * 1024.0);
        } else {
            // Fallback: estimar desde el buffer si /proc no está disponible
            if (!videoBuffer.empty()) {
                memoryUsage = videoBuffer.size() *
                    videoBuffer[0].total() *
                    videoBuffer[0].elemSize() / (1024.0 * 1024.0);
            } else {
                memoryUsage = 0.0;
            }
        }
    }

    bool sendVideoToTelegram(const string& url) {
        if (videoBuffer.empty()) {
            log("ERROR: Buffer vacío, no hay video para enviar");
            return false;
        }

        log("Preparando video para envío...");
        cout << "\n📤 Enviando video al bot...\n";
        
        string path = "temp/temp_detection.mp4";
        Size sz(videoBuffer[0].cols, videoBuffer[0].rows);
        
        log("Creando archivo de video: " + path);
        VideoWriter writer(path,
            VideoWriter::fourcc('m','p','4','v'), 30, sz);

        if (!writer.isOpened()) {
            log("ERROR: No se pudo crear el archivo de video");
            cerr << "❌ Error al crear video\n";
            return false;
        }

        log("Escribiendo " + to_string(videoBuffer.size()) + " frames...");
        for (auto& f : videoBuffer) {
            writer.write(f);
        }
        writer.release();
        
        // Verificar que el archivo existe
        ifstream checkFile(path);
        if (!checkFile.good()) {
            log("ERROR: El archivo de video no existe después de crearlo");
            return false;
        }
        checkFile.close();
        
        // Obtener tamaño del archivo
        ifstream file(path, ios::binary | ios::ate);
        streamsize size = file.tellg();
        file.close();
        log("Archivo creado: " + to_string(size / 1024) + " KB");

        // Enviar con libcurl
        log("Iniciando envío HTTP POST a: " + url);
        CURL* curl = curl_easy_init();
        if (!curl) {
            log("ERROR: No se pudo inicializar CURL");
            cerr << "❌ Error al inicializar CURL\n";
            return false;
        }

        curl_mime* form = curl_mime_init(curl);
        curl_mimepart* part = curl_mime_addpart(form);
        curl_mime_name(part, "video");
        curl_mime_filedata(part, path.c_str());

        // Buffer para respuesta
        string responseBuffer;
        
        // Función para capturar respuesta
        auto writeCallback = +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
            ((string*)userdata)->append(ptr, size * nmemb);
            return size * nmemb;
        };

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBuffer);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);  // 2 minutos timeout

        log("Ejecutando petición HTTP...");
        CURLcode res = curl_easy_perform(curl);

        bool success = false;
        if (res == CURLE_OK) {
            long response_code;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
            
            log("Respuesta HTTP: " + to_string(response_code));
            log("Cuerpo de respuesta: " + responseBuffer);
            
            if (response_code == 200) {
                log("Video enviado exitosamente");
                cout << "✅ Video enviado exitosamente!\n";
                success = true;
            } else {
                log("ERROR: Código HTTP " + to_string(response_code));
                cerr << "❌ Error del servidor: " << response_code << "\n";
                cerr << "   Respuesta: " << responseBuffer << "\n";
            }
        } else {
            string error = curl_easy_strerror(res);
            log("ERROR CURL: " + error);
            cerr << "❌ Error CURL: " << error << "\n";
            cerr << "   ¿Está corriendo el bot de Telegram?\n";
            cerr << "   Ejecuta: python telegram_pose_bot.py\n";
        }

        curl_mime_free(form);
        curl_easy_cleanup(curl);
        
        // Limpiar archivo temporal
        remove(path.c_str());

        return success;
    }

    void run() {
        Mat frame;
        auto lastTime = high_resolution_clock::now();
        bool recording = false;
        int noDetectFrames = 0;

        cout << "\n╔════════════════════════════════════════════════╗\n";
        cout << "║  🎥 DETECCIÓN EN TIEMPO REAL                   ║\n";
        cout << "╠════════════════════════════════════════════════╣\n";
        cout << "║  Controles:                                    ║\n";
        cout << "║    q - Salir                                   ║\n";
        cout << "║    s - Enviar video manualmente                ║\n";
        cout << "║                                                ║\n";
        cout << "║  El video se envía automáticamente después de  ║\n";
        cout << "║  1 segundo sin detectar personas               ║\n";
        cout << "║                                                ║\n";
        cout << "║  Requisito: Mínimo 5 segundos de grabación     ║\n";
        cout << "╚════════════════════════════════════════════════╝\n\n";

        log("Bucle principal iniciado");

        while (true) {
            camera >> frame;
            if (frame.empty()) {
                log("WARNING: Frame vacío recibido");
                continue;
            }

            vector<Rect> detections;
            vector<double> weights;

            // ============================================
            // OPTIMIZACIÓN: Downscaling para aumentar FPS
            // ============================================
            // Reducir la imagen para detección (ej. ancho 320 o 400)
            // HOG es MUY lento en 640x480. 
            double scale = 1.0;
            double targetWidth = 320.0; // Ancho reducido para detección rápida
            Mat smallFrame;
            
            if (frame.cols > targetWidth) {
                scale = frame.cols / targetWidth;
                Size smallSize(targetWidth, frame.rows / scale);
                resize(frame, smallFrame, smallSize);
            } else {
                smallFrame = frame;
            }

            // Detección multiscala en imagen reducida
            hog.detectMultiScale(smallFrame, detections, weights,
                0.0,            // hitThreshold
                Size(8, 8),     // winStride
                Size(16, 16),   // padding
                1.05,           // scale (un poco mayor para menos pirámides)
                2.0,            // groupThreshold
                false           // useMeanshiftGrouping
            );

            // Re-escalar las detecciones al tamaño original
            if (scale > 1.0) {
                for (size_t i = 0; i < detections.size(); i++) {
                    detections[i].x = cvRound(detections[i].x * scale);
                    detections[i].y = cvRound(detections[i].y * scale);
                    detections[i].width = cvRound(detections[i].width * scale);
                    detections[i].height = cvRound(detections[i].height * scale);
                }
            }

            updateFPS(lastTime);
            updateMemoryUsage();

            // Dibujar bounding boxes en el frame ORIGINAL para que salgan en el video
            for (size_t i = 0; i < detections.size(); i++) {
                // Color del bounding box según confianza
                Scalar color;
                if (weights[i] > 1.0) color = Scalar(0, 255, 0);      // Verde: alta confianza
                else if (weights[i] > 0.5) color = Scalar(0, 255, 255); // Amarillo: media
                else color = Scalar(0, 165, 255);                       // Naranja: baja
                
                rectangle(frame, detections[i], color, 2);
                
                // Mostrar confianza
                string conf = "Conf: " + to_string(weights[i]).substr(0, 4);
                putText(frame, conf, 
                       Point(detections[i].x, detections[i].y - 5),
                       FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }

            // Crear imagen para mostrar en pantalla (abierta localmente)
            // Se clona 'frame' que YA TIENE los bounding boxes
            Mat display = frame.clone();

            // Dibujar métricas SOLO en la pantalla local (no en el video grabado)
            // Fondo semi-transparente para las métricas
            rectangle(display, Rect(0, 0, 300, 200), Scalar(0, 0, 0), FILLED);
            rectangle(display, Rect(0, 0, 300, 200), Scalar(0, 0, 0), 1);
            
            putText(display, "FPS: " + to_string((int)fps),
                    Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.6,
                    Scalar(0, 255, 255), 2);
            
            // Memoria RSS real del proceso
            char memStr[64];
            snprintf(memStr, sizeof(memStr), "Memoria RSS: %.1f MB", memoryUsage);
            putText(display, memStr,
                    Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.6,
                    Scalar(0, 255, 255), 2);
            
            putText(display, "Personas: " + to_string(detections.size()),
                    Point(10, 75), FONT_HERSHEY_SIMPLEX, 0.6,
                    Scalar(0, 255, 255), 2);
            
            // Confianza promedio
            if (!weights.empty()) {
                double avgConfidence = 0;
                for (auto w : weights) avgConfidence += w;
                avgConfidence /= weights.size();
                
                char confStr[64];
                snprintf(confStr, sizeof(confStr), "Confianza: %.2f", avgConfidence);
                putText(display, confStr,
                       Point(10, 100), FONT_HERSHEY_SIMPLEX, 0.6,
                       Scalar(0, 255, 255), 2);
            }
            
            // Mostrar frames en buffer
            putText(display, "Buffer: " + to_string(videoBuffer.size()) + "/150 frames",
                   Point(10, 125), FONT_HERSHEY_SIMPLEX, 0.6,
                   Scalar(0, 255, 255), 2);
            
            // Duración actual del video
            if (!videoBuffer.empty()) {
                double duration = videoBuffer.size() / 30.0;
                char durStr[64];
                snprintf(durStr, sizeof(durStr), "Duracion: %.1f seg", duration);
                putText(display, durStr,
                       Point(10, 150), FONT_HERSHEY_SIMPLEX, 0.6,
                       Scalar(0, 255, 255), 2);
            }

            // Indicador de grabación
            if (recording) {
                putText(display, "REC",
                       Point(display.cols - 100, 30), FONT_HERSHEY_SIMPLEX, 1.0,
                       Scalar(0, 0, 255), 3);
                circle(display, Point(display.cols - 120, 23), 8, Scalar(0, 0, 255), -1);
            }

            // Gestión del buffer de video
            if (recording) {
                // Si ya estamos grabando, solo agregar frames hasta llegar al límite
                videoBuffer.push_back(frame.clone());
                
                // Mostrar progreso de grabación
                int framesRecorded = videoBuffer.size();
                int progress = (framesRecorded * 100) / bufferMaxFrames;
                
                // Barra de progreso visual
                rectangle(display, Point(0, display.rows-10), 
                         Point((display.cols * progress)/100, display.rows), 
                         Scalar(0, 0, 255), FILLED);

                if (framesRecorded >= bufferMaxFrames) {
                    log("Grabación completada (5s) - Enviando video...");
                    cout << "\n✅ Grabación completada (" << framesRecorded << " frames). Enviando...\n";
                    
                    bool sent = sendVideoToTelegram("http://localhost:5000/process_video");
                    
                    if (sent) {
                        cout << "✅ Envío exitoso\n\n";
                    } else {
                        cout << "❌ Falló el envío\n\n";
                    }
                    
                    videoBuffer.clear();
                    recording = false;
                    noDetectFrames = 0;
                    cout << "👀 Buscando personas...\n";
                }
            }
            else if (!detections.empty()) {
                // Si NO estamos grabando y detectamos a alguien -> INICIAR GRABACIÓN
                log("Persona detectada - Iniciando grabación de 5 segundos");
                cout << "👤 Persona detectada - Grabando 5 segundos...\n";
                
                recording = true;
                videoBuffer.clear();
                videoBuffer.push_back(frame.clone());
            }

            imshow("HOG + SVM Detector", display);

            char k = waitKey(1);
            if (k == 'q') {
                log("Usuario presionó 'q' - Saliendo");
                break;
            }
            if (k == 's' && !videoBuffer.empty()) {
                if ((int)videoBuffer.size() >= bufferMinFrames) {
                    log("Usuario presionó 's' - Enviando manual");
                    cout << "\n📹 Enviando video manualmente (" 
                         << (videoBuffer.size() / 30.0) << " seg)...\n";
                    sendVideoToTelegram("http://localhost:5000/process_video");
                    videoBuffer.clear();
                    recording = false;
                } else {
                    cout << "⚠️  Video muy corto (" << videoBuffer.size() 
                         << " frames). Necesita ≥150 frames (5 seg)\n";
                }
            }
        }

        camera.release();
        destroyAllWindows();
        log("Cámara cerrada - Programa terminado");
    }
};

int main() {
    try {
        RealtimePersonDetector app;
        app.run();
    } catch (const exception& e) {
        cerr << "❌ Error fatal: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}