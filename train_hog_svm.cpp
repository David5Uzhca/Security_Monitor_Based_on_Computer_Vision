/**
 * train_hog_svm.cpp
 * 
 * Sistema de entrenamiento HOG + SVM para detección de personas
 * Dataset: INRIA Person Dataset
 * 
 * Incluye: Hard Negative Mining para alcanzar ≥80% precisión
 * 
 * Autor: [Tu nombre]
 * Fecha: Enero 2026
 */

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <dirent.h>
#include <random>

using namespace cv;
using namespace cv::ml;
using namespace std;
using namespace chrono;

// ============================================
// CLASE PRINCIPAL DE ENTRENAMIENTO
// ============================================
class PersonDetectorTrainer {
private:
    HOGDescriptor hog;
    Ptr<SVM> svm;
    
    // Parámetros HOG (estándar INRIA)
    Size winSize = Size(64, 128);      // Ventana de detección
    Size blockSize = Size(16, 16);     // Tamaño de bloque
    Size blockStride = Size(8, 8);     // Paso de bloque
    Size cellSize = Size(8, 8);        // Tamaño de celda
    int nbins = 9;                     // Bins de histograma
    
    // Archivo de log
    ofstream logFile;
    
    // Estadísticas
    int totalPositives = 0;
    int totalNegatives = 0;
    
    // Guardar imágenes negativas originales para Hard Negative Mining
    vector<Mat> negativeImagesFullSize;
    
public:
    PersonDetectorTrainer() {
        cout << "\n╔════════════════════════════════════════════════╗\n";
        cout << "║  ENTRENAMIENTO HOG + SVM                       ║\n";
        cout << "║  Dataset: INRIA Person Dataset                 ║\n";
        cout << "║  Con Hard Negative Mining                      ║\n";
        cout << "╚════════════════════════════════════════════════╝\n\n";
        
        // Crear directorios necesarios
        system("mkdir -p logs");
        system("mkdir -p models");
        
        // Abrir archivo de log
        logFile.open("logs/training.log", ios::app);
        log("=== INICIO DE ENTRENAMIENTO ===");
        
        // Inicializar HOG
        log("Inicializando descriptor HOG");
        hog = HOGDescriptor(winSize, blockSize, blockStride, 
                           cellSize, nbins);
        
        cout << "📊 Parámetros HOG configurados:\n";
        cout << "   Ventana: " << winSize << "\n";
        cout << "   Bloque: " << blockSize << "\n";
        cout << "   Paso: " << blockStride << "\n";
        cout << "   Celda: " << cellSize << "\n";
        cout << "   Bins: " << nbins << "\n";
        cout << "   Descriptores por ventana: " << hog.getDescriptorSize() << "\n\n";
        
        // Inicializar SVM
        log("Inicializando SVM");
        svm = SVM::create();
        svm->setType(SVM::C_SVC);
        svm->setKernel(SVM::LINEAR);
        svm->setC(0.01);
        svm->setTermCriteria(
            TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-6)
        );
        
        cout << "🤖 Parámetros SVM configurados:\n";
        cout << "   Tipo: C-SVC\n";
        cout << "   Kernel: LINEAR\n";
        cout << "   C: " << svm->getC() << "\n\n";
    }
    
    ~PersonDetectorTrainer() {
        log("=== FIN DE ENTRENAMIENTO ===");
        logFile.close();
    }
    
    // ============================================
    // FUNCIÓN DE LOGGING
    // ============================================
    void log(const string& message) {
        auto now = system_clock::now();
        auto time = system_clock::to_time_t(now);
        string timeStr = ctime(&time);
        timeStr.pop_back(); // Remover \n
        
        logFile << "[" << timeStr << "] " << message << endl;
        logFile.flush();
    }
    
    // ============================================
    // CARGAR IMÁGENES DE UN DIRECTORIO (png + jpg + bmp)
    // ============================================
    vector<String> loadImagePaths(const string& dirPath) {
        vector<String> allFiles;
        vector<String> temp;
        
        // Buscar todos los formatos comunes
        glob(dirPath + "/*.png", temp);
        allFiles.insert(allFiles.end(), temp.begin(), temp.end());
        
        temp.clear();
        glob(dirPath + "/*.jpg", temp);
        allFiles.insert(allFiles.end(), temp.begin(), temp.end());
        
        temp.clear();
        glob(dirPath + "/*.jpeg", temp);
        allFiles.insert(allFiles.end(), temp.begin(), temp.end());
        
        temp.clear();
        glob(dirPath + "/*.bmp", temp);
        allFiles.insert(allFiles.end(), temp.begin(), temp.end());
        
        // Ordenar para reproducibilidad
        sort(allFiles.begin(), allFiles.end());
        
        return allFiles;
    }
    
    // ============================================
    // EXTRAER CARACTERÍSTICAS HOG
    // ============================================
    vector<float> computeHOG(const Mat& image) {
        vector<float> descriptors;
        Mat resized;
        
        // Redimensionar a tamaño estándar 64x128
        resize(image, resized, winSize);
        
        // Convertir a escala de grises si es necesario
        if (resized.channels() == 3) {
            cvtColor(resized, resized, COLOR_BGR2GRAY);
        }
        
        // Ecualizar histograma para normalizar iluminación
        equalizeHist(resized, resized);
        
        // Calcular descriptores HOG
        try {
            hog.compute(resized, descriptors);
        } catch (const Exception& e) {
            log("ERROR al computar HOG: " + string(e.what()));
            return vector<float>();
        }
        
        return descriptors;
    }
    
    // ============================================
    // CARGA, SPLIT Y PREPARACIÓN DE DATOS
    // ============================================
    bool splitAndLoadDataset(const string& posPath, 
                            const string& negPath,
                            Mat& trainData, Mat& trainLabels,
                            Mat& testData, Mat& testLabels) {
        
        cout << "� PREPARANDO DATASET (Split Automático 80/20)...\n";
        log("Iniciando preparación de dataset con split automático");

        // 1. Obtener listas de archivos
        vector<String> allPosFiles = loadImagePaths(posPath);
        vector<String> allNegFiles = loadImagePaths(negPath);

        if (allPosFiles.empty() || allNegFiles.empty()) {
            cerr << "❌ ERROR: Directorios vacíos o no encontrados.\n";
            log("ERROR: Directorios de datos vacíos");
            return false;
        }

        // 2. Barajar aleatoriamente (Shuffle)
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(allPosFiles.begin(), allPosFiles.end(), std::default_random_engine(seed));
        std::shuffle(allNegFiles.begin(), allNegFiles.end(), std::default_random_engine(seed));

        // 3. Calcular índices de corte (80% Train, 20% Test)
        size_t splitPos = (size_t)(allPosFiles.size() * 0.8);
        size_t splitNeg = (size_t)(allNegFiles.size() * 0.8);

        // Vectores para características
        vector<vector<float>> trainFeatures, testFeatures;
        vector<int> trainLabelsVec, testLabelsVec;

        // ---------------------------------------------------------
        // PROCESAR POSITIVOS (PERSONAS)
        // ---------------------------------------------------------
        cout << "1️⃣  Procesando " << allPosFiles.size() << " imágenes POSITIVAS...\n";
        
        // Train Positives (Con Data Augmentation: Mirroring)
        for (size_t i = 0; i < splitPos; i++) {
            Mat img = imread(allPosFiles[i], IMREAD_COLOR);
            if (img.empty()) continue;

            // Original
            vector<float> feats = computeHOG(img);
            if (!feats.empty()) {
                trainFeatures.push_back(feats);
                trainLabelsVec.push_back(1);
            }

            // Augmentation: Horizontal Flip
            Mat flipped;
            cv::flip(img, flipped, 1); // 1 = horizontal
            vector<float> featsFlip = computeHOG(flipped);
            if (!featsFlip.empty()) {
                trainFeatures.push_back(featsFlip);
                trainLabelsVec.push_back(1);
            }
        }

        // Test Positives (Sin Augmentation - evaluación realista)
        for (size_t i = splitPos; i < allPosFiles.size(); i++) {
            Mat img = imread(allPosFiles[i], IMREAD_COLOR);
            if (img.empty()) continue;

            vector<float> feats = computeHOG(img);
            if (!feats.empty()) {
                testFeatures.push_back(feats);
                testLabelsVec.push_back(1);
            }
        }

        // ---------------------------------------------------------
        // PROCESAR NEGATIVOS (FONDO)
        // ---------------------------------------------------------
        cout << "2️⃣  Procesando " << allNegFiles.size() << " imágenes NEGATIVAS...\n";
        
        int patchesPerImage = 10;

        // Train Negatives
        for (size_t i = 0; i < splitNeg; i++) {
            Mat img = imread(allNegFiles[i], IMREAD_COLOR);
            if (img.empty()) continue;
            if (img.cols < 64 || img.rows < 128) continue;

            // Guardar para Hard Negative Mining (Solo imágenes de entrenamiento)
            negativeImagesFullSize.push_back(img.clone());

            // Extraer parches aleatorios
            for (int k = 0; k < patchesPerImage; k++) {
                int x = rand() % (img.cols - 64);
                int y = rand() % (img.rows - 128);
                Mat patch = img(Rect(x, y, 64, 128));
                
                vector<float> feats = computeHOG(patch);
                if (!feats.empty()) {
                    trainFeatures.push_back(feats);
                    trainLabelsVec.push_back(-1);
                }
            }
        }

        // Test Negatives
        for (size_t i = splitNeg; i < allNegFiles.size(); i++) {
            Mat img = imread(allNegFiles[i], IMREAD_COLOR);
            if (img.empty()) continue;
            if (img.cols < 64 || img.rows < 128) continue;

            // Extraer parches para test
            for (int k = 0; k < patchesPerImage; k++) {
                int x = rand() % (img.cols - 64);
                int y = rand() % (img.rows - 128);
                Mat patch = img(Rect(x, y, 64, 128));
                
                vector<float> feats = computeHOG(patch);
                if (!feats.empty()) {
                    testFeatures.push_back(feats);
                    testLabelsVec.push_back(-1);
                }
            }
        }

        // ---------------------------------------------------------
        // CONVERTIR A MAT
        // ---------------------------------------------------------
        auto convertToMat = [](const vector<vector<float>>& feats, const vector<int>& lbls, Mat& outData, Mat& outLabels) {
            if (feats.empty()) return;
            outData = Mat(feats.size(), feats[0].size(), CV_32F);
            outLabels = Mat(lbls.size(), 1, CV_32S);
            for(size_t i=0; i<feats.size(); i++) {
                for(size_t j=0; j<feats[i].size(); j++) {
                    outData.at<float>(i, j) = feats[i][j];
                }
                outLabels.at<int>(i, 0) = lbls[i];
            }
        };

        convertToMat(trainFeatures, trainLabelsVec, trainData, trainLabels);
        convertToMat(testFeatures, testLabelsVec, testData, testLabels);

        cout << "   ✅ Dataset preparado:\n";
        cout << "      Training: " << trainData.rows << " muestras (" << totalPositives << " pos + aug, " << totalNegatives << " neg patches)\n";
        cout << "      Testing:  " << testData.rows << " muestras\n\n";

        // Actualizar estadísticas globales para el log
        totalPositives = trainLabelsVec.size(); // Aproximado para info
        
        return true;
    }

    // ============================================
    // EXTRAER VECTOR DE PESOS DEL SVM PARA HOG
    // ============================================
    vector<float> getSVMDetector() {
        Mat sv = svm->getSupportVectors();
        Mat alpha, svidx;
        double rho = svm->getDecisionFunction(0, alpha, svidx);
        
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
        detector[sv.cols] = (float)(-rho);
        
        return detector;
    }
    
    // ============================================
    // HARD NEGATIVE MINING
    // ============================================
    bool hardNegativeMining(Mat& trainData, Mat& labels, int round) {
        cout << "\n🔍 HARD NEGATIVE MINING - Ronda " << round << "\n";
        cout << "════════════════════════════════════════════════\n";
        log("Iniciando Hard Negative Mining ronda " + to_string(round));
        
        if (negativeImagesFullSize.empty()) {
            cout << "⚠️  AVISO: No hay imágenes negativas guardadas para HNM (posiblemente dataset pequeño).\n";
            return false;
        }
        
        // Configurar HOG con el SVM actual
        vector<float> detector = getSVMDetector();
        hog.setSVMDetector(detector);
        
        // Buscar falsos positivos en las imágenes negativas
        vector<vector<float>> hardNegFeatures;
        int totalFP = 0;
        
        cout << "   Buscando falsos positivos en " << negativeImagesFullSize.size() 
             << " imágenes negativas de entrenamiento...\n";
        
        for (size_t i = 0; i < negativeImagesFullSize.size(); i++) {
            Mat& negImg = negativeImagesFullSize[i];
            
            vector<Rect> detections;
            vector<double> weights;
            
            // Detectar con parámetros estrictos
            hog.detectMultiScale(negImg, detections, weights,
                0.0,            // hitThreshold
                Size(8, 8),     // winStride
                Size(16, 16),   // padding
                1.05,           // scale
                2.0,            // groupThreshold
                false           // useMeanshiftGrouping
            );
            
            // Cada detección en una imagen negativa es un falso positivo
            for (size_t j = 0; j < detections.size(); j++) {
                Rect& rect = detections[j];
                
                // Verificar que el ROI esté dentro de la imagen
                rect &= Rect(0, 0, negImg.cols, negImg.rows);
                if (rect.width < 10 || rect.height < 10) continue;
                
                Mat patch = negImg(rect);
                Mat resizedPatch;
                resize(patch, resizedPatch, winSize);
                
                vector<float> hogFeatures = computeHOG(resizedPatch);
                if (!hogFeatures.empty()) {
                    hardNegFeatures.push_back(hogFeatures);
                    totalFP++;
                }
            }
            
            // Log CADA imagen para máxima visibilidad
            float progress = ((float)(i + 1) / negativeImagesFullSize.size()) * 100.0f;
            cout << "   Procesadas: " << (i + 1) << "/" << negativeImagesFullSize.size()
                 << " (" << fixed << setprecision(1) << progress << "%) | FP encontrados: " << totalFP << "\r" << flush;
        }
        
        cout << "\n   📊 Falsos positivos encontrados: " << totalFP << "\n";
        log("Falsos positivos encontrados en ronda " + to_string(round) + ": " + to_string(totalFP));
        
        if (totalFP == 0) {
            cout << "   ✅ No se encontraron más falsos positivos. Modelo convergió.\n";
            return true;
        }
        
        // Agregar hard negatives al dataset de entrenamiento
        int oldRows = trainData.rows;
        int numFeatures = trainData.cols;
        int newRows = oldRows + totalFP;
        
        cout << "   Ampliando dataset: " << oldRows << " → " << newRows << " muestras\n";
        
        Mat newTrainData = Mat(newRows, numFeatures, CV_32F);
        Mat newLabels = Mat(newRows, 1, CV_32S);
        
        // Copiar datos existentes
        trainData.copyTo(newTrainData(Rect(0, 0, numFeatures, oldRows)));
        labels.copyTo(newLabels(Rect(0, 0, 1, oldRows)));
        
        // Agregar hard negatives
        for (int i = 0; i < totalFP; i++) {
            for (int j = 0; j < numFeatures; j++) {
                newTrainData.at<float>(oldRows + i, j) = hardNegFeatures[i][j];
            }
            newLabels.at<int>(oldRows + i, 0) = -1;  // Etiqueta negativa
        }
        
        trainData = newTrainData;
        labels = newLabels;
        
        // Re-entrenar SVM
        cout << "   🎓 Re-entrenando SVM con " << newRows << " muestras...\n";
        
        try {
            svm->train(trainData, ROW_SAMPLE, labels);
        } catch (const Exception& e) {
            cerr << "❌ ERROR durante re-entrenamiento: " << e.what() << "\n";
            return false;
        }
        
        return true;
    }
    
    // ============================================
    // ENTRENAR EL SVM
    // ============================================
    bool train(const Mat& trainData, const Mat& labels) {
        cout << "🎓 ENTRENAMIENTO DEL SVM\n";
        cout << "════════════════════════════════════════════════\n";
        cout << "   Muestras de entrenamiento: " << trainData.rows << "\n";
        cout << "   Características: " << trainData.cols << "\n";
        cout << "   Kernel: LINEAR\n";
        cout << "   C: " << svm->getC() << "\n\n";
        
        log("Iniciando entrenamiento SVM");
        
        cout << "⏳ Entrenando... (esto puede tardar varios minutos)\n";
        
        auto startTime = high_resolution_clock::now();
        
        try {
            svm->train(trainData, ROW_SAMPLE, labels);
        } catch (const Exception& e) {
            cerr << "\n❌ ERROR durante el entrenamiento: " << e.what() << "\n";
            return false;
        }
        
        auto endTime = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(endTime - startTime).count();
        
        cout << "\n✅ Entrenamiento inicial completado en " << duration << " segundos\n\n";
        return true;
    }
    
    // ============================================
    // EVALUAR EN CONJUNTO DE PRUEBA
    // ============================================
    void evaluate(const Mat& testData, const Mat& testLabels) {
        cout << "🔍 EVALUACIÓN EN CONJUNTO DE PRUEBA\n";
        cout << "════════════════════════════════════════════════\n";
        
        if (testData.rows == 0) {
            cout << "⚠️  No hay datos de prueba para evaluar.\n";
            return;
        }
        
        Mat predictions;
        svm->predict(testData, predictions);
        
        // Calcular métricas
        int tp = 0, tn = 0, fp = 0, fn = 0;
        
        for (int i = 0; i < testData.rows; i++) {
            float pred = predictions.at<float>(i, 0);
            int truth = testLabels.at<int>(i, 0);
            
            if (pred == 1 && truth == 1) tp++;
            else if (pred == -1 && truth == -1) tn++;
            else if (pred == 1 && truth == -1) fp++;
            else if (pred == -1 && truth == 1) fn++;
        }
        
        // Calcular métricas derivadas
        float accuracy = (float)(tp + tn) / testData.rows * 100.0f;
        float precision = tp > 0 ? (float)tp / (tp + fp) * 100.0f : 0.0f;
        float recall = tp > 0 ? (float)tp / (tp + fn) * 100.0f : 0.0f;
        float f1 = (precision > 0 && recall > 0) ? 
                   2 * (precision * recall) / (precision + recall) : 0.0f;
        
        // Mostrar resultados
        cout << fixed << setprecision(2);
        cout << "\n📈 RESULTADOS (Set de Validación 20%):\n";
        cout << "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        cout << "   Precisión Global (Accuracy): " << accuracy << "%\n";
        cout << "   Precisión Positivos:         " << precision << "%\n";
        cout << "   Sensibilidad (Recall):       " << recall << "%\n";
        cout << "   F1-Score:                    " << f1 << "%\n";
        cout << "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
        
        cout << "📊 MATRIZ DE CONFUSIÓN:\n";
        cout << "   No Persona  " << setw(4) << tn << "  (TN) | " << setw(4) << fp << " (FP)\n";
        cout << "   Persona     " << setw(4) << fn << "  (FN) | " << setw(4) << tp << " (TP)\n\n";
        
        // Guardar métricas en archivo
        ofstream metricsFile("logs/training_metrics.txt");
        metricsFile << "Accuracy: " << accuracy << "%\n";
        metricsFile << "Precision: " << precision << "%\n";
        metricsFile << "Recall: " << recall << "%\n";
        metricsFile << "F1-Score: " << f1 << "%\n";
        metricsFile.close();
    }
    
    // ============================================
    // GUARDAR MODELO
    // ============================================
    void saveModel(const string& filename) {
        cout << "💾 Guardando modelo...\n";
        try {
            svm->save(filename);
            cout << "   ✅ Modelo SVM guardado: " << filename << "\n";
            
            // También exportar el vector detector HOG
            vector<float> detector = getSVMDetector();
            string detectorFile = "models/hog_detector.yml";
            FileStorage fs(detectorFile, FileStorage::WRITE);
            fs << "detector" << detector;
            fs.release();
            cout << "   ✅ Vector detector HOG guardado: " << detectorFile << "\n\n";
            
        } catch (const Exception& e) {
            cerr << "   ❌ ERROR al guardar modelo: " << e.what() << "\n";
        }
    }
};

// ============================================
// FUNCIÓN PRINCIPAL
// ============================================
int main() {
    // Semilla aleatoria
    srand(time(NULL));
    
    PersonDetectorTrainer trainer;
    
    // Rutas del dataset (Estructura unificada)
    string posPath = "data/pos";
    string negPath = "data/neg";
    
    // Verificar si estamos en build/ (intentar ../data)
    if (!opendir(posPath.c_str())) {
        string altPosPath = "../data/pos";
        string altNegPath = "../data/neg";
        if (opendir(altPosPath.c_str())) {
            cout << "⚠️  'data/pos' no encontrado, usando '../data/pos'\n";
            posPath = altPosPath;
            negPath = altNegPath;
        }
    }
    
    // Verificar que existen los directorios
    cout << "🔍 Verificando rutas...\n";
    if (!opendir(posPath.c_str()) || !opendir(negPath.c_str())) {
        cerr << "\n❌ No se encuentran las carpetas 'data/pos' y 'data/neg'.\n";
        cerr << "   Asegúrate de ejecutar prepare_dataset.py primero.\n";
        return 1;
    }
    
    // Variables para datos
    Mat trainData, trainLabels;
    Mat testData, testLabels;

    // 1. CARGA Y SPLIT
    if (!trainer.splitAndLoadDataset(posPath, negPath, trainData, trainLabels, testData, testLabels)) {
        cerr << "❌ Error al cargar datos\n";
        return 1;
    }
    
    // 2. ENTRENAMIENTO INICIAL
    if (!trainer.train(trainData, trainLabels)) {
        cerr << "❌ Error durante el entrenamiento\n";
        return 1;
    }
    
    // 3. HARD NEGATIVE MINING
    // Ejecutar hasta 3 rondas si es necesario
    cout << "\n╔════════════════════════════════════════════════╗\n";
    cout << "║  HARD NEGATIVE MINING                          ║\n";
    cout << "╚════════════════════════════════════════════════╝\n";
    
    for (int round = 1; round <= 3; round++) {
        if (!trainer.hardNegativeMining(trainData, trainLabels, round)) {
            break;
        }
    }
    
    // 4. EVALUACIÓN FINAL
    trainer.evaluate(testData, testLabels);
    
    // 5. GUARDAR
    trainer.saveModel("person_detector_svm.yml");
    
    return 0;
}