cmake_minimum_required(VERSION 3.10)
project(PersonDetector)

# Configuración del compilador
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Opciones de compilación para mejor rendimiento
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
    message(STATUS "Compilando en modo Release con optimizaciones")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall")
    message(STATUS "Compilando en modo Debug")
endif()

# Encontrar OpenCV
find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV encontrado!")
    message(STATUS "  - Versión: ${OpenCV_VERSION}")
    message(STATUS "  - Directorios: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "  - Librerías: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV no encontrado. Instálalo con: sudo apt-get install libopencv-dev")
endif()

include_directories(${OpenCV_INCLUDE_DIRS})

# Encontrar libcurl
find_package(CURL REQUIRED)
if(CURL_FOUND)
    message(STATUS "libcurl encontrado!")
    message(STATUS "  - Versión: ${CURL_VERSION_STRING}")
    message(STATUS "  - Directorios: ${CURL_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "libcurl no encontrado. Instálalo con: sudo apt-get install libcurl4-openssl-dev")
endif()

include_directories(${CURL_INCLUDE_DIR})

# Crear directorio de logs si no existe
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/logs)

# ============================================
# EJECUTABLE 1: Entrenamiento HOG+SVM
# ============================================
add_executable(train_hog_svm train_hog_svm.cpp)

target_link_libraries(train_hog_svm 
    ${OpenCV_LIBS}
)

# Propiedades del ejecutable
set_target_properties(train_hog_svm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "Ejecutable train_hog_svm configurado")

# ============================================
# EJECUTABLE 2: Detección en Tiempo Real
# ============================================
add_executable(real_time_detector real_time_detector.cpp)

target_link_libraries(real_time_detector 
    ${OpenCV_LIBS}
    ${CURL_LIBRARIES}
    pthread  # Para soporte de threads si es necesario
)

# Propiedades del ejecutable
set_target_properties(real_time_detector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

message(STATUS "Ejecutable real_time_detector configurado")

# ============================================
# INFORMACIÓN DE COMPILACIÓN
# ============================================
message(STATUS "")
message(STATUS "==================================================")
message(STATUS "Configuración completa!")
message(STATUS "==================================================")
message(STATUS "Compilador: ${CMAKE_CXX_COMPILER}")
message(STATUS "Flags de compilación: ${CMAKE_CXX_FLAGS}")
message(STATUS "")
message(STATUS "Para compilar:")
message(STATUS "  cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "")
message(STATUS "Ejecutables generados:")
message(STATUS "  - build/train_hog_svm")
message(STATUS "  - build/real_time_detector")
message(STATUS "==================================================")
message(STATUS "")