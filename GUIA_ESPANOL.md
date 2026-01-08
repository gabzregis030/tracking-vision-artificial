# 🎯 Guía del Proyecto Vision Tracking (Español)

## 📖 ¿Qué es este proyecto?

Este es un **sistema de seguimiento de objetos en tiempo real** que utiliza visión computacional con OpenCV y Python.  Te permite rastrear uno o varios objetos en videos o en tiempo real usando tu cámara web.

## 🎬 ¿Cómo funciona?

El sistema funciona en 3 pasos principales:

### 1. **Selección de Objetos**
Al iniciar el programa, se abre una ventana con el primer frame del video. Tú dibujas un rectángulo alrededor del objeto que quieres rastrear usando el mouse.

### 2. **Tracking (Seguimiento)**
El algoritmo de tracking sigue el objeto frame por frame, prediciendo su posición en cada nuevo fotograma.

### 3. **Visualización**
El sistema muestra en tiempo real:
- Rectángulo alrededor del objeto rastreado
- ID del objeto (si son varios)
- Velocidad estimada (opcional)
- FPS (frames por segundo)

## 🏗️ Arquitectura del Proyecto

El proyecto sigue una arquitectura **MVC (Modelo-Vista-Controlador)**:

```
src/
├── models/                  # Lógica de negocio
│   ├── detector.py          # Detección automática de objetos (YOLO, HOG)
│   ├── tracker.py           # Algoritmos de tracking
│   ├── kalman_filter.py     # Filtrado Kalman para suavizar trayectorias
│   ├── speed_calculator.py  # Cálculo de velocidad
│   └── video_processor.py   # Lectura/escritura de videos
│
├── controllers/             # Controladores de la aplicación
│   └── app_controller.py    # Orquesta todo el flujo
│
├── utils/                   # Utilidades
│   ├── config.py            # Configuración
│   └── visualizer.py        # Visualización de resultados
│
└── main.py                  # Punto de entrada CLI
```

### Componentes Principales

#### 🎯 Tracker (models/tracker.py)
Implementa múltiples algoritmos de tracking:

| Algoritmo | Velocidad | Precisión | Mejor para |
|-----------|-----------|-----------|------------|
| **CSRT** | Lenta | Alta | Objetos con cambios de escala |
| **KCF** | Media | Media-Alta | Uso general, buen balance |
| **MOSSE** | Muy Rápida | Media | Tiempo real, recursos limitados |
| **MedianFlow** | Rápida | Media | Movimientos predecibles |

#### 🔍 Detector (models/detector.py)
Detecta objetos automáticamente usando:
- **YOLO**: Deep learning, muy preciso
- **HOG**: Más rápido, menos recursos

#### 📊 Kalman Filter (models/kalman_filter.py)
Suaviza las trayectorias prediciendo la posición del objeto y corrigiendo ruido.

#### 🎥 Video Processor (models/video_processor.py)
Maneja entrada/salida de video desde archivos o cámara.

## 🚀 Instalación

### Requisitos Previos
- Python 3.8 o superior
- **Recomendado**: `uv` (gestor ultra-rápido de paquetes Python)

### Métodos de Instalación

#### Método 1: Usando UV (⚡ Recomendado - Más Rápido)

No necesitas instalar nada! Solo usa `uv run`:

```bash
cd /Users/gregis/.gemini/antigravity/scratch/vision-tracking-project

# Ejecutar directamente sin instalar dependencias
uv run src/main.py --video videos/demo_two_objects.mp4 --multi --num-objects 2 --tracker KCF
```

`uv` automáticamente crea un entorno virtual aislado y descarga todas las dependencias necesarias. Es **mucho más rápido** que pip.

#### Método 2: Instalación tradicional con pip

1. **Navega al directorio**:
```bash
cd /Users/gregis/.gemini/antigravity/scratch/vision-tracking-project
```

2. **Instala las dependencias**:
```bash
pip install -r requirements.txt
```

Esto instalará:
- `opencv-contrib-python`: Biblioteca de visión computacional
- `numpy`: Cálculos numéricos
- `matplotlib`: Visualización
- `pandas`: Análisis de datos
- `pytest`: Testing
- `jupyter`: Notebooks para experimentación

## 💻 Uso del Sistema

### Opción 1: Rastrear UN objeto en un video

**Con UV (recomendado):**
```bash
uv run src/main.py --video videos/mi_video.mp4 --tracker CSRT
```

**Con Python tradicional:**
```bash
python src/main.py --video videos/mi_video.mp4 --tracker CSRT
```

**¿Qué hace?**
1. Abre el primer frame del video
2. Tú dibujas un rectángulo con el mouse alrededor del objeto
3. Presiona ENTER o ESPACIO
4. El sistema rastrea el objeto automáticamente
5. Presiona 'q' para salir

### Opción 2: Rastrear MÚLTIPLES objetos (como 2 objetos) 🔥

**Con UV (recomendado):**
```bash
uv run src/main.py --video videos/mi_video.mp4 --multi --num-objects 2 --tracker KCF
```

**Con Python tradicional:**
```bash
python src/main.py --video videos/mi_video.mp4 --multi --num-objects 2 --tracker KCF
```

**¿Qué hace?**
1. Abre el primer frame
2. Dibujas rectángulo alrededor del PRIMER objeto → presiona ENTER
3. Dibujas rectángulo alrededor del SEGUNDO objeto → presiona ENTER
4. El sistema rastrea AMBOS objetos simultáneamente
5. Cada objeto tiene un color y ID diferente

### Opción 3: Usar tu webcam en tiempo real

**Con UV (recomendado):**
```bash
uv run src/main.py --camera 0 --tracker MOSSE
```

**Con Python tradicional:**
```bash
python src/main.py --camera 0 --tracker MOSSE
```

**¿Qué hace?**
1. Abre tu cámara web
2. Dibuja rectángulo alrededor del objeto que quieres seguir
3. Presiona ENTER
4. El tracking comienza en tiempo real

### Opción 4: Con filtro Kalman (más suave)

**Con UV (recomendado):**
```bash
uv run src/main.py --video videos/mi_video.mp4 --multi --num-objects 2 --tracker KCF --kalman
```

**Con Python tradicional:**
```bash
python src/main.py --video videos/mi_video.mp4 --multi --num-objects 2 --tracker KCF --kalman
```

El filtro Kalman hace el seguimiento más suave y predice mejor movimientos.

## 🎮 Controles Interactivos

Cuando el programa está ejecutándose:

- **Mouse**: Dibuja rectángulos para seleccionar objetos
- **ENTER/ESPACIO**: Confirma selección
- **'q'**: Salir del programa
- **ESC**: Cancelar selección actual

## 📂 Estructura de Archivos

```
vision-tracking-project/
├── src/                     # Código fuente
├── examples/                # Scripts de ejemplo
│   ├── single_object.py     # Ejemplo de 1 objeto
│   ├── multi_object.py      # Ejemplo de múltiples objetos
│   └── webcam_tracking.py   # Ejemplo con webcam
├── videos/                  # Tus videos de prueba (agrégalos aquí)
├── results/                 # Resultados generados
├── docs/                    # Documentación adicional
├── notebooks/               # Jupyter notebooks
├── tests/                   # Tests automatizados
├── requirements.txt         # Dependencias
└── README.md                # README en inglés
```

## 🧪 Ejemplo Práctico: Rastrear 2 Objetos

### Paso a Paso

1. **Coloca tu video** en la carpeta `videos/`:
```bash
# Por ejemplo, copia tu video aquí
cp ~/Desktop/mi_video.mp4 videos/test.mp4
```

2. **Ejecuta el tracking de 2 objetos**:
```bash
python src/main.py --video videos/test.mp4 --multi --num-objects 2 --tracker KCF --kalman
```

3. **Interacción**:
   - Se abre una ventana con el primer frame
   - Dibuja un rectángulo alrededor del PRIMER objeto
   - Presiona ENTER
   - Dibuja un rectángulo alrededor del SEGUNDO objeto  
   - Presiona ENTER
   - ¡El tracking comienza!

4. **Observa**:
   - Cada objeto tiene un rectángulo de diferente color
   - Cada uno tiene su ID (Object 1, Object 2)
   - Puedes ver la velocidad y FPS

## 🎨 Interfaz Gráfica

El proyecto **NO tiene interfaz web**, es una **aplicación de línea de comandos (CLI)** que:
- Abre ventanas con OpenCV para mostrar el video
- Permite interacción con mouse para seleccionar objetos
- Muestra el tracking en tiempo real en ventanas emergentes

**Ventanas que verás:**
1. **Ventana de selección**: Para dibujar rectángulos
2. **Ventana de tracking**: Muestra el seguimiento en tiempo real

## ⚙️ Configuración Avanzada

### Usar diferentes trackers según el caso

```bash
# Para objetos rápidos
python src/main.py --video videos/test.mp4 --tracker MOSSE

# Para máxima precisión (más lento)
python src/main.py --video videos/test.mp4 --tracker CSRT --kalman

# Balance entre velocidad y precisión
python src/main.py --video videos/test.mp4 --tracker KCF
```

## 🔧 Solución de Problemas

### Problema: "ModuleNotFoundError: No module named 'cv2'"
**Solución**: Instala OpenCV:
```bash
pip install opencv-contrib-python
```

### Problema: "Video file not found"
**Solución**: Verifica que la ruta al video sea correcta:
```bash
ls videos/  # Ver qué videos tienes
```

### Problema: El tracking se pierde
**Soluciones**:
1. Usa `--kalman` para mejor predicción
2. Prueba con `CSRT` para mayor precisión
3. Asegúrate de que el objeto inicial esté bien enmarcado

### Problema: Muy lento
**Soluciones**:
1. Usa `--tracker MOSSE` (el más rápido)
2. Reduce el número de objetos rastreados
3. Usa un video de menor resolución

## 📊 Salida del Sistema

El sistema genera:
- **Visualización en tiempo real**: Ventana con el tracking
- **Estadísticas**: FPS, velocidad de objetos
- **Resultados** (opcional): Videos procesados en `results/`

## 🎓 Ejemplos Adicionales

### Ejecutar scripts de ejemplo interactivos

```bash
# Ejemplo de múltiples objetos (interactivo)
python examples/multi_object.py

# Ejemplo de un solo objeto (interactivo)
python examples/single_object.py

# Ejemplo con webcam (interactivo)
python examples/webcam_tracking.py
```

Estos scripts te harán preguntas interactivas para configurar el tracking.

## 📚 Referencias

- **OpenCV Tracking API**: https://docs.opencv.org/4.x/d9/df8/group__tracking.html
- **Algoritmo CSRT**: https://arxiv.org/abs/1611.08461
- **Algoritmo KCF**: https://arxiv.org/abs/1404.7584

## ✨ Resumen Rápido

**Para rastrear 2 objetos en un video:**

```bash
# Método 1: Con UV (⚡ Recomendado - Sin instalación)
uv run src/main.py --video videos/demo_two_objects.mp4 --multi --num-objects 2 --tracker KCF --kalman

# Método 2: Python tradicional (requiere: pip install -r requirements.txt)
python src/main.py --video videos/TU_VIDEO.mp4 --multi --num-objects 2 --tracker KCF --kalman
```

**Pasos:**
1. El sistema abre el primer frame del video
2. Dibuja rectángulo alrededor del objeto 1 → ENTER
3. Dibuja rectángulo alrededor del objeto 2 → ENTER
4. ¡El tracking comienza automáticamente!
5. Presiona 'q' para salir

**Video de demostración incluido**: `videos/demo_two_objects.mp4` tiene 2 objetos en movimiento listos para probar.

---

**¿Necesitas ayuda?** Revisa los ejemplos en la carpeta `examples/` o consulta el README.md principal.

¡Feliz tracking! 🎯
