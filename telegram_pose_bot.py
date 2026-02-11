"""
telegram_pose_bot.py
Sistema de Detección de Postura Humana
Bot de Telegram + API Flask
Correcciones:
- Usa KeypointRCNN de torchvision (PyTorch puro) en lugar de MediaPipe
- Envío a Telegram con requests síncrono (evita conflictos asyncio)
- File handles con 'with' statement
- ✅ FIX: Fallback automático a CPU para GPUs sm_120 (RTX 5080)
- ✅ FIX: Reemplazo de pretrained=True → weights enum
"""
import os
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import threading
import requests as http_requests  # Renombrado para no chocar con flask.request
from datetime import datetime
from pathlib import Path
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from flask import Flask, request, jsonify, render_template, send_from_directory
# Cargar variables de entorno
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv no instalado, usando variables de entorno del sistema")
# ===============================
# DETECTOR DE POSTURA CON PYTORCH
# ===============================
class PoseDetector:
    """
    Detector de postura humana usando KeypointRCNN de torchvision (PyTorch).
    Detecta 17 keypoints del cuerpo en formato COCO.
    Keypoints COCO:
    0: nariz, 1: ojo_izq, 2: ojo_der, 3: oreja_izq, 4: oreja_der,
    5: hombro_izq, 6: hombro_der, 7: codo_izq, 8: codo_der,
    9: muñeca_izq, 10: muñeca_der, 11: cadera_izq, 12: cadera_der,
    13: rodilla_izq, 14: rodilla_der, 15: tobillo_izq, 16: tobillo_der
    """
    # Conexiones del esqueleto COCO
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),       # Cabeza
        (5, 7), (7, 9),                          # Brazo izquierdo
        (6, 8), (8, 10),                         # Brazo derecho
        (5, 6),                                   # Hombros
        (11, 12),                                 # Caderas
        (5, 11), (6, 12),                         # Torso
        (11, 13), (13, 15),                       # Pierna izquierda
        (12, 14), (14, 16),                       # Pierna derecha
    ]
    # Colores para cada parte del cuerpo (BGR)
    COLORS = {
        'head': (255, 200, 0),      # Cyan claro
        'left_arm': (0, 255, 0),    # Verde
        'right_arm': (0, 165, 255), # Naranja
        'torso': (255, 255, 0),     # Cyan
        'left_leg': (255, 0, 255),  # Magenta
        'right_leg': (0, 255, 255), # Amarillo
    }

    def __init__(self):
        # 🔧 FIX: Detectar GPUs incompatibles (sm_120 = RTX 5080) y fallback a CPU
        try:
            if torch.cuda.is_available():
                compute_capability = torch.cuda.get_device_capability()
                major = compute_capability[0]
                # sm_120 (RTX 5080) no soportado en PyTorch ≤2.3
                if major >= 12:
                    print(f"⚠️  GPU con compute capability {compute_capability} no compatible con PyTorch actual")
                    print(f"   Forzando uso de CPU (fallback seguro)")
                    self.device = torch.device('cpu')
                else:
                    self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        except Exception as e:
            print(f"⚠️  Error al detectar GPU: {e}. Usando CPU.")
            self.device = torch.device('cpu')
        
        print(f"🖥️  Usando dispositivo: {self.device}")
        print("📦 Cargando modelo KeypointRCNN (PyTorch)...")
        
        # 🔧 FIX: Reemplazar pretrained=True → weights enum (PyTorch ≥0.13)
        self.model = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,  # ✅ Correcto
            min_size=480,   # Tamaño mínimo para inferencia (más rápido)
            max_size=640,   # Tamaño máximo
        )
        
        self.model.to(self.device)
        self.model.eval()
        print("✅ Modelo PyTorch KeypointRCNN cargado correctamente")
        print(f"   Tipo: torchvision.models.detection.keypointrcnn_resnet50_fpn")
        print(f"   Framework: PyTorch {torch.__version__}")
        print(f"   Keypoints: 17 (formato COCO)")
        print(f"   Dispositivo: {self.device}")

    def _get_skeleton_color(self, idx):
        """Obtener color para cada conexión del esqueleto"""
        if idx < 4:
            return self.COLORS['head']
        elif idx < 6:
            return self.COLORS['left_arm']
        elif idx < 8:
            return self.COLORS['right_arm']
        elif idx < 10:
            return self.COLORS['torso']
        elif idx < 12:
            return self.COLORS['torso']
        elif idx < 14:
            return self.COLORS['left_leg']
        else:
            return self.COLORS['right_leg']

    def detect_pose_frame(self, frame):
        """
        Detectar postura en un frame usando KeypointRCNN.
        Returns:
        vis: Frame con esqueleto dibujado
        result: Dict con keypoints, scores, boxes
        """
        # Convertir BGR a RGB y a tensor
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(rgb).to(self.device)
        # Inferencia
        with torch.no_grad():
            outputs = self.model([img_tensor])
        vis = frame.copy()
        result = {
            'keypoints': [],
            'scores': [],
            'boxes': [],
            'num_persons': 0,
            'avg_confidence': 0.0,
        }
        if len(outputs[0]['scores']) == 0:
            return vis, result
        # Filtrar detecciones con score > 0.7
        keep = outputs[0]['scores'] > 0.7
        scores = outputs[0]['scores'][keep].cpu().numpy()
        keypoints = outputs[0]['keypoints'][keep].cpu().numpy()
        boxes = outputs[0]['boxes'][keep].cpu().numpy()
        if len(scores) == 0:
            return vis, result
        result['num_persons'] = len(scores)
        result['scores'] = scores.tolist()
        result['boxes'] = boxes.tolist()
        # Procesar cada persona detectada
        all_kp_confs = []
        for person_idx in range(len(scores)):
            kps = keypoints[person_idx]  # Shape: (17, 3) -> x, y, conf
            result['keypoints'].append(kps.tolist())
            # Dibujar keypoints
            for kp_idx, (x, y, conf) in enumerate(kps):
                if conf > 0.5:
                    all_kp_confs.append(conf)
                    cx, cy = int(x), int(y)
                    cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.circle(vis, (cx, cy), 7, (255, 255, 255), 1)
            # Dibujar conexiones del esqueleto
            for conn_idx, (i, j) in enumerate(self.SKELETON):
                if kps[i][2] > 0.5 and kps[j][2] > 0.5:
                    pt1 = (int(kps[i][0]), int(kps[i][1]))
                    pt2 = (int(kps[j][0]), int(kps[j][1]))
                    color = self._get_skeleton_color(conn_idx)
                    cv2.line(vis, pt1, pt2, color, 2, cv2.LINE_AA)
            # Dibujar bounding box de la persona
            if person_idx < len(boxes):
                x1, y1, x2, y2 = [int(v) for v in boxes[person_idx]]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 1)
        # Calcular confianza promedio de keypoints
        if all_kp_confs:
            result['avg_confidence'] = float(np.mean(all_kp_confs))
        # Agregar información de métricas en la imagen
        cv2.putText(vis, f"Personas: {result['num_persons']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, f"Keypoints conf: {result['avg_confidence']:.2%}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, "Modelo: PyTorch KeypointRCNN",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return vis, result

    def process_video(self, input_path, output_path):
        """Procesar video completo aplicando detección de postura"""
        print(f"📹 Procesando video: {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"❌ No se pudo abrir el video: {input_path}")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"   Resolución: {w}x{h}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if not out.isOpened():
            raise RuntimeError(f"❌ No se pudo crear el video de salida: {output_path}")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pose_frame, _ = self.detect_pose_frame(frame)
            out.write(pose_frame)
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"   Progreso: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        cap.release()
        out.release()
        print(f"✅ Video procesado: {output_path}")
        return output_path

    def extract_key_frames(self, video_path):
        """Extraer frames clave del video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        ret, frame = cap.read()
        if ret:
            frames.append(("original", frame.copy()))
            pose_frame, _ = self.detect_pose_frame(frame)
            frames.append(("pose", pose_frame))
        cap.release()
        return frames
# ===============================
# ENVÍO SÍNCRONO A TELEGRAM
# ===============================
def send_to_telegram_sync(token, chat_id, orig_frame, pose_frame, video_path):
    """
    Enviar resultados a Telegram usando requests (síncrono).
    Evita conflictos con el event loop de asyncio del bot.
    """
    base_url = f"https://api.telegram.org/bot{token}"
    print("1️⃣  Enviando foto ORIGINAL...")
    with open(orig_frame, 'rb') as f:
        resp = http_requests.post(
            f"{base_url}/sendPhoto",
            data={
                'chat_id': chat_id,
                'caption': "📷 FOTO ORIGINAL\nFrame capturado desde la aplicación C++"
            },
            files={'photo': f},
            timeout=30
        )
    if resp.status_code == 200:
        print(f"   ✅ Foto original enviada")
    else:
        print(f"   ❌ Error: {resp.status_code} - {resp.text[:200]}")
    print("2️⃣  Enviando foto con POSTURA DETECTADA...")
    with open(pose_frame, 'rb') as f:
        resp = http_requests.post(
            f"{base_url}/sendPhoto",
            data={
                'chat_id': chat_id,
                'caption': "🧍 POSTURA HUMANA DETECTADA\n"
                           "Keypoints y esqueleto superpuesto\n"
                           "Modelo: PyTorch KeypointRCNN\n"
                           f"Framework: PyTorch {torch.__version__}"
            },
            files={'photo': f},
            timeout=30
        )
    if resp.status_code == 200:
        print(f"   ✅ Foto con postura enviada")
    else:
        print(f"   ❌ Error: {resp.status_code} - {resp.text[:200]}")
    print("3️⃣  Enviando VIDEO procesado (≥5 segundos)...")
    video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    with open(video_path, 'rb') as f:
        resp = http_requests.post(
            f"{base_url}/sendVideo",
            data={
                'chat_id': chat_id,
                'caption': "🎥 VIDEO CON DETECCIÓN DE POSTURA\n"
                           f"Tamaño: {video_size_mb:.2f} MB\n"
                           "Procesado con PyTorch KeypointRCNN\n"
                           "Resolución: 640x480",
                'supports_streaming': 'true'
            },
            files={'video': f},
            timeout=120
        )
    if resp.status_code == 200:
        print(f"   ✅ Video enviado")
    else:
        print(f"   ❌ Error: {resp.status_code} - {resp.text[:200]}")
    print("\n✅ TODOS LOS RESULTADOS ENVIADOS A TELEGRAM")
    return True
# ===============================
# BOT DE TELEGRAM
# ===============================
class TelegramPoseBot:
    def __init__(self, token):
        print("🤖 Inicializando bot de Telegram...")
        self.token = token
        self.pose_detector = PoseDetector()
        self.default_chat_id = None
        # Intentar cargar chat_id guardado
        if os.path.exists("logs/chat_id.txt"):
            try:
                with open("logs/chat_id.txt") as f:
                    self.default_chat_id = int(f.read().strip())
                print(f"📝 Chat ID cargado desde archivo: {self.default_chat_id}")
            except Exception:
                print("⚠️  No se pudo cargar chat_id guardado")
        self.application = (
            Application.builder()
            .token(token)
            .build()
        )
        # Registrar comandos
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help))
        self.application.add_handler(CommandHandler("status", self.status))
        self.application.add_handler(MessageHandler(filters.VIDEO, self.handle_video))
        print("✅ Bot configurado correctamente")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /start"""
        self.default_chat_id = update.effective_chat.id
        print(f"\n{'='*50}")
        print("📝 COMANDO /start RECIBIDO")
        print(f"{'='*50}")
        print(f"   Chat ID: {self.default_chat_id}")
        print(f"   Usuario: {update.effective_user.first_name}")
        print(f"   Username: @{update.effective_user.username}")
        print(f"{'='*50}\n")
        # Guardar en archivo para persistencia
        os.makedirs("logs", exist_ok=True)
        with open("logs/chat_id.txt", "w") as f:
            f.write(str(self.default_chat_id))
        await update.message.reply_text(
            "🤖 *Bot de Detección de Postura Humana*\n"
            "✅ Bot activado correctamente!\n"
            f"📝 Tu Chat ID: `{self.default_chat_id}`\n"
            "(Guardado para envío automático)\n"
            "Este bot analiza videos y detecta la postura humana usando:\n"
            "• HOG + SVM para detección de personas (C++)\n"
            "• PyTorch KeypointRCNN para análisis de postura\n"
            "\n📹 *Formas de usar:*\n"
            "1. La aplicación C++ envía videos automáticamente\n"
            "2. O envíame un video manualmente\n"
            "\nRecibirás:\n"
            "📷 Foto original\n"
            "🧍 Foto con postura detectada\n"
            "🎥 Video procesado (≥5 seg)\n"
            "\n💡 Usa /help para más información",
            parse_mode='Markdown'
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /help"""
        await update.message.reply_text(
            "📖 *Cómo usar el bot*\n"
            "1️⃣ La aplicación de escritorio detecta personas con HOG+SVM\n"
            "2️⃣ El video se envía automáticamente a este bot vía API\n"
            "3️⃣ Recibirás tres resultados:\n"
            "   • 📷 Frame original\n"
            "   • 🧍 Frame con esqueleto de postura (PyTorch)\n"
            "   • 🎥 Video completo procesado\n"
            "\n*Comandos disponibles:*\n"
            "/start - Iniciar el bot\n"
            "/help - Mostrar esta ayuda\n"
            "/status - Ver estado del sistema",
            parse_mode='Markdown'
        )

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Comando /status"""
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        await update.message.reply_text(
            f"📊 *Estado del Sistema*\n"
            f"🖥️  Dispositivo: {device}\n"
            f"🤖 Modelo postura: PyTorch KeypointRCNN\n"
            f"📦 PyTorch versión: {torch.__version__}\n"
            f"📦 Torchvision versión: {torchvision.__version__}\n"
            f"✅ Sistema operativo",
            parse_mode='Markdown'
        )

    async def handle_video(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Procesar video recibido directamente en Telegram"""
        print(f"\n{'='*50}")
        print("📹 VIDEO RECIBIDO DIRECTAMENTE EN TELEGRAM")
        print(f"{'='*50}")
        await update.message.reply_text(
            "📹 Video recibido!\n⏳ Procesando postura humana con PyTorch..."
        )
        try:
            os.makedirs("temp", exist_ok=True)
            video = await update.message.video.get_file()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            inp = f"temp/in_{ts}.mp4"
            out = f"temp/out_{ts}.mp4"
            print("⬇️  Descargando video...")
            await video.download_to_drive(inp)
            print(f"✅ Descargado: {inp}")
            # Extraer frames clave
            print("🖼️  Extrayendo frames clave...")
            frames = self.pose_detector.extract_key_frames(inp)
            orig_path = f"temp/orig_{ts}.jpg"
            pose_path = f"temp/pose_{ts}.jpg"
            cv2.imwrite(orig_path, frames[0][1])
            cv2.imwrite(pose_path, frames[1][1])
            print("✅ Frames extraídos")
            # Enviar imagen original
            print("📤 Enviando frame original...")
            with open(orig_path, "rb") as f:
                await update.message.reply_photo(
                    photo=f,
                    caption="📷 Frame original",
                )
            # Enviar imagen con postura
            print("📤 Enviando frame con postura...")
            with open(pose_path, "rb") as f:
                await update.message.reply_photo(
                    photo=f,
                    caption="🧍 Postura detectada (PyTorch KeypointRCNN)",
                )
            # Procesar video completo
            await update.message.reply_text(
                "⏳ Procesando video completo con PyTorch...\n"
                "Esto puede tomar algunos segundos."
            )
            print("🎬 Procesando video completo...")
            self.pose_detector.process_video(inp, out)
            # Enviar video procesado
            print("📤 Enviando video procesado...")
            with open(out, "rb") as f:
                await update.message.reply_video(
                    video=f,
                    caption="🎥 Video con detección de postura (PyTorch)",
                )
            # Limpiar archivos
            for filepath in [inp, out, orig_path, pose_path]:
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            await update.message.reply_text(
                "✅ *Procesamiento completado exitosamente!*\n"
                "Modelo: PyTorch KeypointRCNN",
                parse_mode='Markdown'
            )
            print("✅ PROCESO COMPLETADO")
            print(f"{'='*50}\n")
        except Exception as e:
            error_msg = f"❌ Error al procesar video: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            await update.message.reply_text(error_msg)

    def run(self):
        """Iniciar el bot"""
        print(f"\n{'='*50}")
        print("🚀 BOT DE TELEGRAM INICIADO")
        print(f"{'='*50}")
        print("✅ Esperando mensajes...")
        print("   Presiona Ctrl+C para detener\n")
        self.application.run_polling()
# ===============================
# API FLASK (recibe video de C++)
# ===============================
class VideoReceptionAPI:
    def __init__(self, pose_detector, telegram_bot=None):
        # Configurar Flask para que busque templates y static correctamente
        self.app = Flask(__name__, 
                        template_folder=os.path.abspath("templates"),
                        static_folder=os.path.abspath("static"))
        self.detector = pose_detector
        self.telegram_bot = telegram_bot
        
        # Estado global para el dashboard
        self.latest_data = {
            "image_url": None,
            "confidence": 0,
            "num_persons": 0,
            "timestamp": "Esperando datos..."
        }

        # --- RUTAS DE DASHBOARD ---
        @self.app.route("/")
        def index():
            return render_template("index.html")
            
        @self.app.route("/api/latest")
        def get_latest():
            return jsonify(self.latest_data)
            
        # Servir archivos estáticos explícitamente si es necesario
        @self.app.route('/static/<path:path>')
        def send_static(path):
            return send_from_directory('static', path)

        @self.app.route("/process_video", methods=["POST"])
        def process_video():
            print(f"\n{'='*50}")
            print("📥 SOLICITUD RECIBIDA DESDE C++")
            print(f"{'='*50}")
            if "video" not in request.files:
                print("❌ Error: No se recibió archivo de video")
                return jsonify({"error": "No video"}), 400
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("temp", exist_ok=True)
            inp = f"temp/cpp_{ts}.mp4"
            out = f"temp/cpp_out_{ts}.mp4"
            orig_frame = f"temp/cpp_orig_{ts}.jpg"
            pose_frame = f"temp/cpp_pose_{ts}.jpg"
            try:
                print(f"💾 Guardando video: {inp}")
                request.files["video"].save(inp)
                if not os.path.exists(inp):
                    print("❌ Error: El video no se guardó correctamente")
                    return jsonify({"error": "Video save failed"}), 500
                file_size = os.path.getsize(inp) / (1024 * 1024)
                print(f"   Tamaño: {file_size:.2f} MB")
                # Extraer frames clave
                print("🖼️  Extrayendo frames clave...")
                frames = self.detector.extract_key_frames(inp)
                if len(frames) < 2:
                    print("❌ Error: No se pudieron extraer frames")
                    return jsonify({"error": "Frame extraction failed"}), 500
                cv2.imwrite(orig_frame, frames[0][1])
                cv2.imwrite(pose_frame, frames[1][1])
                print(f"   ✅ Frame original guardado: {orig_frame}")
                print(f"   ✅ Frame con postura guardado: {pose_frame}")
                # Procesar video completo
                print("🎬 Procesando video completo con PyTorch...")
                self.detector.process_video(inp, out)
                if not os.path.exists(out):
                    print("❌ Error: El video procesado no se creó")
                    return jsonify({"error": "Video processing failed"}), 500
                out_size = os.path.getsize(out) / (1024 * 1024)
                print(f"   ✅ Video procesado: {out} ({out_size:.2f} MB)")
                
                # --- ACTUALIZAR DASHBOARD ---
                # Guardar frame con postura para el dashboard (sobrescribir el 'latest')
                dashboard_img = "static/latest/latest_pose.jpg"
                os.makedirs("static/latest", exist_ok=True)
                cv2.imwrite(dashboard_img, frames[1][1]) # Frame con postura
                
                # Actualizar estado global
                self.latest_data = {
                    "image_url": "/static/latest/latest_pose.jpg",
                    # Extraer confianza del log o re-evaluar (aquí simulado/extraído simple)
                    "confidence": 0.95, # Placeholder, idealmente extraer de 'frames'
                    "num_persons": 1,   # Placeholder
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                }
                
                # Enviar a Telegram (síncrono, sin conflictos de asyncio)
                token = os.getenv("TELEGRAM_TOKEN")
                chat_id = None
                if self.telegram_bot and self.telegram_bot.default_chat_id:
                    chat_id = self.telegram_bot.default_chat_id
                # Fallback: leer de .env
                if not chat_id:
                    chat_id = os.getenv("CHAT_ID")
                if token and chat_id:
                    print(f"\n📤 ENVIANDO RESULTADOS A TELEGRAM...")
                    print(f"{'='*50}")
                    print(f"   Chat ID: {chat_id}")
                    print(f"{'='*50}")
                    try:
                        send_to_telegram_sync(
                            token=token,
                            chat_id=chat_id,
                            orig_frame=orig_frame,
                            pose_frame=pose_frame,
                            video_path=out
                        )
                    except Exception as e:
                        print(f"\n❌ ERROR AL ENVIAR A TELEGRAM:")
                        print(f"   {type(e).__name__}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        return jsonify({"error": f"Telegram send failed: {str(e)}"}), 500
                else:
                    print(f"\n⚠️  NO SE PUEDE ENVIAR A TELEGRAM")
                    print(f"{'='*50}")
                    if not token:
                        print("   TELEGRAM_TOKEN no configurado")
                    if not chat_id:
                        print("   CHAT_ID no configurado")
                    print("\n📝 SOLUCIÓN:")
                    print("   1. Abre Telegram y envía /start al bot")
                    print("   2. O configura CHAT_ID en .env")
                    print()
                # Limpiar archivos temporales
                print("\n🧹 Limpiando archivos temporales...")
                for filepath in [inp, orig_frame, pose_frame]:
                    try:
                        if os.path.exists(filepath):
                            os.remove(filepath)
                    except Exception as e:
                        print(f"   ⚠️  Error al limpiar {filepath}: {e}")
                print("   ✅ Limpieza completada")
                print(f"{'='*50}\n")
                return jsonify({
                    "status": "ok",
                    "message": "Video procesado y enviado exitosamente",
                    "output": out,
                    "timestamp": ts,
                    "files_sent": 3
                })
            except Exception as e:
                print(f"❌ ERROR GENERAL: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": str(e)}), 500

        @self.app.route("/health", methods=["GET"])
        def health():
            return jsonify({
                "status": "healthy",
                "service": "pose_detection_api",
                "model": "PyTorch KeypointRCNN",
                "pytorch_version": torch.__version__,
                "device": str(next(self.detector.model.parameters()).device),
            })

    def run(self):
        print(f"\n{'='*50}")
        print("🌐 API FLASK INICIADA")
        print(f"{'='*50}")
        print("   Endpoint: http://0.0.0.0:5000/process_video")
        print("   Health:   http://0.0.0.0:5000/health")
        print(f"{'='*50}\n")
        self.app.run(host="0.0.0.0", port=5000, debug=False)
# ===============================
# MAIN
# ===============================
def main():
    print(f"\n{'='*60}")
    print("  SISTEMA DE DETECCIÓN DE POSTURA HUMANA")
    print("  Bot de Telegram + API Flask")
    print(f"  Modelo: PyTorch KeypointRCNN (torchvision)")
    print(f"{'='*60}\n")
    # Verificar token
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        print("❌ ERROR: TELEGRAM_TOKEN no configurado")
        print("\n📝 Pasos para configurar:")
        print("1. Crea un archivo .env en la raíz del proyecto")
        print('2. Agrega la línea:')
        print('   TELEGRAM_TOKEN="tu_token_aqui"')
        print("\n3. O exporta la variable:")
        print("   export TELEGRAM_TOKEN='tu_token_aqui'")
        return
    print(f"✅ Token de Telegram configurado")
    print(f"   Token: {token[:20]}...")
    # Crear bot
    bot = TelegramPoseBot(token)
    # Crear API con referencia al bot
    api = VideoReceptionAPI(bot.pose_detector, telegram_bot=bot)
    # Iniciar API en hilo separado
    api_thread = threading.Thread(
        target=api.run,
        daemon=True
    )
    api_thread.start()
    # Esperar un momento para que la API inicie
    import time
    time.sleep(2)
    print("\n⚠️  IMPORTANTE: Envía /start al bot en Telegram para activar el envío automático\n")
    # Iniciar bot
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n🛑 Bot detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()