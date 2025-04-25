import cv2
import torch
import numpy as np
import threading
import time
import os
from datetime import datetime
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler

# ======= KONFIGURASI =======
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
VIDEO_SOURCE = 'salipkiri.mp4'

# Model YOLOv5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

width, height = 600, 600
triangle_points = np.array([[width, height * 0.32], [width * 0.2, height], [width, height]], np.int32)

detection_active = False
last_send_time = 0
send_image_cooldown = 30

def run_detection(context: CallbackContext):
    global detection_active, last_send_time
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    frame_skip = 5
    frame_count = 0

    while detection_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))

        if frame_count % frame_skip == 0:
            cv2.polylines(frame, [triangle_points], isClosed=True, color=(0, 0, 255), thickness=2)
            results = model(frame)
            df = results.pandas().xyxy[0]
            vehicles = df[(df['name'].isin(['car', 'truck'])) & (df['confidence'] > 0.7)]

            for _, row in vehicles.iterrows():
                xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                label = row['name']
                confidence = row['confidence']
                box_points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                in_triangle = any(cv2.pointPolygonTest(triangle_points, pt, False) >= 0 for pt in box_points)
                color = (0, 255, 0) if not in_triangle else (0, 0, 255)

                if in_triangle:
                    current_time = time.time()
                    if current_time - last_send_time > send_image_cooldown:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                        img_filename = 'overtake.jpg'
                        cv2.imwrite(img_filename, frame)
                        context.bot.send_photo(chat_id=CHAT_ID, photo=open(img_filename, 'rb'),
                                               caption=f'🚨 Pelanggaran: {label} menyalip kiri\n🕒 {timestamp}')

                        clip_duration = 5
                        fps = 20
                        clip_filename = 'detected_clip.mp4'
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - fps * clip_duration // 2))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))

                        frames_recorded = 0
                        while frames_recorded < clip_duration * fps:
                            ret_clip, frame_clip = cap.read()
                            if not ret_clip:
                                break
                            frame_clip = cv2.resize(frame_clip, (width, height))
                            out.write(frame_clip)
                            frames_recorded += 1
                        out.release()

                        context.bot.send_video(chat_id=CHAT_ID, video=open(clip_filename, 'rb'),
                                               caption=f'📹 Cuplikan pelanggaran: {label} menyalip kiri\n🕒 {timestamp}')
                        last_send_time = current_time

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def start(update: Update, context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("▶️ Mulai Deteksi", callback_data='start')],
        [InlineKeyboardButton("⛔ Setop Deteksi", callback_data='stop')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text("Kontrol Deteksi Kendaraan:", reply_markup=reply_markup)

def button_handler(update: Update, context: CallbackContext):
    global detection_active
    query = update.callback_query
    query.answer()

    if query.data == 'start':
        if not detection_active:
            detection_active = True
            query.edit_message_text("✅ Deteksi dimulai. Menunggu pelanggaran...")
            threading.Thread(target=run_detection, args=(context,), daemon=True).start()
        else:
            query.edit_message_text("⚠️ Deteksi sudah berjalan.")
    elif query.data == 'stop':
        if detection_active:
            detection_active = False
            query.edit_message_text("🛑 Deteksi dihentikan.")
        else:
            query.edit_message_text("🔕 Deteksi belum aktif.")

updater = Updater(BOT_TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler('start', start))
dp.add_handler(CallbackQueryHandler(button_handler))

updater.start_polling()
print("🤖 Bot Telegram aktif. Kirim /start ke bot.")
updater.idle()
