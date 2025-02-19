from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import uvicorn
import numpy as np
import cv2

app = FastAPI()

# تحميل النموذج المدرب
model = tf.keras.models.load_model("3dcnn_model.h5")


def preprocess_video(video_file, num_frames=16, target_size=(112, 112)):
    cap = cv2.VideoCapture(video_file)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frames.append(frame)

    cap.release()
    while len(frames) < num_frames:
        frames.append(np.zeros((112, 112, 3), np.uint8))

    return np.array(frames)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    video_path = f"temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(file.file.read())

    frames = preprocess_video(video_path)
    frames = np.expand_dims(frames, axis=0)
    prediction = model.predict(frames)
    class_id = np.argmax(prediction)

    return {"predicted_class": int(class_id), "confidence": float(np.max(prediction))}

# لتشغيل السيرفر
# في الجهاز المحلي: uvicorn main:app --reload
if __name__ == "__main__":
    uvicorn.run("API:app", host="0.0.0.0", port=8000, reload=True)