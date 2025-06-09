import streamlit as st
import asyncio
import os
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import whisper

from moviepy import VideoFileClip

# 修復 Streamlit + asyncio 錯誤
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# 初始化向量模型與索引
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = 384
index = faiss.IndexFlatL2(dimension)
documents = []
vectors = []

# 處理不同格式的文件
def extract_text(file):
    if file.type == "text/plain":
        return file.read().decode("utf-8")

    elif file.type == "application/pdf":
        pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in pdf_doc)

    elif file.type.startswith("image/"):
        image = Image.open(file)
        return pytesseract.image_to_string(image, lang="eng")  # 若需中文用 "chi_tra"

    elif file.type.startswith("video/"):
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(file.read())
        video = VideoFileClip(temp_path)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        os.remove(temp_path)
        os.remove(audio_path)
        return result["text"]

    else:
        return ""

# UI：主頁
st.set_page_config(page_title="RAG 文件問答系統", layout="centered")
st.title("📂 文件檢索問答系統（RAG + Ollama）")
st.caption("支援 `.txt`、`.pdf`、圖片（.jpg/.png）、影片（.mp4）")

# 上傳文件
uploaded_files = st.file_uploader(
    "請上傳文件：",
    accept_multiple_files=True,
    type=["txt", "pdf", "jpg", "jpeg", "png", "mp4"]
)

# 處理並建立知識庫向量
for file in uploaded_files:
    try:
        content = extract_text(file)
        if content:
            docs = content.split("\n\n")
            documents.extend(docs)
            vecs = embed_model.encode(docs)
            vectors.extend(vecs)
            st.success(f"✅ 成功處理：{file.name}")
        else:
            st.warning(f"⚠️ 無法處理內容：{file.name}")
    except Exception as e:
        st.error(f"❌ 錯誤於 {file.name}：{e}")

# 將向量加進 FAISS index
if vectors:
    index.add(np.array(vectors))

# 問答欄位
question = st.text_input(" 請輸入你的問題")
if st.button("送出問題") and question and documents:
    q_vector = embed_model.encode([question])
    D, I = index.search(q_vector, k=3)
    retrieved_docs = "\n".join([documents[i] for i in I[0]])

    # 构造 prompt 並呼叫 Ollama（本地模型）
    prompt = f"以下是文件內容摘要：\n{retrieved_docs}\n\n根據這些內容，回答問題：{question}"
    try:
        response = requests.post("http://localhost:11434/api/generate", json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        })
        answer = response.json().get("response", "⚠️ 沒有回傳有效結果")
        st.markdown("###  回答：")
        st.write(answer)
    except Exception as e:
        st.error(f"❌ 無法連線到本地模型（Ollama）: {e}")