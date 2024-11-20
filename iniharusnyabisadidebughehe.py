import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

# Membaca dataset
file_path = "database_obat_300.xlsx"
data = pd.read_excel(file_path)

# Preprocessing
print("Preprocessing data...")
data['Indikasi'] = data['Indikasi'].str.lower()  # Ubah teks indikasi ke huruf kecil
data['Indikasi'] = data['Indikasi'].fillna('')  # Isi nilai NaN dengan string kosong
data['Nama Obat'] = data['Nama Obat'].str.lower()  # Ubah nama obat ke huruf kecil

# Fitur dan label
X = data['Indikasi']
y = data['Nama Obat']

# TF-IDF untuk representasi teks
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Membagi data ke training dan testing
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Membuat model Machine Learning
print("Melatih model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100  # Akurasi dalam persen
print(f"Akurasi model: {accuracy:.2f}%")


# Fungsi chatbot
def chatbot_rekomendasi(gejala):
    gejala = gejala.lower()  # Ubah input ke huruf kecil
    gejala_tfidf = vectorizer.transform([gejala])  # Transformasi input menggunakan TF-IDF
    rekomendasi = model.predict(gejala_tfidf)  # Prediksi menggunakan model
    return rekomendasi[0]

# Menjalankan chatbot
print("\n=== Chatbot Rekomendasi Obat ===")
while True:
    gejala_input = input("Masukkan gejala yang Anda rasakan (atau ketik 'exit' untuk keluar): ")
    if gejala_input.lower() == 'exit':
        print("Terima kasih telah menggunakan chatbot!")
        break
    rekomendasi_obat = chatbot_rekomendasi(gejala_input)
    print(f"Rekomendasi obat: {rekomendasi_obat}")
