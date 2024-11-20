# Import library yang diperlukan
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Memuat data dari file Excel
file_path = "database_obat_300.xlsx"
data = pd.read_excel(file_path)

# Rename kolom agar lebih konsisten (opsional, sesuai dengan data Anda)
data.rename(columns={
    "Nama Obat": "nama_obat",
    "Indikasi": "indikasi",
    "Dosis": "dosis",
    "Kontraindikasi": "kontraindikasi",
    "Efek Samping": "efek_samping"
}, inplace=True)

# Preprocessing: memastikan kolom 'indikasi' menggunakan huruf kecil
data['indikasi'] = data['indikasi'].str.lower()

# Membuat vektor TF-IDF dari kolom 'indikasi'
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['indikasi'])

# Membuat model KNN untuk mencari rekomendasi obat
model = NearestNeighbors(n_neighbors=3, metric='cosine')  # Mengambil 3 obat terdekat
model.fit(X)

# Fungsi untuk merekomendasikan obat berdasarkan input gejala
def rekomendasi_obat(gejala):
    # Mengubah input gejala menjadi vektor TF-IDF
    gejala_vec = vectorizer.transform([gejala.lower()])
    
    # Mencari obat terdekat
    distances, indices = model.kneighbors(gejala_vec)
    
    # Menampilkan hasil rekomendasi
    print("Rekomendasi obat untuk gejala:", gejala)
    for idx in indices[0]:
        print("- Obat:", data.iloc[idx]['nama_obat'])
        print("  Indikasi:", data.iloc[idx]['indikasi'])
        print("  Dosis:", data.iloc[idx]['dosis'])
        print("  Efek Samping:", data.iloc[idx]['efek_samping'])
        print()

# Menjalankan chatbot
if __name__ == "__main__":
    while True:
        gejala_input = input("Masukkan gejala yang Anda rasakan (atau ketik 'exit' untuk keluar): ")
        if gejala_input.lower() == 'exit':
            print("Terima kasih telah menggunakan chatbot rekomendasi obat!")
            break
        rekomendasi_obat(gejala_input)
