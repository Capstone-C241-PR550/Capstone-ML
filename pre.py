import pandas as pd

# Data
data = {
    "Nama": ["Amanda", "Berlinda", "Bunga", "Cici", "Cintya", "Dede", "Delvia", "Didi", "Elvia", "Eni",
             "Fani", "Febri", "Futri", "Gani", "Gessy", "Hani", "Herliana", "Iki", "Ismail", "Jaki",
             "Jeni", "Julia", "Kiki", "Lani", "Lia", "Masyuri", "Mutia", "Nama", "Nana", "Nining",
             "Oki", "Pia", "Putri", "Reny", "Riki", "Rista", "Sabrina", "Sari", "Sayidah", "Siti",
             "Syerliana", "Tina", "Tri", "Uki", "Widya", "Windi", "Windri", "Yessi", "Zicky"],
    "Penghasilan Bulanan": [5500000, 6200000, 7800000, 8500000, 9100000, 5700000, 6400000, 7900000, 8600000, 9200000,
                            5900000, 6600000, 8100000, 8800000, 9400000, 5400000, 6100000, 8200000, 8900000, 9500000,
                            5800000, 6500000, 8000000, 8700000, 9300000, 9500000, 7000000, 10000000, 10000000, 6500000,
                            8000000, 8000000, 7000000, 10000000, 7000000, 8000000, 7000000, 7500000, 6500000, 10000000,
                            7000000, 6000000, 5500000, 8000000, 10000000, 8000000, 5000000, 9000000, 5000000],
    "Pengeluaran Bulanan": [6000000, 7500000, 5800000, 6500000, 7200000, 5300000, 7800000, 6100000, 7400000, 7900000,
                            5500000, 8000000, 6200000, 7600000, 8100000, 5200000, 7700000, 6000000, 7300000, 7800000,
                            5400000, 7900000, 6300000, 7700000, 8000000, 6000000, 6000000, 7000000, 8000000, 4500000,
                            5500000, 5000000, 5000000, 6000000, 6000000, 5500000, 4500000, 4500000, 4000000, 7000000,
                            5500000, 4500000, 4000000, 5500000, 8000000, 5000000, 3000000, 5500000, 3500000],
    "Tabungan Bulanan": [500000, 1300000, 2000000, 2000000, 1900000, 400000, 1400000, 1800000, 1200000, 1300000,
                         400000, 1400000, 1900000, 1200000, 1300000, 200000, 1600000, 2200000, 1600000, 1700000,
                         400000, 1400000, 1700000, 1000000, 1300000, 3500000, 1000000, 3000000, 2000000, 2000000,
                         2500000, 3000000, 2000000, 4000000, 1000000, 2500000, 2500000, 3000000, 2500000, 3000000,
                         1500000, 1500000, 1500000, 2500000, 2000000, 3000000, 2000000, 3500000, 1500000]
}

# Konversi ke skala 1:1000000 dan gantikan nilai sebelumnya
data["Penghasilan Bulanan"] = [round(x / 1000000, 1) for x in data["Penghasilan Bulanan"]]
data["Pengeluaran Bulanan"] = [round(x / 1000000, 1) for x in data["Pengeluaran Bulanan"]]
data["Tabungan Bulanan"] = [round(x / 1000000, 1) for x in data["Tabungan Bulanan"]]

df = pd.DataFrame(data)

print(df.head())

# Tambahkan kolom hasil analisis
df['Tingkat Tabungan'] = round((df['Tabungan Bulanan'] / df['Penghasilan Bulanan']) * 100)
df['Rasio Pengeluaran'] = round((df['Pengeluaran Bulanan'] / df['Penghasilan Bulanan']) * 100)
df['Sisa Penghasilan'] = round(df['Penghasilan Bulanan'] - df['Pengeluaran Bulanan'])

# Setel 'Sisa Penghasilan' ke 0 jika nilainya kurang dari atau sama dengan 0
df['Sisa Penghasilan'] = df['Sisa Penghasilan'].apply(lambda x: 0 if x <= 0 else x)

# Tambahkan kolom kondisi kesehatan keuangan
def kesehatan_tabungan(row):
    if row['Tingkat Tabungan'] < 20:
        return "kurang"
    elif 20 <= row['Tingkat Tabungan'] <= 30:
        return "baik"
    else:
        return "sangat baik"

df['kesehatan tabungan'] = df.apply(kesehatan_tabungan, axis=1)

# Ekspor DataFrame ke file CSV
df.to_csv('dataset.csv', index=False)

print(df.head())