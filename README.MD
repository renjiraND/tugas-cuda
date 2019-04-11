### Praktikum 3 CUDA

#### Petunjuk penggunaan program:
1. Kompilasi program dengan menjalankan command ```make```
2. Jalankan program dengan menggunakan pemanggilan command ```./cuda```. Dengan NP adalah jumlah proses dan N adalah jumlah elemen array.

#### Pembagian Tugas
13516014 - Renjira Naufhal Dhiaegana
1. Membuat fungsi countort, counting, main pada radixsort.cu
2. Membuat makefile
3. Membuat README.md

13516131 - Intan Nurjanah
1. Membuat countsort.cu
2. Membuat fungsi getMax, radixsort pada radixsort.cu
3. Bugfixing

#### Laporan Pengerjaan
##### Solusi Paralel
Pertama, lakukan pencarian nilai maksimum dari array. Lalu lakukan count_sort terhadap tiap exp. Pada countsort, lakukan pembagian data dan lakukan count terhadap tiap elemen pada array yang sudah dibagi. Lakukan syncthread terhadap semua thread yang sudah dibagi, dan kembalikan array count ke host. Lalu lakukan sort dengan inclusive_scan(dengan library thrust).

##### Analisis solusi yang anda berikan. Apakah mungkin terdapat solusi yang memberikan kinerja lebih baik?
Solusi yang kami berikan hanya baik untuk elemen array yang banyak karena overhead yang terjadi karena pembuatan thread. Baiknya, lakukan radix sort serial apabila jumlah elemen sedikit.

##### Pengukuran kinerja untuk tiap kasus uji (jumlah N pada array) 
1. N = 5000
   Waktu rata-rata program paralel = 0.001632, program serial = 104.660255
2. N = 50000
   Waktu rata-rata program paralel = 17.139648, program serial = 53.781727
3. N = 100000
   Waktu rata-rata program paralel = 31.216896, program serial = 56.234081
4. N = 200000
   Waktu rata-rata program paralel = 68.454399, program serial = 103.973282
5. N = 400000
   Waktu rata-rata program paralel = 129.312347, program serial = 154.401505


##### Analisis perbandingan kinerja serial dan paralel. 
Waktu untuk pengerjaan parallel hanya baik untuk array yang besar. Sedangkan akan lama apabila array kecil karena jumlah thread yang banyak menyebabkan overhead