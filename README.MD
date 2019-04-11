### Praktikum 3 CUDA

#### Petunjuk penggunaan program:
1. Kompilasi program dengan menjalankan command ```make```
2. Jalankan program dengan menggunakan pemanggilan command ```./cuda```. Dengan NP adalah jumlah proses dan N adalah jumlah elemen array.

#### Pembagian Tugas
13516014 - Renjira Naufal
1. Membuat fungsi countSort, main pada radixsort.cu
2. Membuat makefile
3. Membuat hostfile

13516131 - Intan Nurjanah
1. Membuat Sequential_RadixSort.c
2. Membuat fungsi getMax, radixsort pada radixsort.cu
3. Menyusun README.md

#### Laporan Pengerjaan
##### Solusi Paralel
Pertama, hitung banyaknya task. Task dapat dihitung dari jumlah data per processor (s = n/p). Setelah menghitung jumlah task, lakukan MPI_Scatter terhadap data array ary ke masing masing task. Setelah itu lakukan radix sort terhadap local_array yang sudah di scatter sebelumnya. Setelah melakukan radix sort, lakukan mpi gather dari local_array yangsudah di sort masing-masing ke array utama ary.

##### Jumlah Thread yang Digunakan
Pada algoritma MPI_Radixsort digunakan jumlah thread sebanyak lima karena saat menggunakan thread ini membutuhkan waktu yang lebih singkat dibandingkan jika jumlah threadnya lebih sedikit dari lima maupun lebih banyak dari lima

##### Pengukuran Kinerja Tiap Kasus
Berikut adalah perhitungan waktu dalam microsecond
1. N = 5000
   Waktu rata-rata program paralel = 900, program serial = 1600
2. N = 50000
   Waktu rata-rata program paralel = 1100, program serial = 2200
3. N = 100000
   Waktu rata-rata program paralel = 1400, program serial = 2100
4. N = 200000
   Waktu rata-rata program paralel = 1600, program serial = 2300
5. N = 400000
   Waktu rata-rata program paralel = 5000, program serial = 8400

##### Analisis Solusi
Pada algoritma paralel yang telah dibuat, rata-rata waktu yang dibutuhkan justru lebih singkat 50% dibandingkan dengan program serial. Tetapi tidak menutup kemungkinan akan ada program yang lebih efisien dari program ini karena speedup nya masih dibawah 80%.
