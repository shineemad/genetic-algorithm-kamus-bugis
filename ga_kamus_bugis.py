# ============================================================
#  ALGORITMA GENETIKA - PENCARIAN KATA KAMUS BAHASA BUGIS
#  Tugas: Genetic Algorithm (GA) untuk mencari kata dalam
#         kamus bahasa daerah (Bahasa Bugis)
# ============================================================
import csv
import os
import random

# ------------------------------------------------------------
# 1. DATASET : Kamus Bahasa Bugis dimuat dari file CSV
#    (kamus_bugis.csv berisi 100 kata Bugis beserta artinya)
# ------------------------------------------------------------
KAMUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "kamus_bugis.csv")


def muat_kamus(nama_file=KAMUS_FILE):
    """Membaca kamus dari file CSV (kolom: bugis, indonesia)."""
    if not os.path.exists(nama_file):
        raise SystemExit(f"File kamus '{nama_file}' tidak ditemukan. "
                         "Pastikan kamus_bugis.csv ada di folder program.")
    kamus = {}
    with open(nama_file, newline="", encoding="utf-8") as f:
        for baris in csv.DictReader(f):
            kata = baris["bugis"].strip().lower()
            arti = baris["indonesia"].strip()
            if kata:
                kamus[kata] = arti
    return kamus


KAMUS = muat_kamus()

# Gen diambil dari huruf-huruf yang muncul pada kata kamus
# (meniru contoh PPT: populasi berupa kata KOTA/KITA/DATA/KASA)
ALFABET = "".join(sorted(set("".join(KAMUS.keys()))))
UKURAN_POPULASI = 20
PELUANG_MUTASI = 0.2
MAKS_GENERASI = 500


# ------------------------------------------------------------
# Fungsi-fungsi inti Algoritma Genetika
# ------------------------------------------------------------
def buat_individu(panjang):
    """Membuat satu individu (kromosom) berupa string huruf acak."""
    return "".join(random.choice(ALFABET) for _ in range(panjang))


def buat_populasi(panjang, n=UKURAN_POPULASI):
    return [buat_individu(panjang) for _ in range(n)]


def hitung_fitness(individu, target):
    """Fitness = jumlah huruf benar pada posisi yang tepat / panjang kata."""
    benar = sum(1 for a, b in zip(individu, target) if a == b)
    return benar / len(target)


def tabel_fitness(populasi, target, judul="TABEL FITNESS"):
    print(f"\n--- {judul} (target: '{target}') ---")
    print(f"{'Individu':<10}{'Kromosom':<14}{'Huruf Benar':<13}{'Fitness':<8}")
    hasil = []
    for i, ind in enumerate(populasi, 1):
        fit = hitung_fitness(ind, target)
        benar = round(fit * len(target))
        hasil.append(fit)
        print(f"I{i:<9}{ind:<14}{benar:<13}{fit:.4f}")
    print(f"Total fitness = {sum(hasil):.4f}")
    return hasil


def seleksi_roulette(populasi, fitness_list, jumlah=2, tampil=True):
    """Roulette Wheel Selection: probabilitas -> interval kumulatif -> putar."""
    total = sum(fitness_list)
    if total == 0:
        if tampil:
            print("Semua fitness = 0, parent dipilih acak.")
        return random.sample(populasi, jumlah)

    prob = [f / total for f in fitness_list]

    # interval kumulatif
    interval, batas_bawah = [], 0.0
    for p in prob:
        interval.append((batas_bawah, batas_bawah + p))
        batas_bawah += p

    if tampil:
        print("\n--- SELEKSI ROULETTE WHEEL ---")
        print(f"{'Individu':<10}{'Kromosom':<14}{'Fitness':<10}{'Prob.':<10}{'Interval':<20}")
        for i, (ind, f, p, (lo, hi)) in enumerate(
                zip(populasi, fitness_list, prob, interval), 1):
            iv = f"{lo:.4f} - {hi:.4f}" if p > 0 else "-"
            print(f"I{i:<9}{ind:<14}{f:<10.4f}{p:<10.4f}{iv:<20}")

    terpilih = []
    for k in range(jumlah):
        r = random.random()
        for i, (lo, hi) in enumerate(interval):
            if lo <= r < hi:
                terpilih.append(populasi[i])
                if tampil:
                    print(f"Putaran {k+1}: r = {r:.4f}  ->  jatuh di interval "
                          f"I{i+1} ({lo:.4f} - {hi:.4f})  ->  parent = '{populasi[i]}'")
                break
        else:
            terpilih.append(populasi[-1])
    return terpilih


def crossover(parent1, parent2, tampil=True):
    """Single point crossover."""
    titik = random.randint(1, len(parent1) - 1)
    child1 = parent1[:titik] + parent2[titik:]
    child2 = parent2[:titik] + parent1[titik:]
    if tampil:
        print("\n--- CROSSOVER (satu titik) ---")
        print(f"Parent 1 : {parent1}")
        print(f"Parent 2 : {parent2}")
        print(f"Titik potong = posisi {titik}")
        print(f"Child 1  : {parent1[:titik]} + {parent2[titik:]} = {child1}")
        print(f"Child 2  : {parent2[:titik]} + {parent1[titik:]} = {child2}")
    return child1, child2


def mutasi(individu, peluang=PELUANG_MUTASI, tampil=True):
    """Mutasi gen: mengganti satu huruf pada posisi acak."""
    r = random.random()
    if r < peluang:
        pos = random.randint(0, len(individu) - 1)
        huruf_lama = individu[pos]
        huruf_baru = random.choice(ALFABET.replace(huruf_lama, ""))
        hasil = individu[:pos] + huruf_baru + individu[pos + 1:]
        if tampil:
            print(f"Mutasi '{individu}': r={r:.4f} < {peluang} -> posisi {pos+1}: "
                  f"'{huruf_lama}' menjadi '{huruf_baru}'  =>  '{hasil}'")
        return hasil
    if tampil:
        print(f"Mutasi '{individu}': r={r:.4f} >= {peluang} -> tidak terjadi mutasi")
    return individu


def generasi_baru(populasi, fitness_list, target, tampil=True):
    """Elitisme 2 terbaik + child hasil seleksi-crossover-mutasi."""
    urut = sorted(zip(populasi, fitness_list), key=lambda x: x[1], reverse=True)
    baru = [urut[0][0], urut[1][0]]           # elitisme
    while len(baru) < len(populasi):
        p1, p2 = seleksi_roulette(populasi, fitness_list, 2, tampil=False)
        c1, c2 = crossover(p1, p2, tampil=False)
        baru.append(mutasi(c1, tampil=False))
        if len(baru) < len(populasi):
            baru.append(mutasi(c2, tampil=False))
    if tampil:
        print("\n--- POPULASI GENERASI BARU ---")
        for i, ind in enumerate(baru, 1):
            print(f"I{i}: {ind}  (fitness = {hitung_fitness(ind, target):.4f})")
    return baru


# ------------------------------------------------------------
# Fungsi Menu
# ------------------------------------------------------------
def tampilkan_kamus():
    print("\n===== KAMUS BAHASA BUGIS =====")
    print(f"{'No':<4}{'Bugis':<14}{'Indonesia':<20}")
    for i, (bugis, arti) in enumerate(KAMUS.items(), 1):
        print(f"{i:<4}{bugis:<14}{arti:<20}")


def cari_kata():
    kata = input("Masukkan kata (Bugis/Indonesia): ").strip().lower()
    ketemu = False
    for bugis, arti in KAMUS.items():
        kata_arti = arti.lower().replace("/", " ").split()
        if kata == bugis or kata in kata_arti:
            print(f"  -> '{bugis}' artinya '{arti}'")
            ketemu = True
    if not ketemu:
        print("  Kata tidak ditemukan dalam kamus.")


def pilih_target():
    tampilkan_kamus()
    while True:
        kata = input("\nPilih kata target dari kamus: ").strip().lower()
        if kata in KAMUS:
            return kata
        print("Kata tidak ada di kamus, coba lagi.")


def jalankan_ga_penuh(state):
    target = pilih_target()
    populasi = buat_populasi(len(target))
    print(f"\nTarget: '{target}' ({KAMUS[target]}) | Populasi: {UKURAN_POPULASI} "
          f"| Peluang mutasi: {PELUANG_MUTASI}")
    for gen in range(1, MAKS_GENERASI + 1):
        fitness_list = [hitung_fitness(ind, target) for ind in populasi]
        terbaik = max(zip(populasi, fitness_list), key=lambda x: x[1])
        print(f"Generasi {gen:>3}: terbaik = '{terbaik[0]}' (fitness {terbaik[1]:.4f})")
        if terbaik[1] == 1.0:
            print(f"\n>>> Kata '{target}' DITEMUKAN pada generasi ke-{gen}! <<<")
            break
        populasi = generasi_baru(populasi, fitness_list, target, tampil=False)
    else:
        print(f"\nKata belum ditemukan sampai generasi {MAKS_GENERASI}.")
    state["target"] = target
    state["populasi"] = populasi
    state["generasi"] = gen
    # reset hasil tahapan lama agar tidak tercampur dengan populasi baru
    state["fitness"] = None
    state["parents"] = None
    state["children"] = None
    state["mutan"] = None


def main():
    state = {"target": None, "populasi": None, "generasi": 0,
             "fitness": None, "parents": None, "children": None,
             "mutan": None}

    while True:
        print("\n=== Kamus Bahasa Bugis - Algoritma Genetika ===")
        print(" 1. Tampilkan Kamus")
        print(" 2. Cari Kata")
        print(" 3. Jalankan Algoritma Genetika")
        print(" 4. Tampilkan Populasi")
        print(" 5. Hasil Fitness")
        print(" 6. Seleksi Roulette")
        print(" 7. Cross Over")
        print(" 8. Mutasi")
        print(" 9. Generasi Baru")
        print("10. Keluar")
        pilih = input("Pilih menu [1-10]: ").strip()

        if pilih == "1":
            tampilkan_kamus()

        elif pilih == "2":
            cari_kata()

        elif pilih == "3":
            jalankan_ga_penuh(state)

        elif pilih == "4":
            if state["populasi"] is None:
                state["target"] = pilih_target()
                state["populasi"] = buat_populasi(len(state["target"]))
                print("\nPopulasi awal dibuat secara acak.")
            print(f"\n--- POPULASI (target: '{state['target']}') ---")
            for i, ind in enumerate(state["populasi"], 1):
                print(f"I{i}: {ind}")

        elif pilih == "5":
            if state["populasi"] is None:
                print("Buat populasi dulu (menu 4) atau jalankan GA (menu 3).")
                continue
            state["fitness"] = tabel_fitness(state["populasi"], state["target"])

        elif pilih == "6":
            if state["fitness"] is None:
                print("Hitung fitness dulu (menu 5).")
                continue
            state["parents"] = seleksi_roulette(state["populasi"], state["fitness"])

        elif pilih == "7":
            if state["parents"] is None:
                print("Lakukan seleksi dulu (menu 6).")
                continue
            state["children"] = crossover(*state["parents"])

        elif pilih == "8":
            if state["children"] is None:
                print("Lakukan crossover dulu (menu 7).")
                continue
            print("\n--- MUTASI GEN ---")
            state["mutan"] = [mutasi(c) for c in state["children"]]
            tabel_fitness(state["mutan"], state["target"], "EVALUASI CHILD")

        elif pilih == "9":
            if state["fitness"] is None:
                print("Hitung fitness dulu (menu 5).")
                continue
            state["populasi"] = generasi_baru(
                state["populasi"], state["fitness"], state["target"])
            state["generasi"] += 1
            state["fitness"] = None
            state["parents"] = None
            state["children"] = None
            print(f"\nGenerasi ke-{state['generasi']} terbentuk. "
                  f"Hitung ulang fitness lewat menu 5.")

        elif pilih == "10":
            print("Kurru sumange'! (Terima kasih!) Program selesai.")
            break

        else:
            print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()
