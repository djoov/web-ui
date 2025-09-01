# Browser Use Web UI (dengan DSPy)

[](https://github.com/browser-use/web-ui/stargazers)
[](https://link.browser-use.com/discord)
[](https://docs.browser-use.com)
[](https://x.com/warmshao)

Repositori ini adalah antarmuka pengguna web (WebUI) untuk mengontrol agen AI peramban, yang dibangun di atas fondasi `browser-use`. Versi ini telah dimodifikasi secara signifikan untuk mengintegrasikan **DSPy**, sebuah kerangka kerja dari Stanford untuk memprogram *Large Language Models* (LLM) secara sistematis.

Integrasi ini mengubah agen dari sekadar *prompt-driven* menjadi *program-driven*, yang memisahkan logika aplikasi dari parameter LLM. Hasilnya adalah agen yang lebih andal, modular, dan dapat dioptimalkan secara otomatis.

## ✨ Arsitektur Baru dengan DSPy

Perubahan inti dari proyek ini adalah penggantian logika *prompt engineering* manual dengan program DSPy yang terstruktur.

  * **Sebelumnya**: Agen secara manual membuat *prompt* panjang yang berisi instruksi, status peramban, dan riwayat untuk dikirim ke LLM.
  * **Sekarang (dengan DSPy)**:
    1.  **Signature (`GenerateActionSignature`)**: Kita mendefinisikan kontrak input/output yang jelas untuk LLM.
    2.  **Module (`BrowserAgentModule`)**: Sebuah program modular (`dspy.ChainOfThought`) menggunakan *signature* untuk menghasilkan tindakan berikutnya dengan penalaran yang terstruktur.
    3.  **Compiler**: DSPy bertindak sebagai "kompilator" yang mengubah program modular kita menjadi *prompt* yang efektif dan dioptimalkan untuk LLM apa pun yang dipilih.

Manfaat utama dari arsitektur ini adalah:

  * **Portabilitas Model**: Ganti LLM (OpenAI, Ollama, Google, dll.) dengan mudah tanpa mengubah kode agen.
  * **Optimasi Otomatis**: Gunakan *optimizer* DSPy untuk menyempurnakan kinerja agen secara sistematis.
  * **Kode yang Lebih Bersih**: Memisahkan logika program dari detail *prompting* yang rumit.
  * **Keandalan & Debugging**: Lacak penalaran (`rationale`) agen untuk memahami setiap keputusannya.

## 🚀 Cara Menjalankan

### Opsi 1: Instalasi Lokal

Baca [panduan memulai cepat](https://docs.browser-use.com/quickstart#prepare-the-environment) atau ikuti langkah-langkah di bawah ini.

#### Langkah 1: Kloning Repositori

```bash
git clone https://github.com/djoov/web-ui.git
cd web-ui
```

#### Langkah 2: Siapkan Lingkungan Python

Kami merekomendasikan penggunaan `uv` untuk manajemen lingkungan.

```bash
uv venv --python 3.11
source .venv/bin/activate  # Untuk macOS/Linux
.venv\Scripts\activate      # Untuk Windows
```

#### Langkah 3: Instal Dependensi

Pastikan `requirements.txt` menyertakan `dspy-ai` dan `setuptools`.

```bash
uv pip install -r requirements.txt
```

Instal peramban yang diperlukan oleh Playwright.

```bash
playwright install --with-deps chromium
```

#### Langkah 4: Konfigurasi Lingkungan

Salin file `.env.example` ke `.env` dan isi kunci API serta pengaturan LLM Anda.

```bash
cp .env.example .env
```

#### Langkah 5: Jalankan WebUI

```bash
python webui.py --ip 127.0.0.1 --port 7788
```

Buka `http://127.0.0.1:7788` di peramban Anda.

### Opsi 2: Instalasi Docker

#### Prasyarat

  - Docker dan Docker Compose terinstal.

#### Langkah 1 & 2: Kloning dan Konfigurasi

Ikuti Langkah 1 dan 4 dari instalasi lokal.

#### Langkah 3: Bangun dan Jalankan Docker

```bash
docker compose up --build
```

Untuk sistem ARM64 (misalnya, Apple Silicon Mac), gunakan:

```bash
TARGETPLATFORM=linux/arm64 docker compose up --build
```

#### Langkah 4: Akses Aplikasi

  - **Web-UI**: Buka `http://localhost:7788`
  - **VNC Viewer** (untuk melihat interaksi peramban): Buka `http://localhost:6080/vnc.html`
      - Kata sandi VNC default adalah "youvncpassword" (dapat diubah di `.env`).

## Verifikasi Integrasi DSPy

Saat Anda menjalankan tugas pertama kali melalui tab **"🤖 Run Agent"**, periksa terminal Anda. Anda akan melihat log berikut yang mengonfirmasi bahwa DSPy aktif:

```
✅ DSPy configured to use [provider] with model [model_name]
...
🧠 DSPy Reasoning: [Penalaran langkah demi langkah dari LLM]
🤖 DSPy Predicted Action: [Tindakan JSON yang dihasilkan]
```

Log ini adalah bukti bahwa otak agen Anda sekarang ditenagai oleh program DSPy yang telah kita integrasikan.
