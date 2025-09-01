import dspy
import json

class GenerateActionSignature(dspy.Signature):
    """
    Berdasarkan status saat ini, riwayat, dan tugas, hasilkan tindakan selanjutnya untuk berinteraksi dengan peramban.
    Tindakan harus dalam format JSON yang valid dan hanya menggunakan salah satu dari alat yang tersedia.
    """
    
    task = dspy.InputField(desc="Tugas utama yang harus diselesaikan oleh agen.")
    url = dspy.InputField(desc="URL halaman web saat ini.")
    title = dspy.InputField(desc="Judul halaman web saat ini.")
    elements = dspy.InputField(desc="Daftar elemen interaktif yang terlihat di halaman, dalam format JSON.")
    content = dspy.InputField(desc="Konten utama dari halaman web saat ini.")
    history = dspy.InputField(desc="Ringkasan dari tindakan-tindakan sebelumnya.")
    
    action = dspy.OutputField(desc="Tindakan selanjutnya yang harus dilakukan dalam format JSON. Contoh: {'click_element': {'index': 1}}")

class BrowserAgentModule(dspy.Module):
    """Modul DSPy untuk Agen Peramban."""
    def __init__(self):
        super().__init__()
        # Menggunakan ChainOfThought untuk penalaran yang lebih baik
        self.generate_action = dspy.ChainOfThought(GenerateActionSignature)

    def forward(self, task, url, title, elements, content, history):
        # Memanggil prediktor ChainOfThought
        prediction = self.generate_action(
            task=task,
            url=url,
            title=title,
            elements=json.dumps(elements, indent=2), # Pastikan elemen dalam format string JSON
            content=content,
            history=history
        )
        
        return dspy.Prediction(
            action=prediction.action,
            reasoning=prediction.rationale # Menyimpan penalaran untuk debugging
        )