import streamlit as st
import os
from tts_bark import BarkTTS
import base64
import nltk

#nltk.download('punkt_tab')

# NLTK verilerini indir
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')  # Tüm NLTK verilerini indir
        nltk.download('punkt_tab')

def get_audio_player(file_path):
    """Ses dosyasını HTML audio player olarak döndürür"""
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return f'<audio controls><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'


def main():
    # NLTK verilerini indir
    with st.spinner('NLTK verileri yükleniyor...'):
        download_nltk_data()

    st.title("🎙️ Bark Text-to-Speech Dönüştürücü")
    st.write("Metninizi sese dönüştürmek için aşağıdaki alana yazın.")

    # TTS modelini başlat
    if 'tts' not in st.session_state:
        with st.spinner('Model yükleniyor...'):
            st.session_state.tts = BarkTTS()
        st.success('Model yüklendi!')

    # Metin girişi
    text_input = st.text_area("Metninizi buraya yazın:", height=150)

    # Dönüştür butonu
    if st.button("Sese Dönüştür"):
        if text_input.strip():
            try:
                with st.spinner('Ses oluşturuluyor...'):
                    # Benzersiz dosya adı oluştur
                    output_file = f"output_{hash(text_input)}.wav"

                    # Sesi oluştur
                    st.session_state.tts.generate_speech(text_input, output_file)

                    # Ses oynatıcıyı göster
                    st.markdown(get_audio_player(output_file), unsafe_allow_html=True)

                    # Dosyayı indirme linki
                    with open(output_file, "rb") as file:
                        btn = st.download_button(
                            label="Ses Dosyasını İndir",
                            data=file,
                            file_name=output_file,
                            mime="audio/wav"
                        )

                st.success("Ses başarıyla oluşturuldu!")

            except Exception as e:
                st.error(f"Bir hata oluştu: {str(e)}")
        else:
            st.warning("Lütfen bir metin girin.")

    # Eski ses dosyalarını temizle
    st.sidebar.title("Temizlik")
    if st.sidebar.button("Eski Ses Dosyalarını Temizle"):
        cleaned = 0
        for file in os.listdir():
            if file.startswith("output_") and file.endswith(".wav"):
                try:
                    os.remove(file)
                    cleaned += 1
                except:
                    pass
        if cleaned > 0:
            st.sidebar.success(f"{cleaned} ses dosyası temizlendi.")
        else:
            st.sidebar.info("Temizlenecek dosya bulunamadı.")


if __name__ == "__main__":
    main()
