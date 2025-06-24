#  Personalized Book Recommendation System Powered by LLMs | Kişiselleştirilmiş Kitap Öneri Sistemi (LLM Destekli)

This project aims to provide personalized book recommendations through natural conversation using generative AI.  
Bu proje, doğal bir sohbet aracılığıyla kullanıcıya özel kitap önerileri sunmayı amaçlayan bir üretici yapay zeka sistemidir.

The system interacts with users via a chatbot interface, understands their reading preferences, summarizes them using a large language model (LLaMA3-70B), and recommends books based on semantic similarity.  
Sistem, bir sohbet arayüzü üzerinden kullanıcılarla etkileşime geçer, okuma tercihlerini anlar, bunları büyük dil modeli (LLaMA3-70B) ile özetler ve anlamsal benzerlik bazlı kitap önerileri sunar.

---

##  Technologies Used | Kullanılan Teknolojiler

- **Python (Flask API):** Backend development | Arka uç geliştirme
- **HTML + CSS + Vue.js:** Frontend interface | Ön yüz arayüzü
- **Groq API (LLaMA3-70B):** Natural language understanding | Doğal dil anlama
- **SBERT (Sentence-BERT):** Text vectorization for similarity matching | Benzerlik eşleştirme için metin vektörleştirme
- **Kaggle Dataset:** Book metadata | Kitap veriseti

Dataset link: [Goodreads Best Books Ever](https://www.kaggle.com/datasets/arnabchaki/goodreads-best-books-ever)

---

##  System Architecture | Sistem Mimarisi

- **Chat Interface (Vue.js):** Users communicate via a web-based chat.  
  Kullanıcılar web tabanlı sohbet arayüzü ile iletişim kurar.

- **Flask Backend:** Handles API routing and connects frontend to LLM.  
  API yönlendirmesini yapar ve ön yüzü büyük dil modeline bağlar.

- **LLM (LLaMA3-70B):** Understands and summarizes user preferences.  
  Kullanıcı tercihlerini anlar ve özetler.

- **SBERT Embeddings:** Used for content-based recommendation.  
  İçeriğe dayalı öneri için kullanılır.

- **Book Dataset:** Over 50,000 books with titles, genres, descriptions, ratings.  
  50.000'den fazla kitap, başlık, tür, açıklama ve puan bilgisi içerir.

---

##  How It Works | Nasıl Çalışır

1. User starts a chat session and talks about reading preferences.  
   Kullanıcı sohbet başlatır ve okuma tercihlerini belirtir.

2. Preferences are summarized using an LLM (LLaMA3).  
   Tercihler büyük dil modeli ile özetlenir.

3. The system finds the most semantically similar books using SBERT.  
   SBERT ile anlamsal olarak en yakın kitaplar bulunur.

4. Top 5 books are recommended based on content and rating.  
   İçerik ve puana göre en uygun 5 kitap önerilir.

---

##  Installation & Run | Kurulum ve Çalıştırma

```bash
# Install dependencies | Gerekli paketleri yükleyin
pip install flask
pip install flask-cors
pip install pandas
pip install numpy
pip install sentence-transformers
pip install scikit-learn
pip install requests


# Run backend | Arka ucu çalıştırın
python app.py

# Open frontend (open index.html in browser) | Ön yüzü açın (index.html dosyasını tarayıcıda açın)
