# Türkçe adres çözümleme ve eşleştirme deneyleri

[English README](README.md) · [Teknik deney rehberi](docs/RESEARCH_GUIDE.md) ·
[Notebook kaynak envanteri](docs/EXPERIMENTS.md)

Bu depo Teknofest/Hepsiburada adres çözümleme çalışmasında denenen kural,
TF-IDF, transformer/benzerlik ve GIS yaklaşımlarını koruyan bir yarışma/araştırma
arşividir. Üretim servisi veya tek komutla çalışan ortak bir ayrıştırıcı değildir.

## Problem

İki alt görev vardır: serbest biçimli adresi il/ilçe/mahalle ve sokak gibi
bileşenlerle normalleştirmek; adres üzerinden göreve özel `label/cluster`
tahmini yapmak. Her deney iki görevi birden çözmez.

Türkçe karakterler (`İ/ı`, `ğ/ş/ü/ö/ç`), `Mah./Mh.`, `Cad./Cd.`, `Sok./Sk.`
kısaltmaları, eksik alanlar ve karışık sıralama eşleşmeyi zorlaştırır. Aynı mahalle
ve sokak adının farklı illerde bulunması coğrafi bağlam gerektirir. İdari
sınırların değişmesi de kullanılan sözlük/veri sürümünü önemli hale getirir.

## Yaklaşımlar ve veri

- Kural tabanlı deneyler: temizlik, kısaltmalar, idari sözlükler ve sezgisel eşleme.
- TF-IDF deneyleri: seyrek özellikler; benzerlik, sınıflandırıcı veya sinir ağı varyantları.
- Transformer deneyleri: gömmeler; FAISS, kosinüs/fuzzy karşılaştırma ve bazı
  varyantlarda ağaç tabanlı modeller.
- GIS deneyleri: geokodlama ve coğrafi komşuluk ya da idari sınırlarla spatial join.

Tam 25 klasörlük indeks ve kaynakta görülen model farkları
[teknik rehberde](docs/RESEARCH_GUIDE.md) yer alır. Klasör adları doğrulanmış model
listesi veya başarı sırası değildir.

Yaygın şema `train.csv: address,label` ve `test.csv: id,address` biçimindedir;
eğitim dosyası depoda yoktur. GIS akışı ek sınır verisi gerektirebilir. Ortak
bağımlılık listesi sürümlere sabitlenmemiştir; tüm deneyler için doğrulanmış tek
bir ortam bulunmaz. İngilizce README'deki ortam hazırlığı ve deney seçimi
adımlarını izleyin; notebook içindeki Colab/yerel yollarını ayrıca kontrol edin.

## Değerlendirme ve sınırlamalar

Accuracy/Macro-F1, bileşen doğruluğu, idari seviye/kısaltma hataları ve bellek/hız
ölçümleri ortak koşullarda raporlanmalıdır. Aynı veya çok benzer adreslerin eğitim
ve doğrulama kümelerine dağılması veri sızıntısı oluşturabilir. Bu denetimde ortak
benchmark çalıştırılmadı; tarihsel notebook çıktıları güncel başarı iddiası değildir.

`baseline-v2/test_pipeline.py` eksik `untitled13` modülünü içe aktarıyor. İki
notebook'ta sözdizimi sorunu saptandı; ayrıntıları ana README'de bulunur. Ağır ML,
GPU ve geokodlama çalışmaları yürütülmedi.

## Veri hakları

Mevcut `data/test.csv` 217.241 adres satırı içerir. Yayımlama hakkı ve kişisel veri
durumu doğrulanmış değildir. Public depoda bulunması izin kanıtı sayılmaz.
Yeni gerçek adresler eklemeyin veya izinsiz geokod servisine göndermeyin;
notebook çıktıları, önbellek ve geçmiş de ayrıca incelenmelidir. Mevcut MIT
[kod lisansı](LICENSE), veri kümesinin paylaşım hakkını tek başına kanıtlamaz.
