
"""
MUSTAFA YILDIRIM - 190905029 - GÖRÜNTÜ İŞLEME PROJE ÖDEVİ

"""

# Görüntü işleme için gerekli olan OpenCV ve hesaplamalar için gerekli olan numpy kütüphanesini yüklüyorum.

import cv2 
import numpy as np 

# Görüntüyü imread fonksiyonu ile getiriyorum.

img = cv2.imread("Pascal VOC Challenge 2012 Dataset/test/2008_002174.jpg")

genislik = img.shape[1] # Görüntünün genişlik değerini alıyorum.
yukseklik = img.shape[0] # Görüntünün yükseklik değerini alıyorum.

# Kendi oluşturduğum icerisinde kus yazan .names uzantılı dosyayı okuyorum ve nesneListesi'ne ekliyorum.

nesneListesi = []
with open("model/voc.names", "r") as f:
    nesneListesi = [cname.strip() for cname in f.readlines()]

# Nesne algılama modelimi kullanmak için oluşturmuş olduğum 
# .cfg uzantılı dosyamı ve eğittiğim .weights uzantılı dosyayı ekliyorum.

yolov3 = cv2.dnn.readNetFromDarknet("model/yolov3_training.cfg", "model/yolov3_training_last.weights")

# kus sınıfının indexini seçiyorum.

kusIndex = nesneListesi.index("kus")

# Modelin nesne algılama sonuçlarını elde etmek için çıkış katmanlarını cikisKatmanlari listesine ekliyorum.

katmanlar = yolov3.getLayerNames()
cikisKatmanlari = [katmanlar[i - 1] for i in yolov3.getUnconnectedOutLayers()]

# .img uzantılı görüntüyü modele uygun hale getiriyorum.
# Görüntünün pixel değerini 1 / 255.0 ölçeğinde normalize ediyorum.
# Görüntünün boyutunu 416x416 olacak şekilde ayarlıyorum.
# Görüntüyü RGB olarak kullanacağım için swapRB=True değerini veriyorum.
# Görüntüyü kırpmadan boyutunu düzeltmek için crop=False değerini veriyorum.

goruntu = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Oluşturduğum görüntü modelini YOLO modeline girdi olarak veriyorum.

yolov3.setInput(goruntu)

# Oluşan çıkışları cikislar değişkenine aktarıyorum.

cikislar = yolov3.forward(cikisKatmanlari)

# Tespit edilen kus nesnelerini saklamak için bir liste oluşturuyorum.

kuslar = []

# Tespit edilen kus nesnelerinin koordinatlarını alıyorum. 

for cikis in cikislar:
    for bul in cikis:
        sayi = bul[5:]
        tespitId = np.argmax(sayi)
        esikDegeri = sayi[tespitId]

        # Tespit edilen değer eşik değerinden yukarıda bir değerse bu nesnenin konum ve boyut bilgilerini gerçek pixel değerlerine dönüştürüyorum.
        
        if tespitId == kusIndex and esikDegeri > 0.9:
            
            # Genişlik ve yükselik değerlerini hesaplıyorum.
            
            _genislik = int(bul[0] * genislik)
            _yukseklik = int(bul[1] * yukseklik)
            gen = int(bul[2] * genislik)
            yuk = int(bul[3] * yukseklik)
            
            # Oluşturacağım dikdörtgenin sol üst köşesinin koordinatlarını hesaplıyorum.
            
            solX = int(_genislik - gen / 2)
            solY = int(_yukseklik - yuk / 2)

            # Tespit edilen nesnenin konum ve boyut bilgilerini kuslar listesine ekliyorum.
            kuslar.append([solX, solY, gen, yuk])

# Tespit edilen kus nesnelerini, konum ve boyut bilgilerini rectangle fonkiyonunda kullanarak 
# dikdörtgen içine alıyorum ve putText fonksiyonu kullanarak üzerine kus yazıyorum.
for kus in kuslar:
    solX, solY, gen, yuk = kus
    
    # Dikdörtgenin sol üst köşesinin koordinatlarını genişlik ve yükseklik değerleri ile toplayarak dikdörtgenin sağ alt köşesini hesaplıyorum.
    # Dikdörtgenin ve üzerindeki yazının rengini, yazının boyutunu, konumunu, kalınlığını ve yazı tipini ayarlıyorum.
    cv2.rectangle(img, (solX, solY), (solX + gen, solY + yuk), (0, 0, 255), 2)
    cv2.putText(img, nesneListesi[kusIndex], (solX, solY - 10), cv2.FORMATTER_FMT_NUMPY, 1, (0, 0, 255), 2)

# Sonuç görselini ekranda göstermek için imshow fonksiyonu ile sonuç ekranı oluşturuyorum.
while True:
    cv2.imshow("Sonuc Ekrani", img)
    # "x" tuşuna basıldığında döngüden çıkıp, pencerenin kapanmasını sağlıyorum.
    if cv2.waitKey(1) == ord("x"):  
        break

cv2.destroyAllWindows()


            







