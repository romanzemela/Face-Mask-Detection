# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from playsound import playsound
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import RPi.GPIO as GPIO #piny do diod led
from plyer import notification

def wykryj_i_oszacuj_maske(klatka, faceNet, maskNet):
	# Wyciągnięcie rozmiarów klatki i storzenie z nich obiektu BLOB
	# Obiekty BLOB sluza do przechowywania duzych obiektow multimedialnych, np. obrazow
	(h, w) = klatka.shape[:2]
	blob = cv2.dnn.blobFromImage(klatka, 1.0, (300, 300),
		(104.0, 177.0, 123.0)) #

	# Uzycie sieci "faceNet" by wykryc twarze znajdujace sie na ujeciu z kamery
	faceNet.setInput(blob)
	wykryte_twarze = faceNet.forward()

	# inicjalizacja tablicy twarzy, ich lokacji i prawdopodobienst na maski
	twarze = []
	locs = []
	prawdopodobienstwo_maski = []

	# iterowanie po wykrytych twarzach
	for i in range(0, wykryte_twarze.shape[2]):
		# pobranie wartosci prawdopodoienstw. Prawdopodobienstwo mowi nam jaka jest szansa, ze
		# algorytm sie nie pomylil
		prawdopodobienstwo = wykryte_twarze[0, 0, i, 2]

		# Usuniece twarzy ktore maja prawdopodobienstwo mniejsze od min
		if prawdopodobienstwo > args["prawdopodobienstwo"]:
			# obliczenie koordynatow twarzy na ujeciu
			box = wykryte_twarze[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# upewnienie sie, ze obliczone koordynaty znajduja sie na ujeciu.
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Wyciagniecie ROI, czyli interesujacego nas obszaru i zamienienie go
			# z BGR na RGB
			# Dodatkowo zmieniamy rozmiary wykrytej twarzy na 224x224
			twarz = klatka[startY:endY, startX:endX]
			twarz = cv2.cvtColor(twarz, cv2.COLOR_BGR2RGB)
			twarz = cv2.resize(twarz, (224, 224))
			twarz = img_to_array(twarz)
			twarz = preprocess_input(twarz)

			# Dodanie twarzy i jej koordynatow do odpowiednich tablic
			twarze.append(twarz)
			locs.append((startX, startY, endX, endY))

	# Do kolejnego etapu przechodzimy tylko, jezeli wykrylismy min 1 twarz
	if len(twarze) > 0:
		# Na wykrytych twarzach przeprowadzamy kolejne oszacowanie, tym razem
		# sprawdzamy, z jakim prawdopodobienstwem na twarzy znajduje sie maska
		twarze = np.array(twarze, dtype="float32")
		prawdopodobienstwo_maski = maskNet.predict(twarze, batch_size=32)
	
	Adafruit_NeoPixel.setPixelColor(i, LED.Color(0, 0, 0)); #Wygaszenie diody gdy nie ma twarzy w kadrze
	Adafruit_NeoPixel.show();

	# zwracamy krotke (twarz, lokacja)
	return (locs, prawdopodobienstwo_maski)
	
#-----------------------------------------------
#Tutaj znajduje sie kod zwiazany z dioda, prezentujaca wykrycie maseczki
#LED variables
#Przygotowanie diod
def LED_PIN D1
def PWR_pin D2
def LED_COUNT 1
#Funkcja z biblioteki <Adafruit_NeoPixel.h>
Adafruit_NeoPixel = LED(LED_COUNT, LED_PIN, NEO_GRB + NEO_KHZ800);

#-----------------------------------------------------

# Przygotowujemy zmienne i podajemy lokalizacje modeli do wykrywania twarzy i masek
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--twarz", type=str,
	default="wykrywacz_twarzy",
	help="sciezka do folderu z modelem do wykrywania twarzy")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="sciezka do folderu z modelem do wykrywania twarzy")
ap.add_argument("-c", "--prawdopodobienstwo", type=float, default=0.5,
	help="minimalne prawdopodobienstwo do filtrowania twarzy")
args = vars(ap.parse_args())

# Wczytanie modelu do wykrywania twarzy z dysku
print("[INFO] ladowanie modelu do wykrywania twarzy...")
prototxtPath = os.path.sep.join([args["twarz"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["twarz"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Wczytanie modelu do wykrywania maseczek z dysku
print("[INFO] ladowanie modelu do wykrywania maseczek...")
maskNet = load_model(args["model"])

# inicjalizowanie strumienia z kamery
print("[INFO] startowanie przesylania obrazu...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# iterowanie po klatkach z strumienia z kamery
while True:
	# Pobranie klatki obrazu i zmiana jego rozmiaru
	klatka = vs.read()
	klatka = imutils.resize(klatka, width=400)

	# Wykryj twarze na ujeciu i oszacuj czy sa w maseczc
	(locs, prawdopodobienstwo_maski) = wykryj_i_oszacuj_maske(klatka, faceNet, maskNet)

	# Petla iterujaca po wszystkich wykrytych lokacjach twarzy
	for (box, pred) in zip(locs, prawdopodobienstwo_maski):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(maska, bezMaski) = pred

		# Ustalenie etykiety i koloru obramowania wykrytych obiektow na obrazie
		label = "Maska" if maska > bezMaski else "Bez Maski"
		color = (0, 255, 0) if label == "Maska" else (0, 0, 255)
			
		# Dodanie prawdopodobienstwa do etykiety obiektu
		label = "{}: {:.2f}%".format(label, max(maska, bezMaski) * 100)

		# Wystwietlenie etykiety i obramowania na kadrze
		cv2.putText(klatka, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(klatka, (startX, startY), (endX, endY), color, 2)

		# Tutaj reagujemy na maske lub jej brak
		if maska < bezMaski:
			Adafruit_NeoPixel.setPixelColor(i, pixels.Color(0, 150, 0)); #Zapalenie diody na czerwono
		else 
		    Adafruit_NeoPixel.setPixelColor(i, pixels.Color(150, 0, 0)); #Zapalenie diody na zielono
		    
		Adafruit_NeoPixel.show();
    

	# wyswietlenie klatki
	cv2.imshow("klatka", klatka)
	key = cv2.waitKey(1) & 0xFF

	# Wylaczenie programu przyciskiem q
	if key == ord("q"):
		break

# "Sprzatanie", wylaczenie strumienia z kamery i zniszczenie wszystkich okien
cv2.destroyAllWindows()
vs.stop()

