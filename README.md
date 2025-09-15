# HLB-FI-6AxME

# Projektidee

Automatisches Garagentor (o.ä.): Embedded System mit Kamera erkennt Kennzeichen (und/oder RFID) und öffnet ggf. ein Garagentor / Schranken.

Bilderkennung z.B. mit OpenCV https://opencv.org/

## Benötigte Hardware (Variante 1: Raspberry Pi)

- Raspberry Pi (Version mit OpenCV kompatibel)
- Camera Modul für den Raspberry (USB oder Shield?)
- RFID Reader
- Ansteuerung von Aktoren (Servo? Motor?)
- Optional: Display

Links:
- https://opencv.org/blog/raspberry-pi-with-opencv/

## Benötigte Hardware (Variante 2: Microcontroller+Kamera - Applikation - Microcontroller+Aktor)

- Microcontroller (IoT fähig) + Kamera, z.B. ESP32-Cam
- Microcontroller (IoT fähig) + Aktoren, z.B. ESP32, Raspberry Pi Pico, ...
- RFID Reader
- WiFi
- "Server" für die Applikation
- Optional: Display

## Benötigte Software (beide Versionen)

- Bilderkennung: OpenCV + Python
- 
