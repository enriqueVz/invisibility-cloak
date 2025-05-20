import cv2
import numpy as np
import time

def crear_fondo(captura, nºframes=30):
    print("Capturando el fondo, por favor, no te pongas en medio")
    fondos = []
    for f in range(nºframes):
        ret, frame = captura.read()
        if ret:
            fondos.append(frame)
        else:
            print(f"Cuidado, no se ha podido guardar el frame {f+1}/{nºframes}")
        time.sleep(0.1)
    if fondos:
        return np.median(fondos, axis=0).astype(np.uint8)
    else:
        raise ValueError("No se pudo capturar ningún frame del fondo")

def crear_mascara(frame, lower_color, upper_color):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations = 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8), iterations = 1)
    return mask

def aplicar_efecto(frame, mask, fondo):
    mask_inv = cv2.bitwise_not(mask)
    principal = cv2.bitwise_and(frame, frame, mask = mask_inv)
    fondoBG = cv2.bitwise_and(frame, frame, mask = mask)
    return cv2.add(principal, fondoBG)

def main():
    print(f"Tu versión de OpenCV es: {cv2.__version__} ")

    captura = cv2.VideoCapture(0)

    if not captura.isOpened():
        print("Error: No se ha podido abrir la camara")
        return
    
    try:
        fondo = crear_fondo(captura)
    except ValueError as e:
        print(f"Error {e}")
        captura.release()
        return

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    print("Comenzando el loop, para salir, pulsa 'q'.")
    while True:
        ret, frame = captura.read()
        if not ret:
            print("Error: no se ha podido leer el frame")
            time.sleep(1)
            continue

        mask = crear_mascara(frame, lower_blue, upper_blue)
        resultado = aplicar_efecto(frame, mask, fondo)

        cv2.imshow("Capa de la invisibilidad!", resultado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    captura.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()