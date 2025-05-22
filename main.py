import cv2
import numpy as np
import time

def mostrar_mensaje(frame, texto):
    cv2.putText(frame, texto, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frame

def detectar_color_dominante(frame, size_region=20, dibujar=True):
    altura, ancho, _ = frame.shape
    x_inicio = ancho // 2 - size_region // 2
    y_inicio = altura // 2 - size_region // 2
    region = frame[y_inicio:y_inicio+size_region, x_inicio:x_inicio+size_region]

    if dibujar:
        cv2.rectangle(frame, (x_inicio, y_inicio), (x_inicio+size_region, y_inicio+size_region), (0, 255, 0), 2)
        cv2.imshow("Coloca la capa en el recuadro", frame)
        cv2.waitKey(2000)  # Muestra por 2 segundos

    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h_mean = int(np.mean(hsv_region[:,:,0]))
    s_mean = int(np.mean(hsv_region[:,:,1]))
    v_mean = int(np.mean(hsv_region[:,:,2]))

    print(f"Color detectado (HSV): H={h_mean}, S={s_mean}, V={v_mean}")

    h_margin = 10
    s_margin = 60
    v_margin = 60

    lower = np.array([max(0, h_mean - h_margin), max(0, s_mean - s_margin), max(0, v_mean - v_margin)])
    upper = np.array([min(179, h_mean + h_margin), min(255, s_mean + s_margin), min(255, v_mean + v_margin)])

    return lower, upper


def crear_fondo(captura, nºframes=30):
    print("Capturando el fondo, por favor, no te pongas en medio")
    fondos = []
    for f in range(nºframes):
        ret, frame = captura.read()
        if ret:
            fondos.append(frame)
            cv2.imshow("Capturando fondo...", frame)
            cv2.waitKey(1)
        else:
            print(f"No se pudo guardar el frame {f+1}/{nºframes}")
        time.sleep(0.1)
    cv2.destroyWindow("Capturando fondo...")
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
    principal = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fondoBG = cv2.bitwise_and(fondo, fondo, mask=mask)
    return cv2.add(principal, fondoBG)


def main():
    print(f"Tu versión de OpenCV es: {cv2.__version__} ")

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("Error: No se ha podido abrir la cámara")
        return

    fondo = None
    lower_color = None
    upper_color = None

    # Paso 1: Esperar a que el usuario pulse 'c' para capturar fondo
    while True:
        ret, frame = captura.read()
        if not ret:
            continue

        frame_mensaje = mostrar_mensaje(frame.copy(), "Prepárate para capturar el fondo (pulsa 'c')")
        cv2.imshow("Configuración", frame_mensaje)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            fondo = crear_fondo(captura)
            break
        elif key == ord('q'):
            captura.release()
            cv2.destroyAllWindows()
            return

    # Paso 2: Fondo capturado, ahora esperar para capturar el color
    while True:
        ret, frame = captura.read()
        if not ret:
            continue

        frame_mensaje = mostrar_mensaje(frame.copy(), "Fondo capturado. Pulsa 'x' para capturar el objeto invisible")
        cv2.imshow("Configuración", frame_mensaje)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('x'):
            # Mostrar cuadro grande y capturar color
            lower_color, upper_color = detectar_color_dominante(frame, size_region=60, dibujar=True)
            break
        elif key == ord('q'):
            captura.release()
            cv2.destroyAllWindows()
            return

    # Paso 3: Todo listo
    print("Todo listo, la capa ya funciona.")
    while True:
        ret, frame = captura.read()
        if not ret:
            continue

        mask = crear_mascara(frame, lower_color, upper_color)
        resultado = aplicar_efecto(frame, mask, fondo)

        resultado = mostrar_mensaje(resultado, "Presiona 'q' para salir.")
        cv2.imshow("Capa de la invisibilidad!", resultado)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
