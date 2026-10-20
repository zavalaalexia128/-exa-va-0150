from pathlib import Path
import csv
import time

import cv2
import numpy as np

# Print de identificacion al inicio
print("=" * 55)
print("ALUMNO: Zavala Medina | NC: 0150")
print("=" * 55)
print()


# ============================================
# 1. CONFIGURACION DEL PROYECTO
# ============================================

# Se usa parent.parent si el script esta dentro de codigopy-0150
BASE_DIR = Path(__file__).resolve().parent.parent

CARPETA_RESULTADOS = BASE_DIR / "resultados-img-0150"
CARPETA_FOTOGRAMAS = CARPETA_RESULTADOS / "fotogramas"

# Crear directorios asegurando la existencia sin fallar si ya existen
if not CARPETA_RESULTADOS.exists():
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

if not CARPETA_FOTOGRAMAS.exists():
    CARPETA_FOTOGRAMAS.mkdir(parents=True, exist_ok=True)

VIDEO_ENTRADA = (
    BASE_DIR / "video-original-0150" / "video-gato-0150.mp4"
)

ARCHIVO_CSV = CARPETA_RESULTADOS / "catalogo-fotogramas-0150.csv"
ARCHIVO_INFORME = CARPETA_RESULTADOS / "informe-video-0150.txt"


# ============================================
# 2. ABRIR EL VIDEO
# ============================================

if not VIDEO_ENTRADA.exists():
    raise FileNotFoundError(
        f"No se encontro el video: {VIDEO_ENTRADA}\n"
        "Coloca video-gato-0150.mp4 en video-original-0150."
    )

video = cv2.VideoCapture(str(VIDEO_ENTRADA))

if not video.isOpened():
    raise RuntimeError(
        "OpenCV no pudo abrir el video. Verifica el archivo MP4."
    )

ancho_original = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
alto_original = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_original = video.get(cv2.CAP_PROP_FPS)
total_estimado = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

if fps_original <= 0:
    fps_original = 25.0

duracion_estimada = total_estimado / fps_original

print("=" * 55)
print("PRACTICA: EXTRACCION Y PROCESAMIENTO DE FOTOGRAMAS")
print("=" * 55)
print(f"Video: {VIDEO_ENTRADA.name}")
print(f"Resolucion original: {ancho_original} x {alto_original}")
print(f"FPS originales: {fps_original:.2f}")
print(f"Fotogramas estimados: {total_estimado}")
print(f"Duracion estimada: {duracion_estimada:.2f} segundos")
print()


# ============================================
# 3. VARIABLES DE PROCESAMIENTO
# ============================================

ANCHO_MAXIMO = 480
INTERVALO_GUARDADO = 2.0

indice_fotograma = 0
fotogramas_guardados = 0
siguiente_guardado = 0.0

tiempo_procesamiento_total = 0.0
registros = []

inicio_programa = time.perf_counter()


# ============================================
# 4. LEER Y PROCESAR FOTOGRAMAS
# ============================================

try:
    while True:
        inicio_ciclo = time.perf_counter()

        correcto, fotograma = video.read()

        if not correcto:
            break

        indice_fotograma += 1

        # Tiempo aproximado del fotograma dentro del video.
        tiempo_video = (indice_fotograma - 1) / fps_original

        # Reducir resolucion para mejorar el rendimiento.
        alto, ancho = fotograma.shape[:2]

        if ancho > ANCHO_MAXIMO:
            nuevo_alto = int(alto * ANCHO_MAXIMO / ancho)
            fotograma = cv2.resize(
                fotograma,
                (ANCHO_MAXIMO, nuevo_alto),
                interpolation=cv2.INTER_AREA
            )

        alto_procesado, ancho_procesado = fotograma.shape[:2]

        # Convertir a escala de grises.
        gris = cv2.cvtColor(fotograma, cv2.COLOR_BGR2GRAY)

        # Reducir ruido mediante desenfoque gaussiano.
        suavizada = cv2.GaussianBlur(gris, (5, 5), 0)

        # Detectar bordes mediante Canny.
        bordes = cv2.Canny(suavizada, 80, 160)

        # Guardar un fotograma cada dos segundos.
        if tiempo_video >= siguiente_guardado:
            nombre = (
                f"fotograma_{fotogramas_guardados + 1:04d}_"
                f"{tiempo_video:.1f}s.jpg"
            )

            ruta_fotograma = CARPETA_FOTOGRAMAS / nombre

            if cv2.imwrite(str(ruta_fotograma), fotograma):
                fotogramas_guardados += 1

                registros.append({
                    "fotograma_video": indice_fotograma,
                    "tiempo_segundos": round(tiempo_video, 3),
                    "archivo": nombre,
                    "ancho_px": ancho_procesado,
                    "alto_px": alto_procesado,
                    "ruta": str(
                        ruta_fotograma.relative_to(BASE_DIR)
                    )
                })

                print(
                    f"Guardado: {nombre} | "
                    f"tiempo: {tiempo_video:.2f} s"
                )

            siguiente_guardado += INTERVALO_GUARDADO

        # Medir tiempo dedicado al procesamiento.
        fin_procesamiento = time.perf_counter()

        tiempo_ms = (
            fin_procesamiento - inicio_ciclo
        ) * 1000

        tiempo_procesamiento_total += tiempo_ms

        fps_procesamiento = (
            1000 / tiempo_ms if tiempo_ms > 0 else 0
        )

        # Mostrar informacion en la imagen.
        cv2.putText(
            fotograma,
            f"Frame: {indice_fotograma}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            fotograma,
            f"Procesamiento: {tiempo_ms:.1f} ms",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

        cv2.putText(
            fotograma,
            f"FPS procesamiento: {fps_procesamiento:.1f}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1
        )

        # Convertir bordes a tres canales para mostrar junto al video.
        bordes_color = cv2.cvtColor(
            bordes, cv2.COLOR_GRAY2BGR
        )

        comparacion = np.hstack((fotograma, bordes_color))

        cv2.imshow(
            "Video original reducido | Bordes Canny",
            comparacion
        )

        # Ajustar la espera para aproximar la velocidad del video.
        tiempo_ciclo_ms = (
            time.perf_counter() - inicio_ciclo
        ) * 1000

        espera_ms = max(
            1, int(1000 / fps_original - tiempo_ciclo_ms)
        )

        tecla = cv2.waitKey(espera_ms) & 0xFF

        if tecla == ord("q"):
            print("Procesamiento detenido por el usuario.")
            break

finally:
    video.release()
    cv2.destroyAllWindows()


# ============================================
# 5. CREAR EL CATALOGO CSV
# ============================================

with ARCHIVO_CSV.open(
    "w", newline="", encoding="utf-8-sig"
) as archivo_csv:

    campos = [
        "fotograma_video",
        "tiempo_segundos",
        "archivo",
        "ancho_px",
        "alto_px",
        "ruta"
    ]

    escritor = csv.DictWriter(
        archivo_csv, fieldnames=campos
    )

    escritor.writeheader()
    escritor.writerows(registros)


# ============================================
# 6. GENERAR EL INFORME
# ============================================

tiempo_total = time.perf_counter() - inicio_programa

fps_promedio_procesamiento = (
    1000 / (tiempo_procesamiento_total / indice_fotograma)
    if indice_fotograma > 0 and tiempo_procesamiento_total > 0
    else 0
)

with ARCHIVO_INFORME.open("w", encoding="utf-8") as informe:
    informe.write("INFORME DE PROCESAMIENTO DE VIDEO\n")
    informe.write("=" * 45 + "\n")
    informe.write(f"Alumno: Zavala Medina | NC: 0150\n")
    informe.write(f"Video: {VIDEO_ENTRADA.name}\n")
    informe.write(
        f"Resolucion original: "
        f"{ancho_original} x {alto_original}\n"
    )
    informe.write(f"FPS originales: {fps_original:.2f}\n")
    informe.write(f"Duracion estimada: {duracion_estimada:.2f} s\n")
    informe.write(f"Fotogramas leidos: {indice_fotograma}\n")
    informe.write(f"Fotogramas guardados: {fotogramas_guardados}\n")
    informe.write(
        f"FPS promedio de procesamiento: "
        f"{fps_promedio_procesamiento:.2f}\n"
    )
    informe.write(f"Tiempo total de ejecucion: {tiempo_total:.2f} s\n")
    informe.write(f"Intervalo de guardado: {INTERVALO_GUARDADO} s\n")


# ============================================
# 7. MOSTRAR EL RESUMEN FINAL
# ============================================

print()
print("=" * 55)
print("PROCESAMIENTO FINALIZADO")
print("=" * 55)
print(f"Fotogramas leidos: {indice_fotograma}")
print(f"Fotogramas guardados: {fotogramas_guardados}")
print(f"FPS promedio de procesamiento: {fps_promedio_procesamiento:.2f}")
print(f"Tiempo total: {tiempo_total:.2f} segundos")
print(f"Catalogo CSV: {ARCHIVO_CSV}")
print(f"Informe: {ARCHIVO_INFORME}")
print(f"Fotogramas: {CARPETA_FOTOGRAMAS}")
print()
print("=" * 55)
print("ALUMNO: Zavala Medina | NC: 0150")
print("=" * 55)