import os
import time
import requests
import base64
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
imagenesCabecera=14
# Configurar Selenium con ChromeDriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--headless")  # No abrir ventana del navegador
driver = webdriver.Chrome(options=chrome_options)

def descargar_imagenes(desde_url, num_imagenes=50, directorio="imagenes_descargadas", salto=imagenesCabecera, step=2, salto_manual=None):
    if not os.path.exists(directorio):
        os.makedirs(directorio)

    driver.get(desde_url)
    time.sleep(2)

    # Desplazar para cargar más imágenes
    for _ in range(3):
        driver.find_element("tag name", "body").send_keys(Keys.END)
        time.sleep(2)

    # Extraer las imágenes
    soup = BeautifulSoup(driver.page_source, "html.parser")
    imagenes = soup.find_all("img")

    urls = []
    for img in imagenes:
        if img.get("src"):  
            urls.append(img["src"])
        elif img.get("data-src"):  
            urls.append(img["data-src"])

    # Aplicar el salto inicial de 13 imágenes
    urls = urls[salto:]

    # Descargar cada 2 imágenes (una sí, una no)
    urls = urls[::step]

    # Aplicar el salto manual si está definido
    if salto_manual and salto_manual < len(urls):
        print(f"⏩ Saltando la imagen en la posición {salto_manual}")
        del urls[salto_manual]

    # Limitar al número deseado de imágenes
    urls = urls[:num_imagenes]

    # Descargar imágenes
    for i, url in enumerate(urls):
        try:
            if url.startswith("data:image"):  # Si es una imagen en base64
                _, encoded_data = url.split(",", 1)  
                image_data = base64.b64decode(encoded_data)
                with open(f"{directorio}/imagen_{i}.png", "wb") as f:
                    f.write(image_data)
                print(f"✅ Imagen {i+1} guardada desde base64")
            else:  
                response = requests.get(url)
                image = Image.open(BytesIO(response.content))
                image.save(f"{directorio}/imagen_{i}.jpg")
                print(f"✅ Imagen {i+1} descargada: {url}")

        except Exception as e:
            print(f"❌ Error con {url}: {e}")

# 📌 Pega aquí la URL de Google Imágenes cuando la tengas
url_google_imagenes = "https://www.google.es/search?q=+SEAT+Le%C3%B3n+FR+1.5+TSI+150+CV.&rlz=1C2PNBB_enES1092ES1092&sca_esv=cd25324f189e683f&udm=2&biw=1536&bih=695&sxsrf=AHTn8zqh-0cfFgdcVwYP5t8amgjAUf29aA%3A1741765313080&ei=wTrRZ-zPBJ-mkdUPnfnFwAk&ved=0ahUKEwistJK2hYSMAxUfU6QEHZ18EZgQ4dUDCBE&uact=5&oq=+SEAT+Le%C3%B3n+FR+1.5+TSI+150+CV.&gs_lp=EgNpbWciHiBTRUFUIExlw7NuIEZSIDEuNSBUU0kgMTUwIENWLjIEEAAYHjIEEAAYHjIEEAAYHjIEEAAYHjIEEAAYHjIEEAAYHjIEEAAYHjIEEAAYHjIGEAAYCBgeMgYQABgIGB5I4xdQugVYrhRwAngAkAECmAFioAHIDqoBAjI1uAEDyAEA-AEB-AECmAIDoALIAagCCsICBxAjGCcYyQLCAgoQIxgnGMkCGOoCmAMMiAYBkgcDMi4xoAefpgE&sclient=img"
descargar_imagenes(url_google_imagenes, num_imagenes=50, salto=imagenesCabecera, step=2, salto_manual=20)

driver.quit()
