"""
AgroVision IA - Laboratorio Botánico · Detección de Plagas en Tomate
Aplicación de escritorio: identifica enfermedades en hojas de tomate a partir de una foto.

Archivos necesarios en la MISMA carpeta que este script:
  - modelo_hojas.keras   (modelo entrenado con entrenar_modelo_colab.py)
  - etiquetas.json       (lista de clases, en el orden de salida del modelo)
  - fichas.json          (causa, síntomas, tratamiento y prevención por clase)
  - estilos.py           (colores y tema de la interfaz)

Instalación:  pip install tensorflow customtkinter pillow numpy
Ejecución:    python interfaz_ia.py

La IA decide la clase; el código solo muestra la ficha de esa clase.
El modelo ya incluye la normalización: se le pasan píxeles 0..255 (NO dividir entre 255).
"""
import json
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import Image

import estilos
from estilos import *  # noqa: F401,F403  (colores)

# ---------------------------------------------------------------- rutas
CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo_hojas.keras"
RUTA_ETIQUETAS = CARPETA / "etiquetas.json"
RUTA_FICHAS = CARPETA / "fichas.json"

TAM = 224                 # tamaño de entrada del modelo
UMBRAL_CONFIANZA = 0.70   # por debajo de esto se avisa que el resultado es dudoso

TEXTO_SIN_IMAGEN = "🔬  Seleccione una imagen de su laptop\no pegue una ruta local para analizar"
TEXTO_ESPERANDO = "Esperando muestra para análisis..."
TEXTO_FICHA_VACIA = ("Analice una muestra para ver aquí:\n\n"
                     "• Por qué aparece la enfermedad (causa)\n"
                     "• Cómo se cura o se controla (tratamiento)\n"
                     "• Cómo evitarla (prevención)")

ctk.set_appearance_mode(estilos.MODO_APARIENCIA)
ctk.set_default_color_theme(estilos.TEMA_BASE)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sistema de Laboratorio - Detección de Plagas en Tomate")
        self.geometry("1280x740")
        self.minsize(1100, 660)
        self.configure(fg_color=FONDO_VENTANA)

        self.modelo = None
        self.clases = []
        self.fichas = {}
        self.ruta_imagen = None
        self._img_pil = None            # foto original (para redibujarla al cambiar el tamaño)
        self._imagen_ctk = None
        self._img_vacia = ctk.CTkImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)), size=(1, 1))
        self._job_redibujo = None
        self._cuerpos = []              # textos de la ficha (para ajustar el ancho)

        self._construir_interfaz()
        self._ficha_vacia()
        self._cargar_recursos()

    # =========================================================== interfaz
    def _panel(self, parent, columna):
        marco = ctk.CTkFrame(parent, fg_color=FONDO_PANEL, border_width=2,
                             border_color=AZUL_BORDE, corner_radius=20)
        marco.grid(row=0, column=columna, sticky="nsew", padx=8, pady=0)
        return marco

    def _titulo_panel(self, p, texto):
        ctk.CTkLabel(p, text=texto, font=ctk.CTkFont(size=19, weight="bold"),
                     text_color=AZUL_TITULO).grid(row=0, column=0, pady=(16, 8))

    def _construir_interfaz(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self, text="🔬  Laboratorio Botánico · Detección de Plagas en Tomate  🍅",
            font=ctk.CTkFont(size=28, weight="bold"), text_color=TEXTO
        ).grid(row=0, column=0, pady=(14, 8))

        cols = ctk.CTkFrame(self, fg_color="transparent")
        cols.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        cols.grid_rowconfigure(0, weight=1)
        cols.grid_columnconfigure(0, weight=5)
        cols.grid_columnconfigure(1, weight=4)
        cols.grid_columnconfigure(2, weight=4, minsize=330)

        self._columna_muestra(self._panel(cols, 0))
        self._columna_ficha(self._panel(cols, 1))
        self._columna_panel(self._panel(cols, 2))

    # ---- columna 1: visualización de la muestra
    def _columna_muestra(self, p):
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(1, weight=1)
        self._titulo_panel(p, "Visualización de la Muestra")

        self.marco_imagen = ctk.CTkFrame(p, fg_color=FONDO_MUESTRA, border_width=1,
                                         border_color=BORDE_IMAGEN, corner_radius=10)
        self.marco_imagen.grid(row=1, column=0, padx=22, pady=(0, 22), sticky="nsew")
        self.lbl_imagen = ctk.CTkLabel(
            self.marco_imagen, text=TEXTO_SIN_IMAGEN, text_color=TEXTO_SUAVE,
            font=ctk.CTkFont(size=16), justify="center")
        self.lbl_imagen.place(relx=0.5, rely=0.5, anchor="center")

        # la foto se ajusta sola al tamaño del marco
        self.marco_imagen.bind("<Configure>", self._programar_redibujo)

        # interactivo: clic = explorar (sin foto) o ampliar (con foto); resalta al pasar el mouse
        try:
            self.lbl_imagen.configure(cursor="hand2")
        except Exception:  # noqa: BLE001
            pass
        for w in (self.marco_imagen, self.lbl_imagen):
            w.bind("<Button-1>", self._clic_imagen)
            w.bind("<Enter>", self._hover_imagen)
            w.bind("<Leave>", self._hover_imagen)

    # ---- columna 2: ficha de la enfermedad
    def _columna_ficha(self, p):
        p.grid_columnconfigure(0, weight=1)
        p.grid_rowconfigure(4, weight=1)
        self.col_ficha = p
        p.bind("<Configure>", self._ajustar_texto)
        self._titulo_panel(p, "📋  Ficha de la Enfermedad")

        self.lbl_nombre = ctk.CTkLabel(p, text="", text_color=TEXTO,
                                       font=ctk.CTkFont(size=17, weight="bold"))
        self.lbl_nombre.grid(row=1, column=0, padx=10)
        self.lbl_agente = ctk.CTkLabel(p, text="", text_color=TEXTO_SUAVE,
                                       font=ctk.CTkFont(size=12))
        self.lbl_agente.grid(row=2, column=0, pady=(0, 2))
        self.lbl_trat = ctk.CTkLabel(p, text="", font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_trat.grid(row=3, column=0, pady=(0, 6))

        self.caja_ficha = ctk.CTkScrollableFrame(
            p, fg_color=FONDO_CAJA, border_width=1, border_color=BORDE_SUAVE, corner_radius=10)
        self.caja_ficha.grid(row=4, column=0, padx=16, sticky="nsew")
        self.caja_ficha.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            p, text="⚠  Información orientativa. Consulte a un técnico agrónomo.",
            text_color=TEXTO_AVISO, font=ctk.CTkFont(size=11)
        ).grid(row=5, column=0, pady=(8, 14))

    # ---- columna 3: panel de carga y diagnóstico
    def _columna_panel(self, p):
        p.grid_columnconfigure(0, weight=1)
        self._titulo_panel(p, "💻  Panel de Carga y Diagnóstico")

        def caja(fila):
            f = ctk.CTkFrame(p, fg_color=FONDO_CAJA, border_width=1,
                             border_color=BORDE_SUAVE, corner_radius=10)
            f.grid(row=fila, column=0, padx=16, pady=6, sticky="ew")
            f.grid_columnconfigure(0, weight=1)
            return f

        fuente_titulo = ctk.CTkFont(size=13, weight="bold")
        fuente_boton = ctk.CTkFont(size=13, weight="bold")

        # Opción 1
        c1 = caja(1)
        ctk.CTkLabel(c1, text="Opción 1: Archivo local (Laptop)", text_color=TEXTO,
                     font=fuente_titulo).grid(row=0, column=0, pady=(10, 4))
        ctk.CTkButton(c1, text="💻  Explorar Archivos", height=38, corner_radius=10,
                      fg_color=AZUL_BOTON, hover_color=AZUL_BOTON_HOVER,
                      font=fuente_boton, command=self.explorar
                      ).grid(row=1, column=0, padx=14, pady=(0, 14), sticky="ew")

        # Opción 2
        c2 = caja(2)
        ctk.CTkLabel(c2, text="Opción 2: Pegar Ruta de Acceso", text_color=TEXTO,
                     font=fuente_titulo).grid(row=0, column=0, pady=(10, 4))
        self.entry_ruta = ctk.CTkEntry(
            c2, placeholder_text="C:\\ruta\\de\\la\\imagen.jpg", height=34,
            fg_color=FONDO_ENTRADA, border_color=BORDE_ENTRADA)
        self.entry_ruta.grid(row=1, column=0, padx=14, pady=(0, 8), sticky="ew")
        self.entry_ruta.bind("<Return>", lambda e: self.cargar_desde_ruta())
        ctk.CTkButton(c2, text="🔗  Cargar desde Ruta", height=38, corner_radius=10,
                      fg_color=AZUL_BOTON, hover_color=AZUL_BOTON_HOVER,
                      font=fuente_boton, command=self.cargar_desde_ruta
                      ).grid(row=2, column=0, padx=14, pady=(0, 14), sticky="ew")

        # Analizar
        self.btn_analizar = ctk.CTkButton(
            p, text="🔬  Analizar Muestra", height=46, corner_radius=12,
            fg_color=AZUL_ANALIZAR, hover_color=AZUL_ANALIZAR_HOVER,
            font=ctk.CTkFont(size=16, weight="bold"),
            state="disabled", command=self.analizar)
        self.btn_analizar.grid(row=3, column=0, padx=16, pady=10, sticky="ew")

        # Resultado + confianza
        res = caja(4)
        self.lbl_resultado = ctk.CTkLabel(
            res, text=TEXTO_ESPERANDO, text_color=TEXTO_SUAVE,
            font=ctk.CTkFont(size=15, weight="bold"), wraplength=290)
        self.lbl_resultado.grid(row=0, column=0, padx=10, pady=(12, 6))
        self.barra = ctk.CTkProgressBar(res, progress_color=BARRA_PROGRESO,
                                        fg_color=BARRA_FONDO, height=12)
        self.barra.set(0)
        self.barra.grid(row=1, column=0, padx=14, sticky="ew")
        self.lbl_confianza = ctk.CTkLabel(res, text="Confianza: 0%", text_color=AZUL_TITULO,
                                          font=ctk.CTkFont(size=13, weight="bold"))
        self.lbl_confianza.grid(row=2, column=0, pady=(6, 0))
        self.lbl_otras = ctk.CTkLabel(res, text="", text_color=TEXTO_SUAVE,
                                      font=ctk.CTkFont(size=11), wraplength=290)
        self.lbl_otras.grid(row=3, column=0, padx=10, pady=(0, 12))

        # Limpiar / Salir
        fila = ctk.CTkFrame(p, fg_color="transparent")
        fila.grid(row=5, column=0, padx=16, pady=(8, 8), sticky="ew")
        fila.grid_columnconfigure((0, 1), weight=1, uniform="b")
        ctk.CTkButton(fila, text="🧹  Limpiar", height=38, corner_radius=10,
                      fg_color=GRIS_BOTON, hover_color=GRIS_BOTON_HOVER,
                      font=fuente_boton, command=self.limpiar
                      ).grid(row=0, column=0, padx=(0, 6), sticky="ew")
        ctk.CTkButton(fila, text="✖  Salir", height=38, corner_radius=10,
                      fg_color=ROJO_BOTON, hover_color=ROJO_BOTON_HOVER,
                      font=fuente_boton, command=self.destroy
                      ).grid(row=0, column=1, padx=(6, 0), sticky="ew")

        self.lbl_estado = ctk.CTkLabel(p, text="", text_color=TEXTO_SUAVE,
                                       font=ctk.CTkFont(size=11), wraplength=320)
        self.lbl_estado.grid(row=6, column=0, padx=16, pady=(0, 10))

    # =========================================================== ficha
    def _ajustar_texto(self, _event=None):
        ancho = max(200, self.col_ficha.winfo_width() - 90)
        for lbl in self._cuerpos:
            lbl.configure(wraplength=ancho)

    def _vaciar_ficha(self):
        for w in self.caja_ficha.winfo_children():
            w.destroy()
        self._cuerpos = []

    def _agregar_texto(self, texto, color=TEXTO):
        c = ctk.CTkLabel(self.caja_ficha, text=texto, text_color=color, anchor="w",
                         justify="left", font=ctk.CTkFont(size=14))
        c.grid(sticky="ew", padx=10, pady=(12, 2))
        self._cuerpos.append(c)
        self._ajustar_texto()

    def _agregar_seccion(self, titulo, texto, color):
        t = ctk.CTkLabel(self.caja_ficha, text=titulo, text_color=color, anchor="w",
                         font=ctk.CTkFont(size=12, weight="bold"))
        t.grid(sticky="ew", padx=10, pady=(10, 0))
        c = ctk.CTkLabel(self.caja_ficha, text=texto, text_color=TEXTO, anchor="w",
                         justify="left", font=ctk.CTkFont(size=13))
        c.grid(sticky="ew", padx=10, pady=(0, 2))
        self._cuerpos.append(c)
        self._ajustar_texto()

    def _ficha_vacia(self, mensaje=TEXTO_FICHA_VACIA):
        self.lbl_nombre.configure(text="")
        self.lbl_agente.configure(text="")
        self.lbl_trat.configure(text="")
        self._vaciar_ficha()
        self._agregar_texto(mensaje)

    def _ficha_mostrar(self, ficha, nombre, baja_confianza):
        sana = bool(ficha.get("sana"))
        self.lbl_nombre.configure(text=nombre)
        self.lbl_agente.configure(text=ficha.get("agente", ""))
        if sana:
            self.lbl_trat.configure(text="✔ Planta saludable", text_color=VERDE)
        elif ficha.get("tieneTratamiento", True):
            self.lbl_trat.configure(text="✔ Tiene tratamiento", text_color=VERDE)
        else:
            self.lbl_trat.configure(text="✖ No tiene cura (solo manejo)", text_color=ROJO_BOTON)

        self._vaciar_ficha()
        if baja_confianza:
            self._agregar_seccion(
                "⚠ AVISO: CONFIANZA BAJA",
                "El resultado es dudoso. Prueba con otra foto (una sola hoja, bien "
                "iluminada y enfocada) o consulta a un especialista.", NARANJA)

        if sana:
            self._agregar_seccion("DIAGNÓSTICO", ficha.get("causa", ""), TEAL)
        else:
            self._agregar_seccion("¿POR QUÉ APARECE? (CAUSA)", ficha.get("causa", ""), NARANJA)
            self._agregar_seccion("SÍNTOMAS", ficha.get("sintomas", ""), TEAL)
        self._agregar_seccion("TRATAMIENTO (¿CÓMO SE CURA?)", ficha.get("tratamiento", ""), VERDE)
        self._agregar_seccion("PREVENCIÓN", ficha.get("prevencion", ""), MORADO)

    # =========================================================== carga inicial
    def _cargar_recursos(self):
        """Carga etiquetas, fichas y modelo sin congelar la ventana."""
        for ruta in (RUTA_MODELO, RUTA_ETIQUETAS, RUTA_FICHAS):
            if not ruta.exists():
                messagebox.showerror(
                    "Falta un archivo",
                    f"No se encontró:\n{ruta.name}\n\nDebe estar en la misma carpeta que interfaz_ia.py.")
                self.after(100, self.destroy)
                return
        self.clases = json.loads(RUTA_ETIQUETAS.read_text(encoding="utf-8"))
        self.fichas = json.loads(RUTA_FICHAS.read_text(encoding="utf-8"))
        self._estado_modelo()
        threading.Thread(target=self._hilo_modelo, daemon=True).start()

    def _hilo_modelo(self):
        try:
            from tensorflow.keras.models import load_model   # import lento: se hace aquí
            modelo = load_model(RUTA_MODELO)
            self.after(0, lambda: self._modelo_listo(modelo))
        except Exception as e:  # noqa: BLE001
            self.after(0, lambda: self._modelo_error(e))

    def _modelo_listo(self, modelo):
        self.modelo = modelo
        if self.ruta_imagen:
            self.lbl_estado.configure(text="✔ Listo para analizar", text_color=VERDE)
            self.btn_analizar.configure(state="normal")
        else:
            self._estado_modelo()

    def _modelo_error(self, e):
        self.lbl_estado.configure(text="✖ No se pudo cargar el modelo.", text_color=ROJO_BOTON)
        messagebox.showerror("Error", f"No se pudo cargar modelo_hojas.keras:\n{e}")

    def _estado_modelo(self):
        if self.modelo is None:
            self.lbl_estado.configure(text="Cargando modelo de IA...", text_color=NARANJA)
        else:
            self.lbl_estado.configure(text="✔ Modelo de IA listo", text_color=VERDE)

    # =========================================================== foto (vista previa)
    def _programar_redibujo(self, _event=None):
        if self._job_redibujo:
            self.after_cancel(self._job_redibujo)
        self._job_redibujo = self.after(60, self._redibujar_imagen)

    def _redibujar_imagen(self):
        self._job_redibujo = None
        if self._img_pil is None:
            return
        w = self.marco_imagen.winfo_width() - 24
        h = self.marco_imagen.winfo_height() - 24
        if w < 60 or h < 60:
            return
        vista = self._img_pil.copy()
        vista.thumbnail((w, h))
        self._imagen_ctk = ctk.CTkImage(light_image=vista, dark_image=vista, size=vista.size)
        self.lbl_imagen.configure(image=self._imagen_ctk, text="")

    def _hover_imagen(self, _event=None):
        m = self.marco_imagen
        x, y = m.winfo_pointerxy()
        dentro = (m.winfo_rootx() <= x < m.winfo_rootx() + m.winfo_width()
                  and m.winfo_rooty() <= y < m.winfo_rooty() + m.winfo_height())
        m.configure(border_color=AZUL_BORDE if dentro else BORDE_IMAGEN)

    def _clic_imagen(self, _event=None):
        if self.ruta_imagen:
            self.ampliar()
        else:
            self.explorar()

    def ampliar(self):
        if self._img_pil is None:
            return
        img = self._img_pil.copy()
        img.thumbnail((int(self.winfo_screenwidth() * 0.8), int(self.winfo_screenheight() * 0.8)))
        grande = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)

        ventana = ctk.CTkToplevel(self)
        ventana.title(Path(self.ruta_imagen).name)
        ventana.configure(fg_color=FONDO_VENTANA)
        ventana.transient(self)
        lbl = ctk.CTkLabel(ventana, image=grande, text="")
        lbl.image = grande          # evita que Python borre la imagen
        lbl.pack(padx=12, pady=(12, 4))
        ctk.CTkLabel(ventana, text="Clic en la imagen o Esc para cerrar",
                     text_color=TEXTO_SUAVE, font=ctk.CTkFont(size=11)).pack(pady=(0, 8))
        lbl.bind("<Button-1>", lambda e: ventana.destroy())
        ventana.bind("<Escape>", lambda e: ventana.destroy())
        ventana.after(150, ventana.focus)

    # =========================================================== acciones
    def explorar(self):
        ruta = filedialog.askopenfilename(
            title="Selecciona una foto de la hoja",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")])
        if ruta:
            self._abrir_imagen(ruta)

    def cargar_desde_ruta(self):
        ruta = self.entry_ruta.get().strip().strip('"').strip("'")
        if not ruta:
            messagebox.showinfo("Ruta vacía", "Pega primero la ruta de la imagen.")
            return
        if not Path(ruta).is_file():
            messagebox.showerror("Error", f"No se encontró el archivo:\n{ruta}")
            return
        self._abrir_imagen(ruta)

    def _abrir_imagen(self, ruta):
        try:
            img = Image.open(ruta).convert("RGB")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return
        self.ruta_imagen = ruta
        self._img_pil = img
        self._redibujar_imagen()
        self.entry_ruta.delete(0, "end")
        self.entry_ruta.insert(0, ruta)
        self._reiniciar_resultado()
        self._ficha_vacia("Imagen cargada.\n\nPresiona «Analizar Muestra» para ver el diagnóstico.")
        if self.modelo is not None:
            self.lbl_estado.configure(text="✔ Listo para analizar", text_color=VERDE)
            self.btn_analizar.configure(state="normal")
        else:
            self._estado_modelo()

    def _parar_barra(self):
        self.barra.stop()
        self.barra.configure(mode="determinate")
        self.barra.set(0)

    def _reiniciar_resultado(self):
        self.lbl_resultado.configure(text=TEXTO_ESPERANDO, text_color=TEXTO_SUAVE)
        self.lbl_confianza.configure(text="Confianza: 0%", text_color=AZUL_TITULO)
        self.lbl_otras.configure(text="")
        self._parar_barra()

    def limpiar(self):
        self.ruta_imagen = None
        self._img_pil = None
        self._imagen_ctk = None
        self.lbl_imagen.configure(image=self._img_vacia, text=TEXTO_SIN_IMAGEN)
        self.entry_ruta.delete(0, "end")
        self._reiniciar_resultado()
        self._ficha_vacia()
        self.btn_analizar.configure(state="disabled", text="🔬  Analizar Muestra")
        self._estado_modelo()

    def analizar(self):
        if self.modelo is None or not self.ruta_imagen:
            return
        self.btn_analizar.configure(state="disabled", text="Analizando...")
        self.lbl_estado.configure(text="Analizando la muestra...", text_color=NARANJA)
        self.barra.configure(mode="indeterminate")
        self.barra.start()
        threading.Thread(target=self._hilo_analisis, daemon=True).start()

    def _hilo_analisis(self):
        try:
            img = Image.open(self.ruta_imagen).convert("RGB").resize((TAM, TAM))
            x = np.expand_dims(np.asarray(img, dtype="float32"), axis=0)  # 0..255, sin /255
            probs = self.modelo.predict(x, verbose=0)[0]
            self.after(0, lambda: self._mostrar_resultado(probs))
        except Exception as e:  # noqa: BLE001
            self.after(0, lambda: self._error_analisis(e))

    def _error_analisis(self, e):
        self._parar_barra()
        self.lbl_estado.configure(text="✖ Falló el análisis.", text_color=ROJO_BOTON)
        self.btn_analizar.configure(state="normal", text="🔬  Analizar Muestra")
        messagebox.showerror("Error", f"No se pudo analizar la imagen:\n{e}")

    # =========================================================== resultados
    def _mostrar_resultado(self, probs):
        self._parar_barra()
        self.btn_analizar.configure(state="normal", text="🔬  Analizar Muestra")

        orden = np.argsort(probs)[::-1]
        i = int(orden[0])
        confianza = float(probs[i])
        clase = self.clases[i]
        ficha = self.fichas.get(clase, {})
        nombre = ficha.get("etiqueta", clase)
        baja = confianza < UMBRAL_CONFIANZA

        color = VERDE if ficha.get("sana") else NARANJA
        self.lbl_resultado.configure(text=f"Resultado: {nombre}", text_color=color)
        if baja:
            self.lbl_confianza.configure(
                text=f"Confianza: {confianza * 100:.1f}%  ⚠ baja", text_color=NARANJA)
        else:
            self.lbl_confianza.configure(
                text=f"Confianza: {confianza * 100:.1f}%", text_color=AZUL_TITULO)
        self.barra.set(confianza)

        otras = []
        for j in orden[1:3]:
            nom = self.fichas.get(self.clases[int(j)], {}).get("etiqueta", self.clases[int(j)])
            otras.append(f"{nom} {float(probs[j]) * 100:.1f}%")
        self.lbl_otras.configure(text="Otras posibilidades: " + " · ".join(otras))

        self._ficha_mostrar(ficha, nombre, baja)
        self.lbl_estado.configure(text="✔ Análisis terminado.", text_color=VERDE)


if __name__ == "__main__":
    App().mainloop()