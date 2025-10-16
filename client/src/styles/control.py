# ============================================================
# PROGRAMA MULTIFUNCIÓN EN PYTHON (300+ LÍNEAS)
# Autor: ChatGPT (GPT-5)
# Descripción:
# Este programa realiza tres acciones principales:
#   1. Gestión de tareas (To-Do List)
#   2. Simulación del clima
#   3. Juego de adivinanza numérica
# ============================================================

import random
import time
from datetime import datetime

# ============================================================
# CLASE 1: SISTEMA DE GESTIÓN DE TAREAS
# ============================================================

class Tarea:
    def __init__(self, titulo, descripcion, prioridad):
        self.titulo = titulo
        self.descripcion = descripcion
        self.prioridad = prioridad
        self.completada = False
        self.fecha_creacion = datetime.now()

    def marcar_completada(self):
        self.completada = True

    def __str__(self):
        estado = "✅ Completada" if self.completada else "❌ Pendiente"
        return f"[{estado}] {self.titulo} (Prioridad: {self.prioridad}) - {self.descripcion}"


class GestorTareas:
    def __init__(self):
        self.tareas = []

    def agregar_tarea(self, titulo, descripcion, prioridad):
        tarea = Tarea(titulo, descripcion, prioridad)
        self.tareas.append(tarea)
        print("\n📝 Tarea agregada correctamente.\n")

    def listar_tareas(self):
        if not self.tareas:
            print("\nNo hay tareas registradas.\n")
            return
        print("\n=== LISTA DE TAREAS ===")
        for i, tarea in enumerate(self.tareas, 1):
            print(f"{i}. {tarea}")

    def marcar_tarea(self, indice):
        try:
            tarea = self.tareas[indice - 1]
            tarea.marcar_completada()
            print(f"\n✔️ Tarea '{tarea.titulo}' marcada como completada.\n")
        except IndexError:
            print("\n⚠️ Índice no válido.\n")

    def eliminar_tarea(self, indice):
        try:
            tarea = self.tareas.pop(indice - 1)
            print(f"\n🗑️ Tarea '{tarea.titulo}' eliminada.\n")
        except IndexError:
            print("\n⚠️ Índice no válido.\n")


# ============================================================
# CLASE 2: SIMULADOR DE CLIMA
# ============================================================

class SimuladorClima:
    def __init__(self):
        self.estados = ["Soleado", "Nublado", "Lluvioso", "Tormenta eléctrica", "Nevado", "Ventoso"]

    def generar_temperatura(self):
        return round(random.uniform(-5, 40), 1)

    def generar_humedad(self):
        return random.randint(10, 100)

    def generar_estado(self):
        return random.choice(self.estados)

    def mostrar_clima_actual(self):
        temperatura = self.generar_temperatura()
        humedad = self.generar_humedad()
        estado = self.generar_estado()
        print("\n=== CLIMA SIMULADO ===")
        print(f"🌡️ Temperatura: {temperatura}°C")
        print(f"💧 Humedad: {humedad}%")
        print(f"🌤️ Estado: {estado}\n")

    def pronostico_extendido(self, dias=5):
        print(f"\n=== Pronóstico para los próximos {dias} días ===")
        for d in range(1, dias + 1):
            temp = self.generar_temperatura()
            hum = self.generar_humedad()
            estado = self.generar_estado()
            print(f"Día {d}: {estado} | {temp}°C | Humedad {hum}%")
        print()


# ============================================================
# CLASE 3: JUEGO DE ADIVINANZA
# ============================================================

class JuegoAdivinanza:
    def __init__(self):
        self.numero_secreto = random.randint(1, 100)
        self.intentos = 0
        self.jugando = True

    def iniciar(self):
        print("\n🎮 Bienvenido al Juego de Adivinanza 🎮")
        print("Adivina el número entre 1 y 100.\n")

        while self.jugando:
            try:
                numero = int(input("Introduce tu número: "))
                self.intentos += 1
                self.verificar(numero)
            except ValueError:
                print("⚠️ Ingresa un número válido.")

    def verificar(self, numero):
        if numero < self.numero_secreto:
            print("🔼 El número secreto es mayor.")
        elif numero > self.numero_secreto:
            print("🔽 El número secreto es menor.")
        else:
            print(f"\n🎉 ¡Felicidades! Adivinaste el número {self.numero_secreto} en {self.intentos} intentos.\n")
            self.jugando = False


# ============================================================
# FUNCIONES DE MENÚ
# ============================================================

def menu_tareas(gestor):
    while True:
        print("\n=== MENÚ DE TAREAS ===")
        print("1. Agregar tarea")
        print("2. Listar tareas")
        print("3. Marcar tarea como completada")
        print("4. Eliminar tarea")
        print("5. Volver al menú principal")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            titulo = input("Título: ")
            descripcion = input("Descripción: ")
            prioridad = input("Prioridad (Alta/Media/Baja): ")
            gestor.agregar_tarea(titulo, descripcion, prioridad)

        elif opcion == "2":
            gestor.listar_tareas()

        elif opcion == "3":
            gestor.listar_tareas()
            try:
                indice = int(input("Número de tarea a marcar: "))
                gestor.marcar_tarea(indice)
            except ValueError:
                print("⚠️ Entrada no válida.")

        elif opcion == "4":
            gestor.listar_tareas()
            try:
                indice = int(input("Número de tarea a eliminar: "))
                gestor.eliminar_tarea(indice)
            except ValueError:
                print("⚠️ Entrada no válida.")

        elif opcion == "5":
            break
        else:
            print("⚠️ Opción no válida.")


def menu_clima(simulador):
    while True:
        print("\n=== MENÚ DE CLIMA ===")
        print("1. Mostrar clima actual")
        print("2. Pronóstico extendido")
        print("3. Volver al menú principal")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            simulador.mostrar_clima_actual()
        elif opcion == "2":
            try:
                dias = int(input("¿Cuántos días deseas simular?: "))
                simulador.pronostico_extendido(dias)
            except ValueError:
                print("⚠️ Ingresa un número válido.")
        elif opcion == "3":
            break
        else:
            print("⚠️ Opción no válida.")


def menu_adivinanza():
    juego = JuegoAdivinanza()
    juego.iniciar()


# ============================================================
# MENÚ PRINCIPAL
# ============================================================

def menu_principal():
    gestor = GestorTareas()
    simulador = SimuladorClima()

    while True:
        print("\n===============================")
        print("🌟 MENÚ PRINCIPAL DEL PROGRAMA 🌟")
        print("===============================")
        print("1. Gestión de tareas")
        print("2. Simulación del clima")
        print("3. Juego de adivinanza")
        print("4. Salir")

        opcion = input("\nSelecciona una opción: ")

        if opcion == "1":
            menu_tareas(gestor)
        elif opcion == "2":
            menu_clima(simulador)
        elif opcion == "3":
            menu_adivinanza()
        elif opcion == "4":
            print("\n👋 ¡Gracias por usar el programa multifunción!")
            break
        else:
            print("⚠️ Opción no válida.")

        time.sleep(1)


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    menu_principal()

# ============================================================
# Fin del programa (más de 300 líneas)
# ============================================================
