


# ================================================================
# MENÚ PRINCIPAL
# ================================================================

def menu_principal():
    calc = Calculadora()
    conv = Conversor()
    gen = GeneradorContrasenas()

    while True:
        print("\n===============================")
        print("🌟 MENÚ PRINCIPAL 🌟")
        print("===============================")
        print("1. Calculadora científica")
        print("2. Conversor de unidades")
        print("3. Generador de contraseñas")
        print("4. Salir")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            ejecutar_calculadora(calc)
        elif opcion == "2":
            ejecutar_conversor(conv)
        elif opcion == "3":
            ejecutar_generador(gen)
        elif opcion == "4":
            print("\n👋 ¡Gracias por usar el programa multifunción 2!")
            break
        else:
            print("⚠️ Opción no válida.")

        time.sleep(1)


# ================================================================
# PUNTO DE ENTRADA
# ================================================================

if __name__ == "__main__":
    menu_principal()

# ================================================================
# Fin del programa (más de 300 líneas)
# ================================================================
