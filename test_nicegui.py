from nicegui import ui

@ui.page('/')
def index():
    ui.label('¡NiceGUI está funcionando correctamente!')
    ui.button('Haz clic', on_click=lambda: ui.notify('¡Botón presionado!'))

if __name__ == '__main__':
    print("Iniciando servidor NiceGUI de prueba...")
    print("Abre tu navegador y visita: http://localhost:8080")
    ui.run(port=8080) 