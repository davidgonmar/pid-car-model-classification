from nicegui import ui

def main():
    with ui.card():
        ui.label('Aplicación de prueba de NiceGUI').classes('text-2xl')
        ui.button('Haz clic aquí', on_click=lambda: ui.notify('¡Funciona!'))

ui.run(port=8080) 