import time
from ursina import Ursina, Entity, color, window

app = Ursina(borderless=False, fullscreen=False, title="Test Ursina")
window.color = color.rgb(22, 26, 36)
cube = Entity(model='cube', color=color.red, scale=(2,2,2))

for i in range(50):
    app.step()
    time.sleep(0.1)

app.userExit()
