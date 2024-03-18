extends Node
## Handles the zoom in/out logic.

@onready var mandelbrot_texture: Sprite2D = $"../MandelbrotTexture"

@onready var zoom_in_sfx: AudioStreamPlayer = $ZoomIn
@onready var zoom_out_sfx: AudioStreamPlayer = $ZoomOut


func zoom_in(multiplier := 1.5):
	mandelbrot_texture.update_texture_smooth(
		mandelbrot_texture.render_xpos,
		mandelbrot_texture.render_ypos,
		min(mandelbrot_texture.render_zoom * multiplier, 100000)
	)
	zoom_in_sfx.play()


func zoom_out(multiplier := 1.5):
	mandelbrot_texture.update_texture_smooth(
		mandelbrot_texture.render_xpos,
		mandelbrot_texture.render_ypos,
		max(mandelbrot_texture.render_zoom / multiplier, 0.25)
	)
	zoom_out_sfx.play()
