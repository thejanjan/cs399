extends Node
## Handles the zoom in/out logic.

const MSEC_BETWEEN_ZOOM := 550


@onready var mandelbrot_texture: Sprite2D = $"../MandelbrotTexture"

@onready var zoom_in_sfx: AudioStreamPlayer = $ZoomIn
@onready var zoom_out_sfx: AudioStreamPlayer = $ZoomOut

var last_zoom := 0.0


func zoom_in(screen_pos: Vector2, multiplier := 1.5) -> bool:
	if attempt_zoom():
		mandelbrot_texture.update_texture_smooth_screen_space(
			screen_pos, min(mandelbrot_texture.render_zoom * multiplier, 100000)
		)
		zoom_in_sfx.play()
		return true
	return false


func zoom_out(screen_pos: Vector2, multiplier := 1.5) -> bool:
	if attempt_zoom():
		mandelbrot_texture.update_texture_smooth_screen_space(
			screen_pos, max(mandelbrot_texture.render_zoom / multiplier, 0.25)
		)
		zoom_out_sfx.play()
		return true
	return false


func recenter(screen_pos: Vector2) -> bool:
	if attempt_zoom():
		mandelbrot_texture.update_texture_smooth_screen_space(
			screen_pos, mandelbrot_texture.render_zoom
		)
		return true
	return false


func attempt_zoom() -> bool:
	return not mandelbrot_texture.mid_transition
