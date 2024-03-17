extends Sprite2D
## The main texture control for rendering the Mandelbrot Set.

## Additional zoom-in multiplier on the image scale.
const ADDITIONAL_ZOOM := 1.1

## The number of colors in our palette.
const PALETTE_SIZE := 64

## How pixely our render is.
const IMAGE_PIXELIZE := 4.0

## How much we zoom in per step.
const ZOOM_STEP := 1.25

## The base number of iterations to render at the start scale.
const MAX_ITERATION_BASE = 32

## The width and height that the Mandelbrot calculations occur in.
const RENDER_WIDTH := 1.7
const RENDER_HEIGHT := RENDER_WIDTH * 0.75

## The texture width and height that gets produced.
var TEXTURE_WIDTH  := ceili(float(ProjectSettings.get_setting("display/window/size/viewport_width")) / IMAGE_PIXELIZE)
var TEXTURE_HEIGHT := ceili(float(ProjectSettings.get_setting("display/window/size/viewport_height")) / IMAGE_PIXELIZE)

## Our image builder node (calculates in Rust).
@onready var mandelbrot_image_builder: MandelbrotImageBuilder = $"../MandelbrotImageBuilder"

## Palette cache.
var _palette_reds: Array[int] = []
var _palette_greens: Array[int] = []
var _palette_blues: Array[int] = []

## The location to render at.
## These can be modified before rendering.
@export var render_xpos := 0.15  # -0.765
@export var render_ypos := 0.0
@export var render_zoom := 3.5  # 1.0
@export var rerender := false:
	set(x):
		update_texture_instant()

func _ready():
	# Build our color palette.
	for i in range(PALETTE_SIZE):
		var color := Color.from_hsv(float(i) / float(PALETTE_SIZE), 0.7, 0.7)
		_palette_reds.append(roundi(color.r * 255))
		_palette_greens.append(roundi(color.g * 255))
		_palette_blues.append(roundi(color.b * 255))
	
	# Set texture scale.
	scale = Vector2(IMAGE_PIXELIZE * ADDITIONAL_ZOOM, IMAGE_PIXELIZE * ADDITIONAL_ZOOM)
	
	# Set initial texture.
	update_texture_instant()


## Updates the texture with the current render parameters.
func update_texture_instant():
	var width   := RENDER_WIDTH / render_zoom
	var height  := RENDER_HEIGHT / render_zoom
	var bounds_vec: Array[float] = [render_xpos - width, render_xpos + width, render_ypos - height, render_ypos + height]
	var max_iterations := roundi(MAX_ITERATION_BASE * pow(render_zoom, 2))
	var data = mandelbrot_image_builder.create_image_data(
		TEXTURE_WIDTH, TEXTURE_HEIGHT, max_iterations,
		bounds_vec, _palette_reds, _palette_greens, _palette_blues
	)
	var image = Image.create_from_data(TEXTURE_WIDTH, TEXTURE_HEIGHT, false, Image.FORMAT_RGB8, data)
	texture = ImageTexture.create_from_image(image)


## Smoothly updates textures to the adjusted coords.
func update_texture_smooth(xpos: float, ypos: float, zoom: float):
	# TODO
	render_xpos = xpos
	render_ypos = ypos
	render_zoom = zoom
	update_texture_instant()


## Debug input.
func _input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			# Get the mouse coordinates and viewport size.
			var mouse_coordinates := get_viewport().get_mouse_position()
			var viewport_size := get_viewport().get_visible_rect().size
			
			# Round both down to be between -1.0 and 1.0.
			var xpos := ((mouse_coordinates.x / viewport_size.x) - 0.5) * 2
			var ypos := ((mouse_coordinates.y / viewport_size.y) - 0.5) * 2
			
			# Figure out the new xpos and ypos based on these.
			var width_offset   := (RENDER_WIDTH / render_zoom) * xpos
			var height_offset  := (RENDER_HEIGHT / render_zoom) * ypos
			
			# Add these to the current positions.
			render_xpos += width_offset
			render_ypos += height_offset
			
			# Re-render.
			update_texture_instant()
		elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
			# Zoom in and re-render.
			render_zoom *= ZOOM_STEP
			update_texture_instant()
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			# Zoom out and re-render.
			render_zoom /= ZOOM_STEP
			update_texture_instant()
