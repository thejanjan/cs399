extends Sprite2D
## The main texture control for rendering the Mandelbrot Set.

@onready var zoom_manager: Node = $"../ZoomManager"

@onready var music: AudioStreamPlayer = $"../Music"

@onready var transition_texture: Sprite2D = $TransitionTexture
const TRANSITION_DURATION := 0.5

## Additional zoom-in multiplier on the image scale.
const ADDITIONAL_ZOOM := 1.1

## The number of colors in our palette.
const PALETTE_SIZE := 64

## How pixely our render is.
const IMAGE_PIXELIZE := 4.0
const IMAGE_SCALE := IMAGE_PIXELIZE * ADDITIONAL_ZOOM

## How much we zoom in per step.
const ZOOM_STEP := 1.25

## The base number of iterations to render at the start scale.
const MAX_ITERATION_BASE = 16

## The width and height that the Mandelbrot calculations occur in.
const RENDER_WIDTH := 1.7
const RENDER_HEIGHT := RENDER_WIDTH * 0.75

## The width and height of the viewport.
var VIEWPORT_WIDTH:  int = ProjectSettings.get_setting("display/window/size/viewport_width")
var VIEWPORT_HEIGHT: int = ProjectSettings.get_setting("display/window/size/viewport_height")

## The texture width and height that gets produced.
var TEXTURE_WIDTH  := ceili(float(VIEWPORT_WIDTH) / IMAGE_PIXELIZE)
var TEXTURE_HEIGHT := ceili(float(VIEWPORT_HEIGHT) / IMAGE_PIXELIZE)

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

## Determines if we are currently transitioning.
var mid_transition := false

## The current mandelbrot data.
var data: PackedByteArray
signal new_data(data: PackedByteArray)

func _ready():
	# Build our color palette.
	for i in range(PALETTE_SIZE):
		var color := Color.from_hsv(float(i) / float(PALETTE_SIZE), 0.7, 0.7)
		# We ensure the minimum RGB is 1 so that we can check for 0s
		# when building terrain in our kernel ( a bit of a hack .. or is it an optimization? )
		_palette_reds  .append(max(roundi(color.r * 255), 1))
		_palette_greens.append(max(roundi(color.g * 255), 1))
		_palette_blues .append(max(roundi(color.b * 255), 1))
	
	# Set texture scale.
	scale = Vector2(IMAGE_SCALE, IMAGE_SCALE)
	
	# Set initial texture.
	update_texture_instant.call_deferred()
	
	# Start music after texture is rendered.
	music.play.call_deferred()


## Updates the current mandelbrot data.
func update_data() -> PackedByteArray:
	var width   := RENDER_WIDTH / render_zoom
	var height  := RENDER_HEIGHT / render_zoom
	var bounds_vec: Array[float] = [render_xpos - width, render_xpos + width, render_ypos - height, render_ypos + height]
	var max_iterations := roundi(MAX_ITERATION_BASE * pow(render_zoom, 1))
	data = await mandelbrot_image_builder.create_image_data(
		TEXTURE_WIDTH, TEXTURE_HEIGHT, max_iterations,
		bounds_vec, _palette_reds, _palette_greens, _palette_blues
	)
	new_data.emit(data)
	return data


## Obtains the image with the current mandelbrot data.
func generate_image() -> Image:
	return Image.create_from_data(TEXTURE_WIDTH, TEXTURE_HEIGHT, false, Image.FORMAT_RGB8, data)


## Generates a texture with the current mandelbrot data.
func generate_texture() -> ImageTexture:
	return ImageTexture.create_from_image(generate_image())


## Updates the texture with the current render parameters.
func update_texture_instant():
	update_data()
	texture = generate_texture()

## Smoothly updates textures to the adjusted coords in screen space.
func update_texture_smooth_screen_space(screen_pos: Vector2, zoom: float):
	# We need to convert the screen position into render space.
	# If screen_pos=(0, 0), then our render x/y stays the same, as it remains centered.
	# Depending on our distance to the edges of the viewport, the render x/y will lerp
	# to meet the bounds of the currently visible render.
	var width   := RENDER_WIDTH  / render_zoom
	var height  := RENDER_HEIGHT / render_zoom
	var screen_xdelta := inverse_lerp(-VIEWPORT_WIDTH / 2,  VIEWPORT_WIDTH / 2,  screen_pos.x)
	var screen_ydelta := inverse_lerp(-VIEWPORT_HEIGHT / 2, VIEWPORT_HEIGHT / 2, screen_pos.y)
	update_texture_smooth(
		lerp(render_xpos - width, render_xpos + width, screen_xdelta),
		lerp(render_ypos - height, render_ypos + height, screen_ydelta),
		zoom
	)

## Smoothly updates textures to the adjusted coords.
func update_texture_smooth(xpos: float, ypos: float, zoom: float):
	mid_transition = true
	await RenderingServer.frame_post_draw
	
	# Set the new data.
	render_xpos = xpos
	render_ypos = ypos
	render_zoom = zoom
	update_data()
	
	# Freeze the current texture 
	texture = ImageTexture.create_from_image(get_viewport().get_texture().get_image())
	
	# Prepare for transition. A lifetime of denial all for this serene moment...
	var new_texture := generate_texture()
	var frozen_viewport_texture := ImageTexture.create_from_image(
		get_viewport().get_texture().get_image())
	transition_texture.texture = frozen_viewport_texture
	transition_texture.visible = true
	texture = new_texture
	
	# Godot Yuri
	var t := create_tween().set_parallel()
	t.tween_method(func (x): material.set_shader_parameter("alpha", x),
				0.0, 1.0, TRANSITION_DURATION)
	t.tween_method(func (x): transition_texture.material.set_shader_parameter("alpha", x),
				1.0, 0.0, TRANSITION_DURATION)
	await t.finished
	
	# Set back to start
	transition_texture.visible = false
	mid_transition = false


## Debug input.
#func _input(event: InputEvent) -> void:
	#if event is InputEventMouseButton:
		#if event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
			## Get the mouse coordinates and viewport size.
			#var mouse_coordinates := get_viewport().get_mouse_position()
			#var viewport_size := get_viewport().get_visible_rect().size
			#
			## Round both down to be between -1.0 and 1.0.
			#var xpos := ((mouse_coordinates.x / viewport_size.x) - 0.5) * 2
			#var ypos := ((mouse_coordinates.y / viewport_size.y) - 0.5) * 2
			#
			## Figure out the new xpos and ypos based on these.
			#var width_offset   := (RENDER_WIDTH / render_zoom) * xpos
			#var height_offset  := (RENDER_HEIGHT / render_zoom) * ypos
			#
			## Re-render.
			#update_texture_smooth(
				#render_xpos + width_offset,
				#render_ypos + height_offset,
				#render_zoom
			#)
		#elif event.button_index == MOUSE_BUTTON_WHEEL_UP and event.pressed:
			## Zoom in and re-render.
			#zoom_manager.zoom_in()
		#elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN and event.pressed:
			## Zoom out and re-render.
			#zoom_manager.zoom_out()
