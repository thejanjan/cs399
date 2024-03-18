extends Node

@onready var mandelbrot_texture: Sprite2D = $"../MandelbrotTexture"
@onready var mandelbrot_image_builder: MandelbrotImageBuilder = $"../MandelbrotImageBuilder"
@onready var terrain_image: Sprite2D = $TerrainImage

@onready var VIEWPORT_WIDTH:  int = ProjectSettings.get_setting("display/window/size/viewport_width")
@onready var VIEWPORT_HEIGHT: int = ProjectSettings.get_setting("display/window/size/viewport_height")
@onready var SCREEN_LEFT  := roundf(-(VIEWPORT_WIDTH / 2)  * mandelbrot_texture.ADDITIONAL_ZOOM)
@onready var SCREEN_RIGHT := roundf( (VIEWPORT_WIDTH / 2)  * mandelbrot_texture.ADDITIONAL_ZOOM)
@onready var SCREEN_UP    := roundf(-(VIEWPORT_HEIGHT / 2) * mandelbrot_texture.ADDITIONAL_ZOOM)
@onready var SCREEN_DOWN  := roundf( (VIEWPORT_HEIGHT / 2) * mandelbrot_texture.ADDITIONAL_ZOOM)

var terrain_data: PackedByteArray
var vector_points: Array[Vector2] = []
var angle_points := {}  # Vector2: float

func _ready() -> void:
	mandelbrot_texture.new_data.connect(_on_new_data)
	terrain_image.scale = Vector2(
		mandelbrot_texture.IMAGE_SCALE,
		mandelbrot_texture.IMAGE_SCALE
	)

## When we receive new data, build the terrain.
func _on_new_data(data: PackedByteArray):
	terrain_data = mandelbrot_image_builder.generate_terrain_data(
		mandelbrot_texture.TEXTURE_WIDTH,
		mandelbrot_texture.TEXTURE_HEIGHT,
		data
	)
	
	# If our terrain image is visible, update it.
	if terrain_image.visible:
		_update_terrain_image()
	
	# Iterate over the data, create the relevant points.
	var width:  int = mandelbrot_texture.TEXTURE_WIDTH
	var height: int = mandelbrot_texture.TEXTURE_HEIGHT
	vector_points = []
	for index in terrain_data.size():
		# Ignore terrainless points.
		var normal: int = terrain_data[index]
		if normal == 0:
			continue
		
		# This index has a point. Map it from 0 (left) to 1 (right)
		var x := float(index % width) / width
		var y := float(index / width) / height
		
		# Create a vector at this point, mapped onto the scren.
		var new_x := roundi(lerp(SCREEN_LEFT, SCREEN_RIGHT, x)) + 4
		var new_y := roundi(lerp(SCREEN_UP, SCREEN_DOWN, y))    + 4
		var new_vector := Vector2(new_x, new_y)
		vector_points.append(new_vector)
		
		# Calculate the normal of this vector.
		angle_points[new_vector] = mean_angle(normal_into_angles(normal))

## Updates the terrain image node.
func _update_terrain_image():
	var terrain_image_data := PackedByteArray()
	for point in terrain_data:
		for i in 3:
			terrain_image_data.append(point)
	
	terrain_image.texture = ImageTexture.create_from_image(
		Image.create_from_data(
			mandelbrot_texture.TEXTURE_WIDTH,
			mandelbrot_texture.TEXTURE_HEIGHT,
			false, Image.FORMAT_RGB8, terrain_image_data
		)
	)

## Converts our normal byte into a list of angles.
static func normal_into_angles(normal: int) -> Array[float]:
	var angles: Array[float] = []
	if normal & 1:
		angles.append(deg_to_rad(0 + 180))
	if normal & 2:
		angles.append(deg_to_rad(45 + 180))
	if normal & 4:
		angles.append(deg_to_rad(90 + 180))
	if normal & 8:
		angles.append(deg_to_rad(135 + 180))
	if normal & 16:
		angles.append(deg_to_rad(180 + 180))
	if normal & 32:
		angles.append(deg_to_rad(225 + 180))
	if normal & 64:
		angles.append(deg_to_rad(270 + 180))
	if normal & 128:
		angles.append(deg_to_rad(315 + 180))
	return angles

## Calculates the mean angle of a list.
static func mean_angle(angles: Array[float]) -> float:
	# Implementation borrowed from https://rosettacode.org/wiki/Averages/Mean_angle
	var x_part := 0.0
	var y_part := 0.0
	var size := angles.size()
	for angle in angles:
		x_part += cos(angle)
		y_part += sin(angle)
	return atan2(y_part / size, x_part / size)
