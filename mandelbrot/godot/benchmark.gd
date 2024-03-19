extends Node

@export var PALETTE_SIZE    := 15
@export var MAX_ITERATIONS  := 32
@export var BENCHMARK_COUNT := 1
@export var ZOOM_AMOUNT     := 0.5

@onready var mandelbrot_image_builder: MandelbrotImageBuilder = $MandelbrotImageBuilder
@onready var mandelbrot_image: Sprite2D = $MandelbrotImage
@onready var terrain_image: Sprite2D = $TerrainImage
@onready var godot_terrain_builder: Node = $GodotTerrainBuilder


var palette_reds: Array[int] = []
var palette_greens: Array[int] = []
var palette_blues: Array[int] = []

var last_data: PackedByteArray
var last_terrain: PackedByteArray

func _ready():
	for i in range(PALETTE_SIZE):
		var color := Color.from_hsv(float(i) / float(PALETTE_SIZE), 0.7, 0.7)
		palette_reds.append(roundi(color.r * 255))
		palette_greens.append(roundi(color.g * 255))
		palette_blues.append(roundi(color.b * 255))
	
	var viewport_width = ProjectSettings.get_setting("display/window/size/viewport_width")
	var viewport_height = ProjectSettings.get_setting("display/window/size/viewport_height")
	var render_width = ceili(float(viewport_width) / ZOOM_AMOUNT)
	var render_height = ceili(float(viewport_height) / ZOOM_AMOUNT)
	var data
	var results = []
	var sum
	
	var xmiddle := -0.765
	var ymiddle := 0.0
	var width := 1.7
	var height := width * 0.75
	var bounds_vec: Array[float] = [xmiddle - width, xmiddle + width, ymiddle - height, ymiddle + height]
	
	## Benchmark test functions
	## TODO - reimplement these
	
	## Benchmark mandelbrot functions
	## TODO - reimplement Godot mandelbrot
	results = []
	for i in BENCHMARK_COUNT:
		data = measure(render_width, render_height, mandelbrot_image_builder, MAX_ITERATIONS, bounds_vec)
		results.append(data)
	sum = results.reduce(func(accum, number): return accum + number)
	print("Kernel mandelbrot: " + str(round((sum / BENCHMARK_COUNT) * 1000.0) / 1000.0) + " seconds")
	
	## Benchmark terrain functions
	results = []
	for i in BENCHMARK_COUNT:
		data = measure_terrain(render_width, render_height, godot_terrain_builder)
		results.append(data)
	sum = results.reduce(func(accum, number): return accum + number)
	print("Godot terrain: " + str(round((sum / BENCHMARK_COUNT) * 1000.0) / 1000.0) + " seconds")
	
	results = []
	for i in BENCHMARK_COUNT:
		data = measure_terrain(render_width, render_height, mandelbrot_image_builder)
		results.append(data)
	sum = results.reduce(func(accum, number): return accum + number)
	print("Kernel terrain: " + str(round((sum / BENCHMARK_COUNT) * 1000.0) / 1000.0) + " seconds")
	
	# Display the final mandelbrot/terrain.
	mandelbrot_image.scale = Vector2(ZOOM_AMOUNT, ZOOM_AMOUNT)
	mandelbrot_image.texture = ImageTexture.create_from_image(
		create_image(render_width, render_height, last_data)
	)
	
	var terrain_image_data := PackedByteArray()
	var terrain_points := 0
	for point in last_terrain:
		for i in 3:
			terrain_image_data.append(point)
		if point != 0:
			terrain_points += 1
	print("Terrain point count: " + str(terrain_points))
	
	terrain_image.scale = Vector2(ZOOM_AMOUNT, ZOOM_AMOUNT)
	terrain_image.texture = ImageTexture.create_from_image(
		create_image(render_width, render_height, terrain_image_data)
	)


func measure(width, height, builder, iterations, bounds_vec) -> float:
	var start = float(Time.get_ticks_usec())
	last_data = builder.create_image_data(
		width, height, iterations, bounds_vec,
		palette_reds, palette_greens, palette_blues
	)
	var end = float(Time.get_ticks_usec())
	return (end - start) / 1000000.0


func measure_terrain(width, height, builder) -> float:
	if not last_data:
		assert(false, "Terrain measurement called without data being measured beforehand")
	var start = float(Time.get_ticks_usec())
	last_terrain = builder.generate_terrain_data(width, height, last_data)
	var end = float(Time.get_ticks_usec())
	return (end - start) / 1000000.0


func create_image(width: int, height: int, data: PackedByteArray) -> Image:
	return Image.create_from_data(width, height, false, Image.FORMAT_RGB8, data)
