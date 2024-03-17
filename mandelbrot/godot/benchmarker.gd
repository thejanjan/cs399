extends Sprite2D

const PALETTE_SIZE = 15
const MAX_ITERATIONS = 32

@export var image_zoom_in: float = 4.0
@onready var mandelbrot_image_builder: MandelbrotImageBuilder = $"../MandelbrotImageBuilder"

var palette_reds: Array[int] = []
var palette_greens: Array[int] = []
var palette_blues: Array[int] = []


func _ready():
	for i in range(PALETTE_SIZE):
		var color := Color.from_hsv(float(i) / float(PALETTE_SIZE), 0.7, 0.7)
		palette_reds.append(roundi(color.r * 255))
		palette_greens.append(roundi(color.g * 255))
		palette_blues.append(roundi(color.b * 255))
	
	var viewport_width = ProjectSettings.get_setting("display/window/size/viewport_width")
	var viewport_height = ProjectSettings.get_setting("display/window/size/viewport_height")
	var render_width = ceili(float(viewport_width) / image_zoom_in)
	var render_height = ceili(float(viewport_height) / image_zoom_in)
	var data
	
	var results = []
	var sum
	const count = 10
	
	var xmiddle := -0.765
	var ymiddle := 0.0
	var width := 1.7
	var height := width * 0.75
	
	var bounds_vec: Array[float] = [xmiddle - width, xmiddle + width, ymiddle - height, ymiddle + height]
	
	for i in count:
		data = measure_mandelbrot(render_width, render_height, mandelbrot_image_builder, MAX_ITERATIONS, bounds_vec)
		results.append(data)
	sum = results.reduce(func(accum, number): return accum + number)
	print("Kernel mandelbrot: " + str(round((sum / count) * 1000.0) / 1000.0) + " seconds")
	
	data = mandelbrot_image_builder.create_image_data(render_width, render_height, MAX_ITERATIONS, bounds_vec, palette_reds, palette_greens, palette_blues)
	var image = create_image(render_width, render_height, data)
	texture = ImageTexture.create_from_image(image)
	scale = Vector2(image_zoom_in, image_zoom_in)


func benchmark():
	var viewport_width = ProjectSettings.get_setting("display/window/size/viewport_width")
	var viewport_height = ProjectSettings.get_setting("display/window/size/viewport_height")
	var render_width = ceili(float(viewport_width) / image_zoom_in)
	var render_height = ceili(float(viewport_height) / image_zoom_in)
	
	print("Image size: " + str(render_width) + "x" + str(render_height))
	
	var results
	var sum
	var count = 100
	
	print("Test iterations: " + str(count))
	
	results = []
	for i in count:
		results.append(measure(render_width, render_height, test_image_builder))
	sum = results.reduce(func(accum, number): return accum + number)
	print("Test image data: " + str(round((sum / count) * 1000.0) / 1000.0) + " seconds")
	
	results = []
	for i in count:
		results.append(measure(render_width, render_height, mandelbrot_image_builder))
	sum = results.reduce(func(accum, number): return accum + number)
	print("Mandelbrot image data: " + str(round((sum / count) * 1000.0) / 1000.0) + " seconds")
	
	#start = float(Time.get_ticks_usec())
	#image = test_image_builder.create_image(viewport_width, viewport_height)
	#end = float(Time.get_ticks_usec())
	#print("Test image: " + str((end - start) / 1000000.0) + " seconds")
#
	#start = float(Time.get_ticks_usec())
	#image = mandelbrot_image_builder.create_image(viewport_width, viewport_height)
	#end = float(Time.get_ticks_usec())
	#print("Mandelbrot image: " + str((end - start) / 1000000.0) + " seconds")
#
	#texture = ImageTexture.create_from_image(image)
	#scale = Vector2(1.0 / image_zoom_in, 1.0 / image_zoom_in)


func measure(width, height, builder) -> float:
	var start = float(Time.get_ticks_usec())
	builder.create_image_data(width, height)
	var end = float(Time.get_ticks_usec())
	return (end - start) / 1000000.0


func measure_mandelbrot(width, height, builder, iterations, bounds_vec) -> float:
	var start = float(Time.get_ticks_usec())
	builder.create_image_data(width, height, iterations, bounds_vec, palette_reds, palette_greens, palette_blues)
	var end = float(Time.get_ticks_usec())
	return (end - start) / 1000000.0


func create_image(width: int, height: int, data: PackedByteArray) -> Image:
	return Image.create_from_data(width, height, false, Image.FORMAT_RGB8, data)
