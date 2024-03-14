extends Sprite2D

@export var image_zoom_in: float = 2.0

@onready var test_image_builder: Node = $"../TestImageBuilder"
@onready var mandelbrot_image_builder: MandelbrotImageBuilder = $"../MandelbrotImageBuilder"


func _ready():
	pass


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


func create_image(width: int, height: int, data: PackedByteArray) -> Image:
	return Image.create_from_data(width, height, false, Image.FORMAT_RGB8, data)
