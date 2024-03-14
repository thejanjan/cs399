extends Sprite2D

@onready var test_image_builder: Node = $"../TestImageBuilder"
@onready var mandelbrot_image_builder: MandelbrotImageBuilder = $"../MandelbrotImageBuilder"


func _ready():
	var start = float(Time.get_ticks_usec())
	var image = test_image_builder.create_image(300, 200)
	var end = float(Time.get_ticks_usec())
	print("Image generation time: " + str((end - start) / 1000000.0) + " seconds")
	texture = ImageTexture.create_from_image(image)
