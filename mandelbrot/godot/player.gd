extends Sprite2D

const MOVE_SPEED := 120.0
const GRAVITY := 100.0


@onready var terrain_manager: Node = $"../TerrainManager"
@onready var intro_text: RichTextLabel = $"../Interface/IntroText"


var velocity := 0.0
var airborne := true
var current_gravity := 0.0


func _ready():
	intro_text.game_ready.connect(spawn)


func spawn():
	await get_tree().create_timer(0.5).timeout
	if visible:
		return
	visible = true
	modulate = Color(1.0, 1.0, 1.0, 0.0)
	create_tween().tween_property(self, "modulate", Color(1.0, 1.0, 1.0, 1.0), 0.75)


func _physics_process(delta: float) -> void:
	if not visible:
		return
	
	# Obtain the nearest points.
	var distance_cache := {}
	var nearest_points: Array[Vector2] = terrain_manager.vector_points.duplicate()
	nearest_points.sort_custom(
		func (a: Vector2, b: Vector2): \
			return get_or_set(distance_cache, a, a.distance_squared_to(position)) < \
			get_or_set(distance_cache, b, b.distance_squared_to(position))
	)
	
	# Get movement.
	var movement := Input.get_vector("move_left", "move_right", "move_up", "move_down")
	
	# Are we in air?
	if airborne:
		# Allow free movement in air.
		position += movement * MOVE_SPEED * delta
		
		# Perform gravity.
		current_gravity += delta * GRAVITY
		position = position.move_toward(nearest_points[0], current_gravity * delta)
		if position.distance_squared_to(nearest_points[0]) < 0.1:
			airborne = false
			current_gravity = 0.0
	else:
		# Movement is restricted to nearby points.
		var nearest_point_in_direction: Vector2
		for point in nearest_points:
			var point_dist := point.distance_squared_to(position)
			if (point - position).normalized().dot(movement) > 0.5 and \
				point_dist < 1600.0 and point_dist > 0.1:
				nearest_point_in_direction = point
				break
		if nearest_point_in_direction:
			position = position.move_toward(nearest_point_in_direction, MOVE_SPEED * delta)

static func get_or_set(d: Dictionary, k: Variant, v: Variant) -> Variant:
	if k in d:
		return d[k]
	d[k] = v
	return v
