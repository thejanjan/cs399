extends Sprite2D

const AIR_SPEED := 4.0
const MOVE_SPEED := 160.0
const GRAVITY := 5.0
const OUT_OF_BOUNDS_BARRIER := 0.9
const POINTS_TO_SAMPLE_NORMAL := 0.1
const SAMPLE_DISTANCE := pow(120.0, 2)
const ROTATION_WEIGHT := 8.0
const JUMP_POWER := 5.0
const JUMP_DEBOUNCE := 10

var VIEWPORT_WIDTH:  int = ProjectSettings.get_setting("display/window/size/viewport_width")
var VIEWPORT_HEIGHT: int = ProjectSettings.get_setting("display/window/size/viewport_height")

@onready var terrain_manager: Node = $"../TerrainManager"
@onready var intro_text: RichTextLabel = $"../Interface/IntroText"
@onready var zoom_manager: Node = $"../ZoomManager"
@onready var jump_sfx: AudioStreamPlayer = $JumpSfx

var velocity := Vector2()
var move_velocity := Vector2()
var airborne := true
var current_normal := 0.0
var current_gravity := 0.0
var jump_timer := 0

func _ready():
	intro_text.game_ready.connect(spawn)

func spawn():
	await get_tree().create_timer(0.5).timeout
	if visible:
		return
	visible = true
	modulate = Color(1.0, 1.0, 1.0, 0.0)
	create_tween().tween_property(self, "modulate", Color(1.0, 1.0, 1.0, 1.0), 0.75)

func _process(delta: float) -> void:
	if not visible:
		return
	if not airborne:
		# Check if we're out of bounds like this:
		var out_of_bounds: bool = abs(position.x) > ((VIEWPORT_WIDTH  / 2) * OUT_OF_BOUNDS_BARRIER) or \
							 abs(position.y) > ((VIEWPORT_HEIGHT / 2) * OUT_OF_BOUNDS_BARRIER)
		
		# When on the ground, handle zoom in/zoom out requests.
		var just_zoomed := false
		if Input.is_action_pressed("zoom_in"):
			just_zoomed = zoom_manager.zoom_in(position)
		elif Input.is_action_pressed("zoom_out"):
			just_zoomed = zoom_manager.zoom_out(position)
		elif out_of_bounds:
			just_zoomed = zoom_manager.recenter(position)
		if just_zoomed:
			await RenderingServer.frame_post_draw
			position = Vector2(0, 0)
			position = get_nearest_points()[0]
		else:
			# See if we can jump.
			if Input.is_action_just_pressed("jump"):
				airborne = true
				velocity = Vector2.from_angle(current_normal) * JUMP_POWER
				jump_timer = JUMP_DEBOUNCE
				jump_sfx.play()

func _physics_process(delta: float) -> void:
	if not visible:
		return
	
	# Obtain the nearest points.
	var nearest_points := get_nearest_points()
	if not nearest_points:
		return
	
	# Calculate current normal.
	current_normal = get_normal(nearest_points)
	rotation = lerp_angle(rotation, current_normal, ROTATION_WEIGHT * delta)
	
	# Get movement.
	var movement := Input.get_vector("move_left", "move_right", "move_up", "move_down")
	
	# Are we in air?
	if airborne:
		# Allow free movement in air.
		velocity += movement * AIR_SPEED * delta
		
		# Perform gravity.
		var point: Vector2 = nearest_points[0]
		var gravity_vec: Vector2 = position.direction_to(point) * GRAVITY * delta
		velocity += gravity_vec
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
			move_velocity = position.direction_to(nearest_point_in_direction).normalized() * MOVE_SPEED * delta
		else:
			move_velocity = Vector2()
	
	if airborne:
		if jump_timer > 0:
			jump_timer -= 1
		else:
			for point: Vector2 in nearest_points.slice(0, 5):
				var point_dist := position.distance_squared_to(point)
				if point_dist < velocity.length() or point_dist < 16.0:
					airborne = false
					position = point
					current_gravity = 0.0
					velocity = Vector2()
					break
	
	position += velocity + move_velocity

func get_nearest_points() -> Array[Vector2]:
	var nearest_points: Array[Vector2] = terrain_manager.vector_points.duplicate()
	var distance_cache := {}
	nearest_points.sort_custom(
		func (a: Vector2, b: Vector2): \
			return get_or_set(distance_cache, a, a.distance_squared_to(position)) < \
			get_or_set(distance_cache, b, b.distance_squared_to(position))
	)
	return nearest_points

func get_normal(nearest_points: Array[Vector2]) -> float:
	var angles_sampled: Array[float] = []
	for point in nearest_points:
		var angle: float = terrain_manager.angle_points[point]
		if position.distance_squared_to(point) < SAMPLE_DISTANCE or airborne:
			angles_sampled.append(angle)
		if len(angles_sampled) > ceili(len(nearest_points) * POINTS_TO_SAMPLE_NORMAL):
			break
	return terrain_manager.mean_angle(angles_sampled)

static func get_or_set(d: Dictionary, k: Variant, v: Variant) -> Variant:
	if k in d:
		return d[k]
	d[k] = v
	return v
