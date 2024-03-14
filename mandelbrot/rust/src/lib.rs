extern crate ocl;
use ocl::ProQue;
use godot::prelude::*;
use godot::engine::Node;
use godot::engine::INode;
use godot::engine::Image;
use godot::engine::image::Format;

struct MyExtension;

#[gdextension]
unsafe impl ExtensionLibrary for MyExtension {}

#[derive(GodotClass)]
#[class(base=Node, init)]
struct MandelbrotImageBuilder {
    base: Base<Node>
}

#[godot_api]
impl INode for MandelbrotImageBuilder {
}

#[godot_api]
impl MandelbrotImageBuilder {
    #[func]
    fn create_image(&mut self, width: i32, height: i32) -> Gd<Image> {
        let mut data = PackedByteArray::new();
        match compute(width, height) {
            Ok(v) => for element in &v {data.push(*element)},
            Err(e) => panic!("{e}")
        }

        let image = Image::create_from_data(width, height, false, Format::RGB8, data);
        image.expect("No image was created.")
    }
}

fn compute(width: i32, height: i32) -> ocl::Result<Vec<u8>> {
    let src = r#"
        __kernel void mandelbrot(__global char* buffer, char scalar) {
            buffer[get_global_id(0)] += scalar;
        }
    "#;

    let pro_que = ProQue::builder().src(src).dims(width * height * 3).build()?;

    let buffer = pro_que.create_buffer::<u8>()?;

    let kernel = pro_que
        .kernel_builder("mandelbrot")
        .arg(&buffer)
        .arg(255u8)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut vec = vec![0u8; buffer.len()];
    buffer.read(&mut vec).enq()?;

    Ok(vec)
}
