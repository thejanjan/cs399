extern crate ocl;

use ocl::prm::Char;
use ocl::Buffer;
use ocl::ProQue;
use godot::prelude::*;
use godot::engine::Node;
use godot::engine::INode;
use godot::builtin::Array;
use ocl::SpatialDims;

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
    fn create_image_data(&mut self, width: i32, height: i32, iterations: u32, bounds: Array<f32>, g_reds: Array<u8>, g_greens: Array<u8>, g_blues: Array<u8>) -> PackedByteArray {
        // Create color vectors.
        assert!(g_reds.len() == g_greens.len() && g_greens.len() == g_blues.len(), "Provided color vectors did not have equal lengths");
        let mut reds = vec![0u8; g_reds.len()];
        let mut greens = vec![0u8; g_reds.len()];
        let mut blues = vec![0u8; g_reds.len()];
        let mut bounds_vec = vec![0f32; bounds.len()];

        for i in 0..g_reds.len() {
            reds[i]   = g_reds.get(i);
            greens[i] = g_greens.get(i);
            blues[i]  = g_blues.get(i);
        }
        for i in 0..bounds.len() {
            bounds_vec[i]   = bounds.get(i);
        }
        
        // Now perform kernels.
        let mut data = PackedByteArray::new();

        // Defer mandelbrot work to kernels
        match compute_iterations(width as usize, height as usize, iterations, bounds_vec, reds, greens, blues) {
            Ok(v) => for i in v {data.push(i);},
            Err(e) => panic!("{e}")
        }

        data
    }

    #[func]
    fn generate_terrain_data(&mut self, width: i32, height: i32, data: PackedByteArray) -> PackedByteArray {
        // Setup terrain data.
        let mut data_vec = vec![0u8; data.len()];
        for i in 0..data.len() {
            data_vec[i] = data.get(i);
        }

        let mut terrain = PackedByteArray::new();

        // Defer work to kernels
        match compute_terrain(width as usize, height as usize, data_vec) {
            Ok(v) => for i in v {terrain.push(i);},
            Err(e) => panic!("{e}")
        }

        terrain
    }

    #[func]
    fn create_image_data_test(&mut self, width: i32, height: i32) -> PackedByteArray {
        let mut data = PackedByteArray::new();
        match compute_test(width, height) {
            Ok(v) => for i in v {data.push(i);},
            Err(e) => panic!("{e}")
        }

        data
    }
}

fn compute_iterations(width: usize, height: usize, iterations: u32, bounds_vec: Vec<f32>,
                      reds: Vec<u8>, greens: Vec<u8>, blues: Vec<u8>) -> ocl::Result<Vec<u8>> {
    let src = r#"
        __kernel void mandelbrot(__global uchar *data, __global float *bounds, uint MAX_ITERATIONS,
            __global uchar *reds, __global uchar *greens, __global uchar *blues, uint colors) {
            // Get image globals.
            int Px = get_global_id(0);
            int Py = get_global_id(1);
            int width = get_global_size(0);
            int height = get_global_size(1);

            // Determine x0 and y0 based off of image bounds (inlined lerp)
            float x0_weight = (float)Px / (float)(width - 1);
            float y0_weight = (float)Py / (float)(height - 1);
            float x0 = (bounds[0] * (1.0f - x0_weight)) + (bounds[1] * x0_weight);
            float y0 = (bounds[2] * (1.0f - y0_weight)) + (bounds[3] * y0_weight);

            // Do algorithm.
            float x = 0.0f;
            float y = 0.0f;
            int iteration = 0;
            float xtemp;
            int offset = (3 * Py * width) + (Px * 3);

            while (((x*x) + (y*y)) <= 4.0f) {
                xtemp = (x*x) - (y*y) + x0;
                y = 2*x*y + y0;
                x = xtemp;
                iteration++;

                if (iteration == MAX_ITERATIONS) {
                    data[offset]     = 0;
                    data[offset + 1] = 0;
                    data[offset + 2] = 0;
                    break;
                }
            }

            // Insert RGB values into data buffer.
            if (iteration != MAX_ITERATIONS) {
                data[offset]     = reds[iteration % colors];
                data[offset + 1] = greens[iteration % colors];
                data[offset + 2] = blues[iteration % colors];
            }
        }
    "#;

    // Setup dimension and proque.
    let dims = SpatialDims::new(Some(width), Some(height), Some(1usize))?;
    let pro_que = ProQue::builder().src(src).dims(dims).build()?;

    // Create data buffers.
    let data: Buffer<u8> = Buffer::builder()
        .len(width * height * 3)
        .queue(pro_que.queue().clone())
        .build()?;

    // let bounds = pro_que.create_buffer::<f32>()?;
    let bounds = Buffer::builder()
        .len(4)
        .queue(pro_que.queue().clone())
        .copy_host_slice(&bounds_vec)
        .build()?;

    let red_buffer = Buffer::builder()
        .len(reds.len())
        .queue(pro_que.queue().clone())
        .copy_host_slice(&reds)
        .build()?;

    let green_buffer = Buffer::builder()
        .len(greens.len())
        .queue(pro_que.queue().clone())
        .copy_host_slice(&greens)
        .build()?;

    let blue_buffer = Buffer::builder()
        .len(blues.len())
        .queue(pro_que.queue().clone())
        .copy_host_slice(&blues)
        .build()?;

    let colors = (reds.len() as u32) - 1;

    // Perform kernel.
    let kernel = pro_que
        .kernel_builder("mandelbrot")
        .arg(&data)
        .arg(&bounds)
        .arg(&iterations)
        .arg(&red_buffer)
        .arg(&green_buffer)
        .arg(&blue_buffer)
        .arg(&colors)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    // Retrieve data.
    let mut vec = vec![0u8; data.len()];
    data.read(&mut vec).enq()?;

    // We're done.
    Ok(vec)
}

fn compute_terrain(width: usize, height: usize, data_vec: Vec<u8>) -> ocl::Result<Vec<u8>> {
    let src = r#"
        __kernel void make_terrain(__global uchar *terrain, __global uchar *data) {
            // Get image globals.
            int Px = get_global_id(0) * 2;  // 2 from working at a lower resolution
            int Py = get_global_id(1) * 2;
            int width = get_global_size(0) * 2;
            int height = get_global_size(1) * 2;

            // Only work in bounds.
            if (Px > 0 && Py > 0 && Px < (width - 1) && Py < (height - 1)) {
                // Calculate indices of adjacent points.
                int LEFT  = ((Px - 1) + (Py * width)) * 3;  // 3 from iterating over RGB data
                int RIGHT = ((Px + 1) + (Py * width)) * 3;
                int UP    = (Px       + ((Py - 1) * width)) * 3;
                int DOWN  = (Px       + ((Py + 1) * width)) * 3;

                // If we have two 0 neighbors, then set the terrain at this point to max uchar.
                // Note that we deliberately ensure that the minimum red is 1 so that
                // we guarantee find 0s as max iteration terrain.
                int NEIGHBORS = (data[LEFT] == 0) + (data[RIGHT] == 0) + (data[UP] == 0) + (data[DOWN] == 0);
                if (NEIGHBORS == 3) {
                    terrain[get_global_id(0) + (get_global_id(1) * get_global_size(0))] = 255;
                }
            }
        }
    "#;

    // Setup dimension and proque.
    let dims = SpatialDims::new(Some(width / 2), Some(height / 2), Some(1usize))?;
    let pro_que = ProQue::builder().src(src).dims(dims).build()?;

    // Create data buffers.
    let terrain = pro_que.create_buffer::<u8>()?;

    let data = Buffer::builder()
        .len(data_vec.len())
        .queue(pro_que.queue().clone())
        .copy_host_slice(&data_vec)
        .build()?;

    // Perform kernel.
    let kernel = pro_que
    .kernel_builder("make_terrain")
        .arg(&terrain)
        .arg(&data)
    .build()?;

    unsafe {
        kernel.enq()?;
    }

    // Retrieve data.
    let mut vec = vec![0u8; terrain.len()];
    terrain.read(&mut vec).enq()?;

    // We're done.
    Ok(vec)
}

fn compute_test(width: i32, height: i32) -> ocl::Result<Vec<u8>> {
    let src = r#"
        __kernel void test(__global char* buffer, char scalar) {
            buffer[get_global_id(0)] = scalar;
        }
    "#;

    let pro_que = ProQue::builder().src(src).dims(width * height * 3).build()?;

    let buffer = pro_que.create_buffer::<u8>()?;

    let kernel = pro_que
        .kernel_builder("test")
        .arg(&buffer)
        .arg(&Char::new(0xFFu8 as i8))
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut vec = vec![0u8; buffer.len()];
    buffer.read(&mut vec).enq()?;

    Ok(vec)
}
