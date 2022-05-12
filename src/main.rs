extern crate core;

mod denoise_shader_gray;
mod denoise_shader_gray_compiled;
mod denoise_shader_frag;
mod vertex_shader;
mod denoise_compute;
mod denoise_frag;

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::device::{DeviceCreateInfo, Features, QueueCreateInfo};
use vulkano::device::DeviceExtensions;
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer};
use vulkano::device::{Device, Queue};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::format::Format;
use vulkano::image::{ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage};
use vulkano::instance::Instance;
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode, SamplerReductionMode};
use vulkano::sync::GpuFuture;
use vulkano::{Version};
use png::{BitDepth, ColorType};

pub(crate) fn vlk_init() -> (Arc<Device>, Arc<Queue>) {
        let instance =
            Instance::new(InstanceCreateInfo { application_name: None, application_version: Version::V1_3, enabled_extensions: InstanceExtensions::none(),
                enabled_layers: vec![], engine_name: None, engine_version: Default::default(), function_pointers: None, max_api_version: None, _ne: Default::default() })
                .expect("failed to create instance");

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
             .find(|&q| q.supports_compute())
             .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

        let devinfo = DeviceCreateInfo {
            enabled_extensions:  DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::none()
            },
            enabled_features: Features{
                shader_storage_image_extended_formats: true,
                sampler_anisotropy: false,
                ..Features::none()
            },
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            _ne: Default::default()
        };

        let (device, mut queues) = {
                Device::new(
                    physical_device,
                        devinfo,
                )
                    .expect("failed to create device")
        };

        //let limits = device.physical_device().limits();


        let queue = queues.next().unwrap();
        (device, queue)
}

enum UsingShader {
    Fragment,
    Compute
}

fn main() {

    let self_name = std::env::args().nth(0).unwrap();
    let usage_str = format!("Usage:\n{} filename_in filename_out fragment|compute", self_name);
    let filename_in = std::env::args().nth(1).expect(&format!("{}\n no filename_in found", usage_str));
    let filename_out = std::env::args().nth(2).expect(&format!("{}\n no filename_out found", usage_str));
    let shader_type_str = std::env::args().nth(3).expect(&format!("{}\n no shader type provided. Use fragment or compute", usage_str));

    let shader_type = match shader_type_str.as_str() {
        "fragment" => UsingShader::Fragment,
        "compute" => UsingShader::Compute,
        v => panic!("Unknown shader type {}. Use fragment or compute.", v)
    };

    let (device, queue) = vlk_init();
    let img_file = File::open(filename_in).unwrap();
    let mut decoder = png::Decoder::new(img_file);
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();

    assert_eq!(reader.info().bit_depth, BitDepth::Eight);
    assert_eq!(reader.info().color_type, ColorType::Grayscale);

    let (img_w, img_h) = reader.info().size();
    let mut buffer = vec![0; (img_w * img_h) as usize];
    reader.next_frame(&mut buffer).unwrap();

    let input_usage = BufferUsage{
        transfer_source: true,
        transfer_destination: false,
        uniform_texel_buffer: false,
        storage_texel_buffer: false,
        uniform_buffer: false,
        storage_buffer: true,
        index_buffer: false,
        vertex_buffer: false,
        indirect_buffer: false,
        device_address: true,
        _ne: Default::default()
    };

    let result_buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage{
        transfer_source: false,
        transfer_destination: true,
        uniform_texel_buffer: false,
        storage_texel_buffer: false,
        uniform_buffer: true,
        storage_buffer: true,
        index_buffer: false,
        vertex_buffer: false,
        indirect_buffer: false,
        device_address: true,
        _ne: Default::default()
    }, false, (0..img_w*img_h).map(|_|0u8)).expect("failed to create offsets buffer");

    let img_buf = CpuAccessibleBuffer::from_iter(device.clone(), input_usage, false, buffer.iter().map(|v| (*v as f32)))
        .expect("failed to create buffer");

    let input_img = StorageImage::with_usage(device.clone(),
                             ImageDimensions::Dim2d { width: img_w, height: img_h, array_layers: 1},
                             Format::R32_SFLOAT,
                             ImageUsage {
                                 transfer_source: false,
                                 transfer_destination: true,
                                 sampled: true,
                                 storage: true,
                                 color_attachment: false,
                                 depth_stencil_attachment: false,
                                 transient_attachment: false,
                                 input_attachment: false
                             },
                             ImageCreateFlags::none(),
                             Some(queue.family())).unwrap();
    let result_img = StorageImage::with_usage(device.clone(),
                                             ImageDimensions::Dim2d { width: img_w, height: img_h, array_layers: 1},
                                             Format::R8_UINT,
                                             ImageUsage {
                                                 transfer_source: true,
                                                 transfer_destination: false,
                                                 sampled: false,
                                                 storage: true,
                                                 color_attachment: true,
                                                 depth_stencil_attachment: false,
                                                 transient_attachment: false,
                                                 input_attachment: false
                                             },
                                             ImageCreateFlags::none(),
                                             Some(queue.family())).unwrap();

    //Buffer to image
    let command_buffer = {
        let mut builder =
            AutoCommandBufferBuilder::primary(device.clone(), queue.family(), CommandBufferUsage::OneTimeSubmit).unwrap();
        builder
            .copy_buffer_to_image(img_buf.clone(), input_img.clone()).unwrap();
        builder.build().unwrap()
    };
    println!("Execute copy buffer to image");
    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();


    let sampler = Sampler::new(device.clone(), SamplerCreateInfo {
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        mipmap_mode: SamplerMipmapMode::Linear,
        reduction_mode: SamplerReductionMode::WeightedAverage,
        address_mode: [SamplerAddressMode::Repeat; 3],
        mip_lod_bias: 0.0,
        lod: 0.0..=0.0,
        anisotropy: None,
        //anisotropy: Some(4.0),
        ..Default::default()
    }).unwrap();


    match shader_type {
        UsingShader::Fragment => denoise_frag::denoise(device.clone(), queue.clone(), input_img.clone(), result_img.clone(), sampler.clone()),
        UsingShader::Compute => denoise_compute::denoise(device.clone(), queue.clone(), input_img.clone(), result_img.clone(), sampler.clone())
    }

    //Read filtered image
    let command_buffer = {
        let mut builder =
            AutoCommandBufferBuilder::primary(device.clone(), queue.family(), CommandBufferUsage::OneTimeSubmit).unwrap();
        builder
            .copy_image_to_buffer(result_img.clone(), result_buf.clone()).unwrap();
        builder.build().unwrap()
    };
    println!("Execute copy image to buffer");
    let finished = command_buffer.execute(queue.clone()).unwrap();
    finished.then_signal_fence_and_flush().unwrap()
            .wait(None).unwrap();

    let rl = result_buf.read().unwrap();
    let result_bytes: Vec<u8> = rl.to_vec();

    let path_out = Path::new(&filename_out);
    let file = File::create(path_out).unwrap();
    let w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, img_w, img_h); // Width is 2 pixels and height is 1.
    encoder.set_color(reader.info().color_type);
    encoder.set_depth(reader.info().bit_depth);

    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&result_bytes).unwrap();
}
