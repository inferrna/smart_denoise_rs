mod denoise_shader;
mod denoise_shader_compiled;

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use image::io::Reader as ImageReader;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::device::{DeviceCreateInfo, Features, QueueCreateInfo};
use vulkano::device::DeviceExtensions;
use vulkano::instance::{InstanceCreateInfo, InstanceExtensions};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::format::Format;
use vulkano::image::{ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage};
use vulkano::image::view::{ImageView, ImageViewCreateInfo, ImageViewType};
use vulkano::instance::Instance;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::query::{QueryControlFlags, QueryPool, QueryPoolCreateInfo, QueryType};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
use vulkano::sync::GpuFuture;
use vulkano::{sync, Version};


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

fn main() {

    let self_name = std::env::args().nth(0).unwrap();
    let usage_str = format!("Usage:\n{} filename_in filename_out", self_name);
    let filename_in = std::env::args().nth(1).expect(&format!("{}\n no filename_in found", usage_str));
    let filename_out = std::env::args().nth(2).expect(&format!("{}\n no filename_out found", usage_str));


    let (device, queue) = vlk_init();
    let img_file = File::open(filename_in).unwrap();
    let mut decoder = png::Decoder::new(img_file);
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().unwrap();
    let (img_w, img_h) = reader.info().size();
    let mut buffer = vec![0; (img_w * img_h) as usize];
    reader.next_frame(&mut buffer).unwrap();

    let path_out = Path::new(&filename_out);
    let file = File::create(path_out).unwrap();
    let mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, img_w, img_h); // Width is 2 pixels and height is 1.
    encoder.set_color(reader.info().color_type);
    encoder.set_depth(reader.info().bit_depth);


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
                                                 color_attachment: false,
                                                 depth_stencil_attachment: false,
                                                 transient_attachment: false,
                                                 input_attachment: false
                                             },
                                             ImageCreateFlags::none(),
                                             Some(queue.family())).unwrap();

    let shader = crate::denoise_shader::load(device.clone()).unwrap();
    //let shader = crate::denoise_shader_compiled::load(device.clone()).unwrap();
    let compute_pipeline =  ComputePipeline::new(device.clone(), shader.entry_point("main").unwrap(), &(), None, |_| {})
            .expect("failed to create compute pipeline");

    let sampler = Sampler::new(device.clone(), SamplerCreateInfo {
        mag_filter: Filter::Linear,
        min_filter: Filter::Linear,
        address_mode: [SamplerAddressMode::Repeat; 3],
        mip_lod_bias: 1.0,
        lod: 0.0..=100.0,
        anisotropy: None,
        //anisotropy: Some(4.0),
        ..Default::default()
    }).unwrap();


    let input_view = ImageView::new_default(input_img.clone()).unwrap();
    let output_view = ImageView::new_default(result_img.clone()).unwrap();

    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();

    dbg!(layout.bindings());

    let items = [WriteDescriptorSet::image_view_sampler(0, input_view.clone(), sampler.clone()),
                     WriteDescriptorSet::image_view(1, output_view.clone())];

    let set = PersistentDescriptorSet::new(layout.clone(), items).unwrap();

    let push_constants = denoise_shader::ty::Parameters {
        Width: img_w,
        Height: img_h,
        sigma: 7.0,
        kSigma: 3.0,
        threshold: 0.4
    };

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


    //Computation itself
    let now = Instant::now();
    let command_buffer = {
        let mut builder =
            AutoCommandBufferBuilder::primary(device.clone(), queue.family(), CommandBufferUsage::OneTimeSubmit).unwrap();
        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, compute_pipeline.layout().clone(), 0, set.clone())
            .push_constants(compute_pipeline.layout().clone(), 0, push_constants)
            .dispatch([img_w, img_h, 1]).unwrap();
        builder.build().unwrap()
    };
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    println!("Execute computation itself taken {} milliseconds", now.elapsed().as_millis());

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

    /*
        // Create a query pool for occlusion queries, with 3 slots.
        let query_pool = QueryPool::new(
            device.clone(),
            QueryPoolCreateInfo {
                query_count: 2,
                ..QueryPoolCreateInfo::query_type(QueryType::Occlusion)
          },
        )
        .unwrap();
        //Copy to input image
        let command_buffer = unsafe {
            let mut builder =
                AutoCommandBufferBuilder::primary(device.clone(), queue.family(), CommandBufferUsage::OneTimeSubmit).unwrap();
            builder
            .reset_query_pool(query_pool.clone(), 0..2).unwrap()
            .begin_query(query_pool.clone(), 0, QueryControlFlags { precise: false }).unwrap()
            .copy_buffer_to_image(img_buf.clone(), input_img.clone()).unwrap()
            .end_query(query_pool.clone(), 0).unwrap()
            .begin_query(query_pool.clone(), 1, QueryControlFlags { precise: false }).unwrap()
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, compute_pipeline.layout().clone(), 0, set.clone())
            .push_constants(compute_pipeline.layout().clone(), 0, push_constants)
            .dispatch([img_w, img_h, 1]).unwrap()
            .end_query(query_pool.clone(), 1).unwrap()
            .copy_image_to_buffer(result_img.clone(), result_buf.clone()).unwrap();
            builder.build().unwrap()
        };

        */

    let rl = result_buf.read().unwrap();
    let result_bytes: Vec<u8> = rl.to_vec();

    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&result_bytes).unwrap();

}
