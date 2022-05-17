use std::sync::Arc;
use std::time::Instant;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, Queue};
use vulkano::image::{ImageAccess, StorageImage};
use vulkano::image::view::ImageView;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sampler::Sampler;
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano::format::Format;
use crate::{DenoiseParams, ShaderParams, UsingShader};

pub(crate) fn denoise(device: Arc<Device>, queue: Arc<Queue>, input_img: Arc<StorageImage>, result_img: Arc<StorageImage>,
                      sampler: Arc<Sampler>, denoise_params: DenoiseParams, use_hsv: bool) {

    let shader = crate::generated::get_denoise_shader(device.clone(), result_img.format(), UsingShader::Compute, use_hsv);

    let compute_pipeline = ComputePipeline::new(device.clone(), shader.entry_point("main").unwrap(), &(), None, |_| {})
            .expect("failed to create compute pipeline");

    let input_view = ImageView::new_default(input_img.clone()).unwrap();
    let output_view = ImageView::new_default(result_img.clone()).unwrap();

    let layout = compute_pipeline.layout().set_layouts().get(0).unwrap();

    //dbg!(layout);

    let items = [WriteDescriptorSet::image_view_sampler(0, input_view.clone(), sampler.clone()),
        WriteDescriptorSet::image_view(1, output_view.clone())];

    let set = PersistentDescriptorSet::new(layout.clone(), items).unwrap();

    let img_w = input_img.dimensions().width();
    let img_h = input_img.dimensions().height();

    let push_constants = ShaderParams::new(img_w, img_h, denoise_params);

    //Computation itself
    let now = Instant::now();
    let command_buffer = {
        let mut builder =
                 AutoCommandBufferBuilder::primary(device.clone(), queue.family(), CommandBufferUsage::OneTimeSubmit).unwrap();
        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(PipelineBindPoint::Compute, compute_pipeline.layout().clone(), 0, set.clone())
            .push_constants(compute_pipeline.layout().clone(), 0, push_constants)
            .dispatch([img_w / 8, img_h / 8, 1]).unwrap();
        builder.build().unwrap()
    };
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    println!("Execute computation itself taken {} milliseconds", now.elapsed().as_millis());
}