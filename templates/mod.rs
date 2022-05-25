use handlebars::Handlebars;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

pub fn generate_shaders() {
    let mut datas = vec![];
    {
        let mut data = HashMap::new();
        //8-bit grayscale
        data.insert("output_format", "r8ui");
        data.insert("max_val", "255");
        data.insert("swizzle_vec", "r");
        data.insert("processing_t", "float");
        data.insert("output_t", "uvec4");
        data.insert("is_int_type", "true");
        datas.push(data);
    }

    {
        let mut data = HashMap::new();
        //16-bit grayscale
        data.insert("output_format", "r16ui");
        data.insert("max_val", "255");
        data.insert("swizzle_vec", "r");
        data.insert("processing_t", "float");
        data.insert("output_t", "uvec4");
        data.insert("is_int_type", "true");
        datas.push(data);
    }

    {
        let mut data = HashMap::new();
        //8-bit rgba
        data.insert("output_format", "rgba8ui");
        data.insert("max_val", "255");
        data.insert("swizzle_vec", "rgba");
        data.insert("processing_t", "vec4");
        data.insert("output_t", "uvec4");
        data.insert("is_int_type", "true");
        data.insert("is_vector_type", "true");
        datas.push(data);
    }


    {
        let mut data = HashMap::new();
        //16-bit rgba
        data.insert("output_format", "rgba16ui");
        data.insert("max_val", "65535");
        data.insert("swizzle_vec", "rgba");
        data.insert("processing_t", "vec4");
        data.insert("output_t", "uvec4");
        data.insert("is_int_type", "true");
        data.insert("is_vector_type", "true");
        datas.push(data);
    }

    {
        let mut data = HashMap::new();
        //32-bit float
        data.insert("output_format", "r32f");
        //data.insert("max_val", "1.0"); //Assuming data is already normalized
        data.insert("swizzle_vec", "r");
        data.insert("processing_t", "float");
        data.insert("output_t", "vec4");
        datas.push(data);
    }

    let shader_typed_datas: Vec<HashMap<&str, &str>> = datas.into_iter().flat_map(|mut d| {
        let mut df = d.clone();
        d.insert("compute", "compute");
        d.insert("shadert_enum_val", "UsingShader::Compute");
        df.insert("fragment", "fragment");
        df.insert("shadert_enum_val", "UsingShader::Fragment");
        vec![d, df]
    }).collect();

    let shader_typed_datas_hsv: Vec<HashMap<&str, &str>> = shader_typed_datas.into_iter()
        .flat_map(|mut d| {
            if d.get("is_vector_type").eq(&Some(&"true")) {
                let mut dhsv = d.clone();
                dhsv.insert("is_hsv", "_hsv");
                vec![d, dhsv]
            } else {
                vec![d]
            }
    }).collect();

    let format_pairs: HashMap::<String, String>  = vec![
        ("r8ui".to_string(), "R8_UINT".to_string()),
        ("r16ui".to_string(), "R16_UINT".to_string()),
        ("rgba8ui".to_string(), "R8G8B8A8_UINT".to_string()),
        ("r32f".to_string(), "R32_SFLOAT".to_string()),
        ("rgba16ui".to_string(), "R16G16B16A16_UINT".to_string())]
        .into_iter()
        .collect();

    let mut shader_mods = vec![];
    let mut shader_matchers = vec!["\n".to_string()];

    let base_path = "src/generated";




    let mut handlebars = Handlebars::new();

    for algorythm in ["Smart", "Radial"] {
        handlebars
            .register_template_file(algorythm, format!("templates/denoise_shader_{}.mustache", algorythm.to_lowercase()))
            .unwrap();

        for d in shader_typed_datas_hsv.iter() {
            let shader = handlebars.render(algorythm, d).unwrap();

            let shadert = d.get("compute").unwrap_or(&"fragment");
            let shadert_enum_val = d.get("shadert_enum_val").unwrap();

            let filtering_type = d.get("is_hsv").unwrap_or(&"");
            let is_hsv = d.get("is_hsv").is_some();

            let format = d.get("output_format").unwrap();
            let vk_type = format_pairs.get(*format).unwrap();

            let name = format!("denoise_shader_{}_{}{}{}", shadert, format, algorythm.to_lowercase(), filtering_type);
            shader_mods.push(format!("pub(crate) mod {};", &name));
            let align = 48 - (vk_type.len() + shadert_enum_val.len() + is_hsv.to_string().len());
            shader_matchers.push(format!("{:>12}(Format::{}, {}, {}, Algo::{}) => {:align$}{}::load(device.clone()),", "", vk_type, shadert_enum_val, is_hsv, algorythm, "", &name));
            let mut file = File::create(format!("{}/{}.rs", base_path, &name)).unwrap();
            file.write_all(shader.as_bytes()).unwrap();
        }
    }

    let mut file = File::create(format!("{}/mod.rs", base_path)).unwrap();
    file.write_all(shader_mods.join("\n").as_bytes()).unwrap();

    file.write_all(r#"

use std::sync::Arc;
use vulkano::format::Format;
use vulkano::device::Device;
use vulkano::shader::ShaderModule;
use crate::UsingShader;
use crate::Algo;

    pub(crate) fn get_denoise_shader(device: Arc<Device>, format: Format, shader_type: UsingShader, use_hsv: bool, algo: Algo) -> Arc<ShaderModule> {
        match (format, shader_type, use_hsv, algo) {"#.as_bytes()).unwrap();

    file.write_all(shader_matchers.join("\n").as_bytes()).unwrap();

    file.write_all(r#"
            _ => unimplemented!(),
        }.unwrap()
    }
    "#.as_bytes()).unwrap();

}