# smart_denoise_rs
Port of https://github.com/BrutPitt/glslSmartDeNoise to vulkan compute shaders

```
Usage: denoise_image [OPTIONS] --filename-in <FILENAME_IN> --filename-out <FILENAME_OUT> --shader-type <SHADER_TYPE> --algo <ALGO>

Options:
  -f, --filename-in <FILENAME_IN>
          Path to the input file. Only png is currently accepted

      --filename-out <FILENAME_OUT>
          Path to the input png file. Will use same format as the input one

      --shader-type <SHADER_TYPE>
          Which shader type to use
          
          [possible values: fragment, compute]

      --sigma <SIGMA>
          Sigma parameter

      --k-sigma <K_SIGMA>
          kSigma parameter

      --threshold <THRESHOLD>
          threshold parameter

      --use-hsv
          Process denoise in HSV (H & V actually) space

      --algo <ALGO>
          Using algorythm

          Possible values:
          - smart:  Smart denoise, reimplementation of https://github.com/BrutPitt/glslSmartDeNoise/
          - radial: Radial denoise. Better for thin lines like hairs, leaves, grass, etc

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
```
