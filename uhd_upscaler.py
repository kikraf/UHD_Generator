import os

from PIL import Image

from modules import shared, images, devices, scripts, scripts_postprocessing, ui_common, generation_parameters_copypaste
from modules.shared import opts


def run_postprocessing(extras_mode, imagen, image_folder, input_dir, output_dir, show_extras_results, *args, save_output: bool = True):
    print("entrando a postprocessing")
    devices.torch_gc()

    shared.state.begin()
    shared.state.job = 'extras'

    image_data = []
    image_names = []
    outputs = []

    outpath = opts.outdir_samples or opts.outdir_extras_samples

    infotext = ''

    image_paths = image_folder.split(";")

    for img in image_paths:
        image = Image.open(img)
        image_data.append(image)
        name = os.path.splitext(img.orig_name)[0]
        image_names.append(name)

        width, height = image.size

        parts = []
        parts_names = []
        parts_outputs = []
        image_cropped=False

        if width > 256:
            image_cropped = True
            parts.append(image.crop((0, 0, width / 2, height / 2)))
            parts.append(image.crop((width / 2, 0, width, height / 2)))
            parts.append(image.crop((0, height / 2, width / 2, height)))
            parts.append(image.crop((width / 2, height / 2, width, height)))

            parts_names.append(name.split(".")[0]+"_0."+name.split(".")[1])
            parts_names.append(name.split(".")[0]+"_1."+name.split(".")[1])
            parts_names.append(name.split(".")[0]+"_2."+name.split(".")[1])
            parts_names.append(name.split(".")[0]+"_3."+name.split(".")[1])

            # Guardar las partes
            parts[0].save(parts_names[0])
            parts[1].save(parts_names[1])
            parts[2].save(parts_names[2])
            parts[3].save(parts_names[3])

            image.close()

            for image,name in zip(parts, parts_names): 

                shared.state.textinfo = name

                existing_pnginfo = image.info or {}

                pp = scripts_postprocessing.PostprocessedImage(image.convert("RGB"))

                scripts.scripts_postproc.run(pp, args)

                if opts.use_original_name_batch and name is not None:
                    basename = os.path.splitext(os.path.basename(name))[0]
                else:
                    basename = ''

                infotext = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in pp.info.items() if v is not None])

                if opts.enable_pnginfo:
                    pp.image.info = existing_pnginfo
                    pp.image.info["postprocessing"] = infotext

                if save_output:
                    images.save_image(pp.image, path=outpath, basename=basename, seed=None, prompt=None, extension=opts.samples_format, info=infotext, short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=existing_pnginfo, forced_filename=None)

                parts_outputs.append(pp.image)

            img = Image.new('RGB', (images[0].width*2, images[0].width*2))

            for i in range(2):
                for j in range(2):
                    idx = j + 2 * i
                    if idx >= len(parts_outputs):
                        break
                    img.paste(parts_outputs[idx], (i * parts_outputs[0].width, j * parts_outputs[0].width))

            outputs.append(img)
    # elif extras_mode == 2:
    #     assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
    #     assert input_dir, 'input directory not selected'

    #     image_list = shared.listfiles(input_dir)
    #     for filename in image_list:
    #         try:
    #             image = Image.open(filename)
    #         except Exception:
    #             continue
    #         image_data.append(image)
    #         image_names.append(filename)
    # else:
    #     assert image, 'image not selected'

    #     image_data.append(image)
    #     image_names.append(None)

    # if extras_mode == 2 and output_dir != '':
    #     outpath = output_dir
    # else:




    devices.torch_gc()

    return outputs, ui_common.plaintext_to_html(infotext), ''


def run_extras(extras_mode, resize_mode, image, image_folder, input_dir, output_dir, show_extras_results, gfpgan_visibility, codeformer_visibility, codeformer_weight, upscaling_resize, upscaling_resize_w, upscaling_resize_h, upscaling_crop, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility, upscale_first: bool, save_output: bool = True):
    """old handler for API"""

    args = scripts.scripts_postproc.create_args_for_run({
        "Upscale": {
            "upscale_mode": resize_mode,
            "upscale_by": upscaling_resize,
            "upscale_to_width": upscaling_resize_w,
            "upscale_to_height": upscaling_resize_h,
            "upscale_crop": upscaling_crop,
            "upscaler_1_name": extras_upscaler_1,
            "upscaler_2_name": extras_upscaler_2,
            "upscaler_2_visibility": extras_upscaler_2_visibility,
        },
        "GFPGAN": {
            "gfpgan_visibility": gfpgan_visibility,
        },
        "CodeFormer": {
            "codeformer_visibility": codeformer_visibility,
            "codeformer_weight": codeformer_weight,
        },
    })

    return run_postprocessing(extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output=save_output)
