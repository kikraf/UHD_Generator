import gradio as gr
from PIL import Image
from modules import scripts, shared, gfpgan_model, codeformer_model, ui_common, call_queue, postprocessing
from modules import script_callbacks, shared
import os
from extensions.UHD_Generator import uhd_upscaler

def load_paths(file, textbox):
    paths=""
    for path in file:
        paths=paths+";"
    textbox=paths
    return textbox

def show_images(image_paths):
    images = []
    for path in image_paths:
        # Cargar imagen usando PIL.Image.open()
        image = Image.open(path)
        images.append(image)

    return images

def change_scale(mult):
    exp = 9 + mult
    size =pow(2, exp)
    return size


def on_ui_tabs():
    tab_index = gr.State(value=1)

    with gr.Blocks() as UHD_interface:
        with gr.Row().style(equal_height=True, variant='compact'):
            with gr.Column(variant='compact'):
                with gr.Tabs(elem_id="mode_uhd"):
                    with gr.TabItem('Escalado Multiple', elem_id='UHD_autoupscale') as tab_autoupscale:

                        # Crear un selector de archivos
                        image_batch = gr.File(label="Batch Process", file_count="multiple", interactive=True, type="file", elem_id="extras_image_batch")

                        # Crear una galeria para mostrar las imagenes
                        input_gallery = gr.Gallery(label="Imagnes Seleccionadas").style(grid=4)

                        scale_mult = gr.Slider(0, 10, step=1, value=1)
                        scale_to = gr.Textbox(label="Escalar a", value=1024, interactive=False)
                        #Unused elements
                        extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="extras_image")
                        extras_image.visible = False
                        extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir")
                        extras_batch_input_dir.visible = False
                        extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir")
                        extras_batch_output_dir.visible = False
                        show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results")
                        show_extras_results.visible = False
                        #tab_autoupscale.select(fn=lambda: 1, inputs=[], outputs=[tab_index])
                        script_inputs = scripts.scripts_postproc.setup_ui()

                    for s in script_inputs[1:]:
                        if s.elem_id == "extras_upscaler_1":
                            s.value = "R-ESRGAN 4x+"
                        elif s.elem_id == "extras_upscaling_resize_w":
                            s.value = scale_to.value
                        elif s.elem_id == "extras_upscaling_resize_h":
                            s.value = scale_to.value
                        print("+ "+s.elem_id+": "+str(s.value))

                        # if s.elem_id != "extras_upscaling_resize_w" and s.elem_id !="extras_upscaler_1":
                        #     s.visible=False
                    


                    
                    submit = gr.Button('Generate', elem_id="uhd_generate", variant='primary')

                    
            #Preview/progress
            with gr.Column(variant="compact"):
                result_images, html_info_x, html_info, html_log = ui_common.create_output_panel("UHD", shared.opts.outdir_extras_samples)



        # image_batch.change(
        #     fn=load_paths,
        #     inputs=[image_batch, extras_batch_input_dir],
        #     outputs=[extras_batch_input_dir]
        # )

        scale_mult.change(
            fn=change_scale,
            inputs=[scale_mult],
            outputs=[scale_to]
        )

        submit.click(
            fn=call_queue.wrap_gradio_gpu_call(uhd_upscaler.run_postprocessing, extra_outputs=[None, '']),
            inputs=[
                tab_index,
                extras_image,
                image_batch,
                extras_batch_input_dir,
                extras_batch_output_dir,
                show_extras_results,
                *script_inputs,
                True
            ],
            outputs=[
                result_images,
                html_info_x,
                html_info,
            ]
        )

        image_batch.change(
            fn=show_images,
            inputs=image_batch,
            outputs=input_gallery
        )

    return (UHD_interface, "Generador UltraHD", "UHD_interface"),


script_callbacks.on_ui_tabs(on_ui_tabs)
