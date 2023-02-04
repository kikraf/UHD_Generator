import gradio as gr
from PIL import Image
from modules import scripts, shared, gfpgan_model, codeformer_model, ui_common, call_queue
from modules import script_callbacks, shared
import os
from extensions.UHD_Generator import uhd_upscaler

def show_images(image_paths):
    images = []
    for path in image_paths:
        # Cargar imagen usando PIL.Image.open()
        image = Image.open(path)
        images.append(image)

    return images

def select_upscaler(upscaler):
    print(upscaler)
    print (shared.sd_upscalers[upscaler].name)
    return


def on_ui_tabs():
    tab_index = gr.State(value=0)

    with gr.Blocks() as UHD_interface:
        with gr.Row().style(equal_height=False, variant='compact'):
            with gr.Column(variant='compact'):
                with gr.Tabs(elem_id="mode_uhd"):
                    with gr.TabItem('Escalado Multiple', elem_id='UHD_autoupscale') as tab_autoupscale:

                        # Crear un selector de archivos
                        image_batch = gr.File(label="Selecciona imágenes",file_count="multiple",file_types=['.jpg','.png'])
                        # Crear una galeria para mostrar las imagenes
                        input_gallery = gr.Gallery(label="Imagnes Seleccionadas").style(grid=4)
                        uhd_upscale_ratio = gr.Slider(label="Upscale Ratio", value=2, step=1, minimum=2, maximum=4)
                        #options = [1024, 2048, 4096, 8192, 16284]
                        #slider = gr.Slider(options=options, default=1024, label="Potencia de 2", type=int)
                        uhd_scaler = gr.Radio(label='Upscaler', elem_id="sp_scaler",
                                                choices=[x.name for x in shared.sd_upscalers],
                                                value=shared.sd_upscalers[0].name, type="index")
                        print("Scaler: "+ uhd_scaler.value)

                    submit = gr.Button('Generate', elem_id="extras_generate", variant='primary')

                    script_inputs = scripts.scripts_postproc.setup_ui()

                    print(script_inputs)
                    for s in script_inputs[1:]:
                        if s.elem_id != "extras_upscaling_resize_w" or s.elem_id !="extras_upscaler_1":
                            s.visible=False
                    
            #Preview/progress
            with gr.Column():
                result_images, html_info_x, html_info, html_log = ui_common.create_output_panel("extras", shared.opts.outdir_extras_samples)

        #Unused elements
        extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil", elem_id="extras_image", display=False)
        extras_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs, placeholder="A directory on the same machine where the server is running.", elem_id="extras_batch_input_dir", display=False)
        extras_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs, placeholder="Leave blank to save images to the default path.", elem_id="extras_batch_output_dir", display=False)
        show_extras_results = gr.Checkbox(label='Show result images', value=True, elem_id="extras_show_extras_results", display=False)

        tab_autoupscale.select(fn=lambda: 1, inputs=[], outputs=[tab_index])

        uhd_scaler.change(
            fn=select_upscaler,
            inputs=[uhd_scaler],
            outputs=[]
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
                *script_inputs
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
