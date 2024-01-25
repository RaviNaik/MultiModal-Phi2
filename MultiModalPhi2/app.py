import gradio as gr
from PIL import Image
from inference.main import MultiModalPhi2

messages = []

multimodal_phi2 = MultiModalPhi2(
    modelname_or_path="RaviNaik/Llava-Phi2",
    temperature=0.2,
    max_new_tokens=1024,
    device="cpu",
)


def add_content(chatbot, text, image, audio_upload, audio_mic) -> gr.Chatbot:
    textflag, imageflag, audioflag = False, False, False
    if text not in ["", None]:
        chatbot.append((text, None))
        textflag = True
    if image is not None:
        chatbot.append(((image,), None))
        imageflag = True
    if audio_mic is not None:
        chatbot.append(((audio_mic,), None))
        audioflag = True
    else:
        if audio_upload is not None:
            chatbot.append(((audio_upload,), None))
            audioflag = True
    if not any([textflag, imageflag, audioflag]):
        # Raise an error if neither text nor file is provided
        raise gr.Error("Enter a valid text, image or audio")
    return chatbot


def clear_data():
    return {prompt: None, image: None, audio_upload: None, audio_mic: None, chatbot: []}


def run(history, text, image, audio_upload, audio_mic):
    if text in [None, ""]:
        text = None

    if audio_upload is not None:
        audio = audio_upload
    elif audio_mic is not None:
        audio = audio_mic
    else:
        audio = None

    print("text", text)
    print("image", image)
    print("audio", audio)

    if image is not None:
        image = Image.open(image)
    outputs = multimodal_phi2(text, audio, image)
    # outputs = ""

    history.append((None, outputs.title()))
    return history, None, None, None, None


with gr.Blocks() as demo:
    gr.Markdown("## MulitModal Phi2 Model Pretraining and Finetuning from Scratch")
    gr.Markdown(
        """This is a multimodal implementation of [Phi2](https://huggingface.co/microsoft/phi-2) model.
        
        Please find the source code and training details [here](https://github.com/RaviNaik/ERA-CAPSTONE/MultiModalPhi2).
        
        ### Details:
        1. LLM Backbone: [Phi2](https://huggingface.co/microsoft/phi-2)
        2. Vision Tower: [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)
        3. Audio Model: [Whisper Tiny](https://huggingface.co/openai/whisper-tiny) 
        4. Pretraining Dataset: [LAION-CC-SBU dataset with BLIP captions(200k samples)](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
        5. Finetuning Dataset: [Instruct 150k dataset based on COCO](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
        6. Finetuned Model: [RaviNaik/Llava-Phi2](https://huggingface.co/RaviNaik/Llava-Phi2)
        """
    )
    with gr.Row():
        with gr.Column(scale=4):
            # Creating a column with a scale of 6
            with gr.Box():
                with gr.Row():
                    # Adding a Textbox with a placeholder "write prompt"
                    prompt = gr.Textbox(
                        placeholder="Enter Prompt", lines=2, label="Query", value=None
                    )
                # Creating a column with a scale of 2
                with gr.Row():
                    # Adding image
                    image = gr.Image(type="filepath", value=None)
                # Creating a column with a scale of 2
                with gr.Row():
                    # Add audio
                    audio_upload = gr.Audio(source="upload", type="filepath")
                    audio_mic = gr.Audio(
                        source="microphone", type="filepath", format="mp3"
                    )

        with gr.Column(scale=8):
            with gr.Box():
                with gr.Row():
                    chatbot = gr.Chatbot(
                        avatar_images=("🧑", "🤖"),
                        height=550,
                    )
                with gr.Row():
                    # Adding a Button
                    submit = gr.Button()
                    clear = gr.Button(value="Clear")

    submit.click(
        add_content,
        inputs=[chatbot, prompt, image, audio_upload, audio_mic],
        outputs=[chatbot],
    ).success(
        run,
        inputs=[chatbot, prompt, image, audio_upload, audio_mic],
        outputs=[chatbot, prompt, image, audio_upload, audio_mic],
    )

    clear.click(
        clear_data,
        outputs=[prompt, image, audio_upload, audio_mic, chatbot],
    )

demo.launch(server_port=8881)
