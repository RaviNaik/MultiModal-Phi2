import soundfile as sf
import librosa
import torch
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

from .model import LlavaPhiForCausalLM
from .conversation import conv_templates, SeparatorStyle

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class AudioLanguageConnector:
    def __init__(self, projection_dim):
        model_name = "microsoft/phi-2"
        self.phi2_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.phi2_tokenizer.pad_token = self.phi2_tokenizer.eos_token
        self.phi2_tokenizer.max_length = projection_dim

    def __call__(self, text):
        text = f"<audio_start> {text} <audio_end>"
        tokens = self.phi2_tokenizer(
            text, return_tensors="pt", return_attention_mask=False
        )
        return tokens


class WhisperWithProjection:
    def __init__(self, projection_dim, device):
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-tiny", device_map=device
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-tiny", device_map=device
        )
        self.model.config.forced_decoder_ids = None
        # self.audio_language_connector = AudioLanguageConnector(projection_dim)

    def __call__(self, audio):
        array, sampling_rate = sf.read(audio)
        resampled_array = librosa.resample(
            array,
            orig_sr=sampling_rate,
            target_sr=16000,
        )
        input_features = self.processor(
            resampled_array, sampling_rate=16000, return_tensors="pt"
        ).input_features
        # generate token ids
        predicted_ids = self.model.generate(input_features.to(self.device))
        # decode token ids to text
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        # audio_embeddings = self.audio_language_connector(transcription)
        return transcription


class MultiModalPhi2:
    def __init__(
        self,
        modelname_or_path="RaviNaik/Llava-Phi2",
        temperature=0.2,
        max_new_tokens=1024,
        device="cuda:0",
    ):
        self.model_name = modelname_or_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.disable_torch_init()
        self.whisper_w_proj = WhisperWithProjection(projection_dim=512, device=device)
        self.load_pretrained_model()

    def disable_torch_init(self):
        """
        Disable the redundant torch default initialization to accelerate model creation.
        """
        setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
        setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

    def load_pretrained_model(self):
        self.model = LlavaPhiForCausalLM.from_pretrained(
            self.model_name, device_map=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_name)
        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(
            self.model.config, "mm_use_im_patch_token", True
        )
        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )

    def tokenizer_image_token(
        self,
        prompt,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors=None,
    ):
        prompt_chunks = [
            tokenizer(chunk).input_ids for chunk in prompt.split("<image>")
        ]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if (
            len(prompt_chunks) > 0
            and len(prompt_chunks[0]) > 0
            and prompt_chunks[0][0] == tokenizer.bos_token_id
        ):
            offset = 1
            input_ids.append(prompt_chunks[0][0])
        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == "pt":
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f"Unsupported tensor type: {return_tensors}")
        return input_ids

    def __call__(self, text, audio, image):
        if text is None:
            text = ""
        if image is not None:
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + text
            )
            conv = conv_templates["phi-2_v0"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = self.tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0)

            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ].to(self.device)
        else:
            qs = text
            conv = conv_templates["phi-2_v0"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]

            image_tensor = None

        if audio is not None:
            audio_transcript = self.whisper_w_proj(audio)
            audio_embed = self.tokenizer(audio_transcript, return_tensors="pt")[
                "input_ids"
            ]
            input_ids = torch.concat([input_ids, audio_embed], dim=1)
        input_ids = input_ids.to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with torch.inference_mode():
            if image is not None:
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                    pad_token_id=self.tokenizer.eos_token_id,  # Pad token
                    use_cache=True,
                )
            else:
                output_ids = self.model.generate(
                    input_ids,
                    do_sample=True,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                    pad_token_id=self.tokenizer.eos_token_id,  # Pad token
                    use_cache=True,
                )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            (input_ids != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs
