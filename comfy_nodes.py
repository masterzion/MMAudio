"""ComfyUI wrapper for MMAudio video-to-synchronized-audio inference."""
from pathlib import Path
import threading

import torch
import torchaudio
import folder_paths
from comfy_api.latest import io

MODEL_VARIANTS = ["small_16k", "small_44k", "medium_44k", "large_44k", "large_44k_v2"]


class MMAudioVideoSource_UNUSED:
    @classmethod
    def define_schema(cls):
        files = [f for f in folder_paths.get_filename_list("input")
                 if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv"))]
        return io.Schema(
            node_id="MMAudioVideoSource",
            display_name="MMAudio: Video Source",
            category="MMAudio",
            inputs=[io.Combo.Input("video_path", options=sorted(files), upload=io.UploadType.video)],
            outputs=[io.String.Output(display_name="video_path")],
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_path": ("STRING", {
            "default": "",
            "multiline": False,
            "video_upload": True,
        })}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "source"
    CATEGORY = "MMAudio"

    def source(self, video_path):
        return (folder_paths.get_annotated_filepath(video_path),)

    @classmethod
    def execute(cls, video_path):
        return io.NodeOutput(folder_paths.get_annotated_filepath(video_path))

ROOT = Path(__file__).resolve().parent
_CACHE = {}
_LOCK = threading.Lock()


def _get_pipeline(variant, device, dtype, num_steps):
    from .mmaudio.eval_utils import all_model_cfg
    from .mmaudio.model.flow_matching import FlowMatching
    from .mmaudio.model.networks import get_my_mmaudio
    from .mmaudio.model.utils.features_utils import FeaturesUtils
    key = (variant, device, str(dtype), num_steps)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        cfg = all_model_cfg[variant]
        old = Path.cwd()
        try:
            # The upstream configs use paths relative to the repository root.
            import os
            os.chdir(ROOT)
            cfg.download_if_needed()
            net = get_my_mmaudio(cfg.model_name).to(device, dtype).eval()
            net.load_weights(torch.load(cfg.model_path, map_location=device, weights_only=True))
            features = FeaturesUtils(
                tod_vae_ckpt=cfg.vae_path,
                synchformer_ckpt=cfg.synchformer_ckpt,
                enable_conditions=True,
                mode=cfg.mode,
                bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path,
                need_vae_encoder=False,
            ).to(device, dtype).eval()
        finally:
            os.chdir(old)
        pipe = (cfg, net, features, FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps))
        _CACHE[key] = pipe
        return pipe


class MMAudioVideoToAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "prompt": ("STRING", {"default": "synchronized natural sound", "multiline": True}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "variant": (MODEL_VARIANTS, {"default": "large_44k_v2"}),
            "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1, "tooltip": "0 detects the input video duration automatically; positive values override it."}),
            "cfg_strength": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            "num_steps": ("INT", {"default": 25, "min": 1, "max": 100}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_audio"
    CATEGORY = "MMAudio"

    @torch.inference_mode()
    def generate_audio(self, video, prompt, negative_prompt, variant, duration, cfg_strength, num_steps, seed):
        # qamba: automatic source-video duration
        import math
        if duration == 0:
            duration = float(video.get_duration())
        if not math.isfinite(duration) or not 0.5 <= duration <= 60:
            raise ValueError("MMAudio supports videos from 0.5 to 60 seconds; trim longer clips before generating audio")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        from .mmaudio.eval_utils import generate, load_video
        cfg, net, features, fm = _get_pipeline(variant, device, dtype, num_steps)
        source = video.get_stream_source()
        if not isinstance(source, (str, Path)):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(source.getvalue())
                source = tmp.name
        info = load_video(Path(source), duration)
        seq_cfg = cfg.seq_cfg
        seq_cfg.duration = info.duration_sec
        net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
        rng = torch.Generator(device=device)
        rng.manual_seed(int(seed))
        audio = generate(
            info.clip_frames.unsqueeze(0), info.sync_frames.unsqueeze(0), [prompt],
            negative_text=[negative_prompt], feature_utils=features, net=net, fm=fm,
            rng=rng, cfg_strength=cfg_strength,
        ).float().cpu()[0]
        return ({"waveform": audio.unsqueeze(0), "sample_rate": seq_cfg.sampling_rate},)


NODE_CLASS_MAPPINGS = {"MMAudioVideoToAudio": MMAudioVideoToAudio}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MMAudioVideoToAudio": "MMAudio: Video to Synchronized Audio",
}
