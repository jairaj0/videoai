import os
import time
import shutil
import inspect
import subprocess
import threading
from typing import List
import torch

# --- Hopper Architecture (H100) Max-Performance Flags ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

if not hasattr(torch.compiler, "nested_compile_region"):
    torch.compiler.nested_compile_region = lambda fn: fn

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage, OffloadMode
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_pipelines.utils.types import ImageConditioningInput, ImageReplace, ImageGuide
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.model.video_vae import get_video_chunks_number

app = FastAPI(title="LTX-2.5 Multi-Modal Production Server (LoRA-Enabled)")

print("⏳ Initializing Models into H100 80GB VRAM...")

MODEL_BASE = "models/ltx-2.5"
model_paths = ModelPaths.from_split(
    transformer_path=f"{MODEL_BASE}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    text_encoder_path=f"{MODEL_BASE}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    video_vae_path=f"{MODEL_BASE}/vae/ltx-2.5-video-vae-bf16.safetensors",
    audio_vae_path=f"{MODEL_BASE}/vae/ltx-2.5-audio-vae-bf16.safetensors",
)

offload_choice = getattr(OffloadMode, "NONE", None)

# Exact LoRA path verification matching repository scan
distilled_lora_list = []
distilled_lora_path = f"{MODEL_BASE}/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
if os.path.exists(distilled_lora_path):
    print(f"✅ Loading Verified Distilled LoRA: {distilled_lora_path}")
    distilled_lora_list.append(distilled_lora_path)
else:
    print(f"⚠️ Notice: LoRA not found at {distilled_lora_path}, running base weights.")

loras_list = []

pipe = A2VidPipelineTwoStage(
    model_paths=model_paths,
    distilled_lora=distilled_lora_list,
    spatial_upsampler_path=f"{MODEL_BASE}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    loras=loras_list,
    device=torch.device("cuda:0"),
    offload_mode=offload_choice,
)

print("🚀 Pipeline loaded with verified LoRA! API Server Ready on port 8000.")

JOBS = {}
gpu_lock = threading.Lock()

def create_binding(cls, frame_index: int):
    sig = inspect.signature(cls.__init__)
    kwargs = {}
    for p in sig.parameters:
        if p in ("frame_index", "frame_idx", "frame", "target_frame", "index"):
            kwargs[p] = frame_index
    try:
        return cls(**kwargs)
    except Exception:
        try:
            return cls(frame_index)
        except Exception:
            return cls()

def make_img_input(file_path: str, strength: float, binding_obj):
    sig = inspect.signature(ImageConditioningInput.__init__)
    param_names = [p for p in sig.parameters.keys() if p != "self"]
    
    kwargs = {}
    for p in param_names:
        if p in ("path", "image_path", "file_path", "img_path"):
            kwargs[p] = file_path
        elif p == "strength":
            kwargs[p] = strength
        elif p in ("binding", "replace", "guide"):
            kwargs[p] = binding_obj

    if len(kwargs) == len(param_names) or "path" in kwargs or "image_path" in kwargs:
        try:
            return ImageConditioningInput(**kwargs)
        except TypeError:
            pass

    try:
        return ImageConditioningInput(file_path, strength, binding_obj)
    except Exception:
        return ImageConditioningInput(file_path)

def build_multi_asset_conditioning(first_frame_path: str = None, ref_assets: list = None):
    """
    Sirf Frame 0 ko base canvas banata hai.
    Timeline par random frames replace nahi karta taaki video slideshow na bane.
    """
    conditioning_list = []
    
    start_canvas = first_frame_path
    if not start_canvas and ref_assets and len(ref_assets) > 0:
        start_canvas = ref_assets[0]

    if start_canvas and os.path.exists(start_canvas):
        binding = create_binding(ImageReplace, frame_index=0)
        conditioning_list.append(make_img_input(start_canvas, 1.0, binding))

    if ref_assets:
        for asset_path in ref_assets:
            if asset_path and os.path.exists(asset_path) and asset_path != start_canvas:
                try:
                    binding = create_binding(ImageGuide, frame_index=0)
                    conditioning_list.append(make_img_input(asset_path, 0.70, binding))
                except Exception:
                    pass
                        
    return conditioning_list

def render_worker(
    job_id: str,
    raw_audio: str,
    first_frame_img: str,
    ref_image_paths: list,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    frame_rate: float,
    duration: float,
    cfg_scale: float,
    modality_scale: float,
    output_path: str
):
    stereo_audio = f"/tmp/audio_{job_id}_stereo.wav"
    raw_video_tmp = f"/tmp/raw_render_{job_id}.mp4"
    files_to_clean = [raw_audio, stereo_audio, first_frame_img, raw_video_tmp] + (ref_image_paths or [])

    try:
        with gpu_lock:
            torch.cuda.empty_cache()
            JOBS[job_id]["status"] = "processing"

            cmd = [
                "ffmpeg", "-y", "-i", raw_audio,
                "-ac", "2", "-ar", "44100", stereo_audio
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            images_list = build_multi_asset_conditioning(first_frame_img, ref_image_paths)
            guider_params = MultiModalGuiderParams(cfg_scale=cfg_scale, modality_scale=modality_scale)

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=seed,
                        height=height,
                        width=width,
                        num_frames=None,
                        frame_rate=frame_rate,
                        num_inference_steps=steps,
                        video_guider_params=guider_params,
                        images=images_list,
                        audio_path=stereo_audio,
                        audio_start_time=0.0,
                        audio_max_duration=duration,
                    )

                chunks_num = 1
                try:
                    chunks_num = get_video_chunks_number(result.num_frames, result.tiling_config)
                except Exception:
                    pass

                generated_audio = getattr(result, "audio", None)
                encode_video(
                    video=result.video,
                    fps=frame_rate,
                    audio=generated_audio if generated_audio is not None else None,
                    output_path=raw_video_tmp,
                    video_chunks_number=chunks_num,
                )

            lossless_cmd = [
                "ffmpeg", "-y", "-i", raw_video_tmp,
                "-c:v", "libx264", "-crf", "16", "-preset", "fast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                output_path
            ]
            remux_res = subprocess.run(lossless_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if remux_res.returncode != 0:
                shutil.copyfile(raw_video_tmp, output_path)

            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["output_path"] = output_path

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
    finally:
        torch.cuda.empty_cache()
        for f in files_to_clean:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

@app.get("/health")
def health():
    return {
        "status": "ready",
        "device": torch.cuda.get_device_name(0),
        "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
        "active_jobs": len(JOBS),
        "distilled_lora_loaded": len(distilled_lora_list) > 0
    }

@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    first_frame: UploadFile = File(None),
    ref_images: List[UploadFile] = File([]),
    prompt: str = Form(...),
    negative_prompt: str = Form("worst quality, blurry, distorted, jitter, low resolution"),
    seed: int = Form(42),
    steps: int = Form(20),
    height: int = Form(1088),
    width: int = Form(1920),
    frame_rate: float = Form(24.0),
    duration: float = Form(5.0),
    cfg_scale: float = Form(3.0),
    modality_scale: float = Form(0.65),
    output_path: str = Form(None)
):
    timestamp = int(time.time())
    job_id = f"render_{seed}_{timestamp}"

    raw_audio = f"/tmp/raw_audio_{job_id}_{audio.filename}"
    with open(raw_audio, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    first_frame_path = None
    if first_frame and first_frame.filename:
        ext = os.path.splitext(first_frame.filename)[1] or ".png"
        first_frame_path = f"/tmp/first_frame_{job_id}{ext}"
        with open(first_frame_path, "wb") as buffer:
            shutil.copyfileobj(first_frame.file, buffer)

    saved_ref_paths = []
    if ref_images:
        for idx, r_file in enumerate(ref_images):
            if r_file.filename:
                r_ext = os.path.splitext(r_file.filename)[1] or ".png"
                r_path = f"/tmp/ref_{job_id}_{idx}{r_ext}"
                with open(r_path, "wb") as buffer:
                    shutil.copyfileobj(r_file.file, buffer)
                saved_ref_paths.append(r_path)

    if not output_path:
        output_path = f"/workspace/LTX-2/outputs/{job_id}.mp4"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    JOBS[job_id] = {
        "status": "queued",
        "output_path": output_path,
        "created_at": timestamp
    }

    background_tasks.add_task(
        render_worker,
        job_id,
        raw_audio,
        first_frame_path,
        saved_ref_paths,
        prompt,
        negative_prompt,
        seed,
        steps,
        width,
        height,
        frame_rate,
        duration,
        cfg_scale,
        modality_scale,
        output_path
    )

    return {"job_id": job_id, "status": "queued"}

@app.get("/status/{job_id}")
def check_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JOBS[job_id]

@app.get("/download/{job_id}")
def download_video(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet.")
    return FileResponse(path=job["output_path"], media_type="video/mp4", filename=os.path.basename(job["output_path"]))
