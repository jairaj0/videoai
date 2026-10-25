import os
import time
import shutil
import inspect
import subprocess
import threading
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
from fastapi.responses import FileResponse, RedirectResponse
from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage, OffloadMode
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_pipelines.utils.types import ImageConditioningInput, ImageReplace
from ltx_pipelines.utils.media_io import encode_video
from ltx_core.model.video_vae import get_video_chunks_number

app = FastAPI(title="LTX-2.5 A2V Master Production Server")

print("⏳ Initializing A2V Two-Stage Models directly into H100 80GB VRAM...")

MODEL_BASE = "models/ltx-2.5"
model_paths = ModelPaths.from_split(
    transformer_path=f"{MODEL_BASE}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    text_encoder_path=f"{MODEL_BASE}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    video_vae_path=f"{MODEL_BASE}/vae/ltx-2.5-video-vae-bf16.safetensors",
    audio_vae_path=f"{MODEL_BASE}/vae/ltx-2.5-audio-vae-bf16.safetensors",
)

# Zero Offloading: Complete pipeline locked in HBM3 VRAM for maximum speed
offload_choice = getattr(OffloadMode, "NONE", None)

pipe = A2VidPipelineTwoStage(
    model_paths=model_paths,
    distilled_lora=[],
    spatial_upsampler_path=f"{MODEL_BASE}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    loras=[],
    device=torch.device("cuda:0"),
    offload_mode=offload_choice,
)

print("🚀 Pipeline loaded into HBM3 VRAM! API Server Ready on port 8000.")

JOBS = {}
gpu_lock = threading.Lock()

def create_image_replace_binding(frame_index: int = 0):
    sig = inspect.signature(ImageReplace.__init__)
    params = sig.parameters
    kwargs = {}
    for p in params:
        if p in ("frame_index", "frame_idx", "frame", "target_frame", "index"):
            kwargs[p] = frame_index
    try:
        return ImageReplace(**kwargs)
    except Exception:
        try:
            return ImageReplace(frame_index)
        except Exception:
            return ImageReplace()

def build_image_input(image_path: str, strength: float = 1.0):
    binding = create_image_replace_binding(frame_index=0)
    sig = inspect.signature(ImageConditioningInput.__init__)
    params = sig.parameters
    kwargs = {}
    for p_name in params:
        if p_name == "self":
            continue
        if p_name in ("image", "path", "image_path", "file_path"):
            kwargs[p_name] = image_path
        elif p_name == "strength":
            kwargs[p_name] = strength
        elif p_name == "binding":
            kwargs[p_name] = binding

    return ImageConditioningInput(**kwargs)

def render_worker(
    job_id: str,
    raw_audio: str,
    temp_image: str,
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
    try:
        with gpu_lock:
            torch.cuda.empty_cache()
            JOBS[job_id]["status"] = "processing"

            # 1. Audio pre-processing (Stereo 44.1kHz)
            cmd = [
                "ffmpeg", "-y", "-i", raw_audio,
                "-ac", "2",
                "-ar", "44100",
                stereo_audio
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise RuntimeError(f"FFmpeg audio conversion failed: {res.stderr.decode()}")

            images_list = []
            if temp_image and os.path.exists(temp_image):
                img_input = build_image_input(temp_image, strength=1.0)
                images_list.append(img_input)

            guider_params = MultiModalGuiderParams(cfg_scale=cfg_scale, modality_scale=modality_scale)

            # 2. Native BF16 & Inference Mode for H100
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

                if hasattr(result, "save"):
                    result.save(raw_video_tmp)
                elif hasattr(result, "save_video"):
                    result.save_video(raw_video_tmp)
                else:
                    chunks_num = 1
                    try:
                        chunks_num = get_video_chunks_number(result.num_frames, result.tiling_config)
                    except Exception:
                        pass

                    encode_video(
                        video=result.video,
                        fps=frame_rate,
                        audio=getattr(result, "audio", None),
                        output_path=raw_video_tmp,
                        video_chunks_number=chunks_num,
                    )

            # 3. Fast Lossless Remux (CRF 16 + Fast preset for zero block artifacts)
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
        for f in (raw_audio, stereo_audio, temp_image, raw_video_tmp):
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass

@app.get("/")
def index():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {
        "status": "ready",
        "device": torch.cuda.get_device_name(0),
        "vram_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        "vram_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
        "active_jobs": len(JOBS)
    }

@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    image: UploadFile = File(None),
    prompt: str = Form(...),
    negative_prompt: str = Form("worst quality, blurry, distorted, jitter, noisy, low resolution"),
    seed: int = Form(42),
    steps: int = Form(16),
    height: int = Form(1088),
    width: int = Form(1920),
    frame_rate: float = Form(24.0),
    duration: float = Form(4.0),
    cfg_scale: float = Form(3.0),
    modality_scale: float = Form(0.6),
    output_path: str = Form(None)
):
    timestamp = int(time.time())
    job_id = f"render_{seed}_{timestamp}"
    raw_audio = f"/tmp/raw_audio_{job_id}_{audio.filename}"
    temp_image = None

    with open(raw_audio, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    if image and image.filename:
        img_ext = os.path.splitext(image.filename)[1] or ".png"
        temp_image = f"/tmp/image_{job_id}{img_ext}"
        with open(temp_image, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

    if not output_path:
        output_path = f"/workspace/LTX-2/outputs/{job_id}.mp4"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    JOBS[job_id] = {
        "status": "queued",
        "output_path": output_path,
        "created_at": timestamp
    }

    background_tasks.add_task(
        render_worker, job_id, raw_audio, temp_image, prompt, negative_prompt,
        seed, steps, width, height, frame_rate, duration, cfg_scale, modality_scale, output_path
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Render task queued successfully."
    }

@app.get("/status/{job_id}")
def check_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JOBS[job_id]

@app.get("/download/{job_id}")
def download_video(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet.")
    
    file_path = job.get("output_path")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Rendered file missing from disk.")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=os.path.basename(file_path)
    )
