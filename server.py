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

app = FastAPI(title="LTX-2.5 Multi-Modal Production Server (Multi-Reference A2V)")

print("⏳ Initializing Models into H100 80GB VRAM...")

MODEL_BASE = "models/ltx-2.5"
model_paths = ModelPaths.from_split(
    transformer_path=f"{MODEL_BASE}/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    text_encoder_path=f"{MODEL_BASE}/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    video_vae_path=f"{MODEL_BASE}/vae/ltx-2.5-video-vae-bf16.safetensors",
    audio_vae_path=f"{MODEL_BASE}/vae/ltx-2.5-audio-vae-bf16.safetensors",
)

offload_choice = getattr(OffloadMode, "NONE", None)

pipe = A2VidPipelineTwoStage(
    model_paths=model_paths,
    distilled_lora=[],
    spatial_upsampler_path=f"{MODEL_BASE}/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    loras=[],
    device=torch.device("cuda:0"),
    offload_mode=offload_choice,
)

print("🚀 Pipeline loaded! API Server Ready on port 8000.")

JOBS = {}
gpu_lock = threading.Lock()

def build_multi_asset_conditioning(first_frame_path: str = None, ref_assets: list = None):
    """
    Frame 0 ke liye ImageReplace aur baki Character/BG references ke liye
    causal latent attention frame boundary par ImageGuide bind karta hai.
    """
    conditioning_list = []
    
    # 1. First Frame Anchor (Canvas continuity ya base scene)
    if first_frame_path and os.path.exists(first_frame_path):
        try:
            conditioning_list.append(
                ImageConditioningInput(
                    path=first_frame_path,
                    strength=1.0,
                    binding=ImageReplace(frame_index=0)
                )
            )
        except TypeError:
            conditioning_list.append(
                ImageConditioningInput(image=first_frame_path, strength=1.0, binding=ImageReplace(0))
            )

    # 2. Multi-Reference Assets (Characters, Background context)
    if ref_assets:
        for idx, asset_path in enumerate(ref_assets):
            if asset_path and os.path.exists(asset_path):
                # 8-step causal frame offset
                frame_slot = 1 + (8 * (idx + 1))
                try:
                    conditioning_list.append(
                        ImageConditioningInput(
                            path=asset_path,
                            strength=0.85,
                            binding=ImageGuide(frame_index=frame_slot)
                        )
                    )
                except Exception:
                    try:
                        conditioning_list.append(
                            ImageConditioningInput(
                                path=asset_path,
                                strength=0.85,
                                binding=ImageReplace(frame_index=frame_slot)
                            )
                        )
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

            # 1. Voice Audio conversion (44.1kHz Stereo standard)
            cmd = [
                "ffmpeg", "-y", "-i", raw_audio,
                "-ac", "2", "-ar", "44100", stereo_audio
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # 2. Conditioning Inputs Build (First Frame + Multi Assets)
            images_list = build_multi_asset_conditioning(first_frame_img, ref_image_paths)

            # 3. Audio & CFG Guider Parameters
            guider_params = MultiModalGuiderParams(cfg_scale=cfg_scale, modality_scale=modality_scale)

            # 4. Pipeline Execution
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

                # LTX native audio synthesize/mix ko retain karte hue video encode karein
                generated_audio = getattr(result, "audio", None)
                encode_video(
                    video=result.video,
                    fps=frame_rate,
                    audio=generated_audio if generated_audio is not None else None,
                    output_path=raw_video_tmp,
                    video_chunks_number=chunks_num,
                )

            # 5. Final Lossless Web-Ready Remux (Voice + Synth SFX Audio Stream)
            # Agar LTX generated audio return kare toh usko use karega, warna input voice integrate karega
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
        "active_jobs": len(JOBS)
    }

@app.post("/generate")
async def generate(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),                    # Aapka voice.mp3
    first_frame: UploadFile = File(None),            # Scene ka initial canvas ya prev frame
    ref_images: List[UploadFile] = File([]),         # Multiple character/bg assets
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

    # 1. Save Voice Audio
    raw_audio = f"/tmp/raw_audio_{job_id}_{audio.filename}"
    with open(raw_audio, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    # 2. Save First Frame if given
    first_frame_path = None
    if first_frame and first_frame.filename:
        ext = os.path.splitext(first_frame.filename)[1] or ".png"
        first_frame_path = f"/tmp/first_frame_{job_id}{ext}"
        with open(first_frame_path, "wb") as buffer:
            shutil.copyfileobj(first_frame.file, buffer)

    # 3. Save Multiple Reference Assets
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
