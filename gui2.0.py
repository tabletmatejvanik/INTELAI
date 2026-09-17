import os
import time
import random
import re
import warnings
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import trimesh
import gradio as gr

# Potlačení varování z knihoven
warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from diffusers import (
    ZImagePipeline, 
    ShapEPipeline, 
    AnimateDiffPipeline, 
    MotionAdapter, 
    DDIMScheduler
)
from diffusers.utils import export_to_ply, export_to_video
from transformers import AutoModelForCausalLM, AutoTokenizer

# Složky pro ukládání
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
STL_DIR = os.path.join(SCRIPT_DIR, "stl_outputs")
VIDEO_DIR = os.path.join(SCRIPT_DIR, "video_outputs")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio_outputs")

for d in [OUTPUT_DIR, STL_DIR, VIDEO_DIR, AUDIO_DIR]:
    os.makedirs(d, exist_ok=True)

# Detekce zařízení (CUDA pro Nvidia/ROCm, XPU pro Intel Arc, popř. CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = "xpu"
else:
    DEVICE = "cpu"

print(f"Detekované zařízení pro akceleraci: {DEVICE}")

# Lazy loading modelů
z_image_pipe = None
shape_pipe = None
animate_video_pipe = None
yue_tokenizer = None
yue_model = None


def get_z_image_pipe():
    global z_image_pipe
    if z_image_pipe is None:
        print("[+] Načítám Z-Image-Turbo pipeline...")
        z_image_pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        )
        if DEVICE != "cpu":
            z_image_pipe.enable_sequential_cpu_offload(device=DEVICE)
        if hasattr(z_image_pipe, "vae") and hasattr(z_image_pipe.vae, "enable_tiling"):
            z_image_pipe.vae.enable_tiling()
        if hasattr(z_image_pipe, "vae") and hasattr(z_image_pipe.vae, "enable_slicing"):
            z_image_pipe.vae.enable_slicing()
    return z_image_pipe


def get_shape_pipe():
    global shape_pipe
    if shape_pipe is None:
        print("[+] Načítám OpenAI Shap-E pipeline (CPU / float32)...")
        shape_pipe = ShapEPipeline.from_pretrained(
            "openai/shap-e",
            torch_dtype=torch.float32,
        )
        shape_pipe.to("cpu")
    return shape_pipe


def get_animate_video_pipe():
    """Načte AnimateDiff model pro rychlé a plynulé generování videa."""
    global animate_video_pipe
    if animate_video_pipe is None:
        print("[+] Načítám AnimateDiff Video pipeline...")
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2", 
            torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32
        )
        
        animate_video_pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
        )
        
        # Optimalizace plánovače pro plynulejší render
        scheduler = DDIMScheduler.from_config(
            animate_video_pipe.scheduler.config,
            beta_schedule="linear",
            steps_offset=1,
            clip_sample=False
        )
        animate_video_pipe.scheduler = scheduler

        if DEVICE != "cpu":
            animate_video_pipe.enable_sequential_cpu_offload(device=DEVICE)
            if hasattr(animate_video_pipe, "vae"):
                animate_video_pipe.vae.enable_slicing()
                
    return animate_video_pipe


def get_yue_model():
    """Načte model YuE pro generování plnohodnotné hudby."""
    global yue_tokenizer, yue_model
    if yue_model is None:
        print("[+] Načítám YuE Music Generator (mker2/YuE-s1-7B-anneal-en-cot)...")
        model_id = "mker2/YuE-s1-7B-anneal-en-cot"
        
        yue_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        yue_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16 if DEVICE != "cpu" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if DEVICE != "cpu":
            yue_model.to(DEVICE)
            print(f"[+] YuE načten na {DEVICE}.")
        else:
            yue_model.to("cpu")
            
    return yue_tokenizer, yue_model


# --- 2D Generátor ---
def generate_image(prompt, width, height, steps, seed, randomize_seed, custom_name):
    pipe = get_z_image_pipe()
    if randomize_seed or seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    generator = torch.Generator("cpu").manual_seed(int(seed))
    print(f"\n[+] 2D Generuji: '{prompt}'")
    
    image = pipe(
        prompt=prompt,
        height=int(height),
        width=int(width),
        num_inference_steps=int(steps),
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    
    clean_name = custom_name.strip() if custom_name else ""
    clean_name = re.sub(r'[^\w\s-]', '', clean_name).strip().replace(" ", "_")
    filename = f"{clean_name}_seed{seed}.png" if clean_name else f"z_image_{int(time.time())}_seed{seed}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    
    return image, filepath, seed


# --- 3D STL Generátor ---
def generate_stl(prompt, steps, guidance, custom_name):
    try:
        pipe = get_shape_pipe()
        print(f"\n[+] 3D Generuji STL: '{prompt}'")
        
        mesh_output = pipe(
            prompt=prompt,
            guidance_scale=float(guidance),
            num_inference_steps=int(steps),
            frame_size=256,
            output_type="mesh"
        ).images[0]
        
        clean_name = custom_name.strip() if custom_name else ""
        clean_name = re.sub(r'[^\w\s-]', '', clean_name).strip().replace(" ", "_")
        if not clean_name:
            clean_name = f"model_3d_{int(time.time())}"
            
        temp_ply = os.path.join(STL_DIR, f"{clean_name}_temp.ply")
        final_stl = os.path.join(STL_DIR, f"{clean_name}.stl")
        
        export_to_ply(mesh_output, temp_ply)
        loaded_mesh = trimesh.load(temp_ply)
        loaded_mesh.export(final_stl)
        
        if os.path.exists(temp_ply):
            os.remove(temp_ply)
            
        return final_stl, final_stl, "Model úspěšně vygenerován!"
    except Exception as e:
        return None, None, f"Chyba: {str(e)}"


# --- Video Generátor (AnimateDiff) ---
def generate_video(prompt, width, height, num_frames, steps, seed, randomize_seed, custom_name):
    try:
        pipe = get_animate_video_pipe()
        if randomize_seed or seed == -1:
            seed = random.randint(0, 2**32 - 1)
            
        generator = torch.Generator("cpu").manual_seed(int(seed))
        print(f"\n[+] AnimateDiff Video Generuji: '{prompt}'")
        
        output = pipe(
            prompt=prompt,
            negative_prompt="bad quality, worst quality, blurry, deformed",
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            num_inference_steps=int(steps),
            guidance_scale=7.5,
            generator=generator,
        )
        
        video_frames = output.frames[0]
        
        clean_name = custom_name.strip() if custom_name else ""
        clean_name = re.sub(r'[^\w\s-]', '', clean_name).strip().replace(" ", "_")
        filename = f"{clean_name}_seed{seed}.mp4" if clean_name else f"video_{int(time.time())}_seed{seed}.mp4"
        filepath = os.path.join(VIDEO_DIR, filename)
        
        export_to_video(video_frames, filepath, fps=8)
        print(f"[✓] Video uloženo do: {filepath}")
        
        return filepath, filepath, seed, "Video úspěšně vygenerováno!"
    except Exception as e:
        print(f"[!] Chyba při generování videa: {e}")
        return None, None, seed, f"Chyba: {str(e)}"


# --- YuE Hudební Generátor ---
def generate_audio(text_prompt, genre_style, custom_name):
    try:
        tokenizer, model = get_yue_model()
        print(f"\n[+] YuE Hudba Generuji: '{text_prompt}' | Žánr: {genre_style}")
        
        formatted_prompt = f"<|genre|>{genre_style}<|lyrics|>{text_prompt}"
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.9,
                top_p=0.93,
                do_sample=True
            )
            
        # Dekódování audio stopy
        sampling_rate = 44100
        audio_array = output_tokens.cpu().numpy().squeeze()
        
        clean_name = custom_name.strip() if custom_name else ""
        clean_name = re.sub(r'[^\w\s-]', '', clean_name).strip().replace(" ", "_")
        filename = f"{clean_name}.wav" if clean_name else f"yue_song_{int(time.time())}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        # Uložení WAV
        wavfile.write(filepath, rate=sampling_rate, data=(audio_array * 32767).astype(np.int16))
        print(f"[✓] Hudba uložena do: {filepath}")
        
        return filepath, filepath, "Hudba úspěšně vygenerována pomocí YuE!"
    except Exception as e:
        print(f"[!] Chyba při generování hudby: {e}")
        return None, None, f"Chyba: {str(e)}"


# --- Gradio UI Rozhraní ---
with gr.Blocks(title="Matěj's AI Studio (2D / 3D / Video / Audio)") as demo:
    gr.Markdown("# 🚀 AI Studio (2D, 3D, Video & YuE Music Gen)")
    
    with gr.Tabs():
        # TAB 1: 2D Obrázky
        with gr.TabItem("🖼️ 2D Obrázky (Z-Image-Turbo)"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_img = gr.Textbox(label="Prompt", lines=3, value="A futuristic cyberpunk city with glowing neon lights, photorealistic, 8k")
                    filename_img = gr.Textbox(label="Název souboru (volitelně)", placeholder="např. muj_obrazek")
                    with gr.Row():
                        width_img = gr.Slider(512, 1536, 64, value=1024, label="Šířka")
                        height_img = gr.Slider(512, 1536, 64, value=1024, label="Výška")
                    with gr.Row():
                        steps_img = gr.Slider(1, 20, 1, value=9, label="Kroky")
                        seed_img = gr.Number(value=42, label="Seed", precision=0)
                    rand_img = gr.Checkbox(label="🎲 Náhodný Seed", value=True)
                    btn_img = gr.Button("🚀 Vygenerovat obrázek", variant="primary")
                with gr.Column(scale=1):
                    out_img = gr.Image(label="Výsledek", type="pil")
                    out_img_path = gr.Textbox(label="Cesta k souboru", interactive=False)
                    out_img_seed = gr.Number(label="Použitý Seed", interactive=False)

            btn_img.click(
                fn=generate_image,
                inputs=[prompt_img, width_img, height_img, steps_img, seed_img, rand_img, filename_img],
                outputs=[out_img, out_img_path, out_img_seed]
            )

        # TAB 2: 3D STL Modely
        with gr.TabItem("🧊 3D STL Modely (Shap-E)"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_stl = gr.Textbox(label="Prompt pro 3D model", lines=3, value="a simple shark figure, 3d model")
                    filename_stl = gr.Textbox(label="Název STL souboru", placeholder="např. zralok")
                    with gr.Row():
                        steps_stl = gr.Slider(15, 100, 5, value=64, label="Kroky")
                        guidance_stl = gr.Slider(1.0, 20.0, 0.5, value=7.5, label="Guidance Scale")
                    btn_stl = gr.Button("🎲 Vygenerovat 3D STL Model", variant="primary")
                with gr.Column(scale=1):
                    out_3d = gr.Model3D(label="3D Náhled")
                    out_stl_path = gr.Textbox(label="Cesta k STL", interactive=False)
                    status_stl = gr.Textbox(label="Stav", interactive=False)

            btn_stl.click(
                fn=generate_stl,
                inputs=[prompt_stl, steps_stl, guidance_stl, filename_stl],
                outputs=[out_3d, out_stl_path, status_stl]
            )

        # TAB 3: Text-to-Video (AnimateDiff)
        with gr.TabItem("🎬 Video (AnimateDiff)"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_vid = gr.Textbox(label="Prompt pro video", lines=3, value="A robotic cat walking in a glowing cyberpunk neon alley, masterpiece, highly detailed")
                    filename_vid = gr.Textbox(label="Název MP4 souboru", placeholder="např. kocka_video")
                    with gr.Row():
                        width_vid = gr.Slider(256, 768, 64, value=512, label="Šířka")
                        height_vid = gr.Slider(256, 768, 64, value=512, label="Výška")
                    with gr.Row():
                        frames_vid = gr.Slider(8, 32, 4, value=16, label="Počet snímků (Frames)")
                        steps_vid = gr.Slider(10, 50, 1, value=25, label="Kroky")
                    with gr.Row():
                        seed_vid = gr.Number(value=42, label="Seed", precision=0)
                        rand_vid = gr.Checkbox(label="🎲 Náhodný Seed", value=True)
                    btn_vid = gr.Button("🎬 Vygenerovat Video", variant="primary")
                with gr.Column(scale=1):
                    out_vid = gr.Video(label="Přehrávač videa")
                    out_vid_path = gr.Textbox(label="Cesta k MP4", interactive=False)
                    out_vid_seed = gr.Number(label="Použitý Seed", interactive=False)
                    status_vid = gr.Textbox(label="Stav", interactive=False)

            btn_vid.click(
                fn=generate_video,
                inputs=[prompt_vid, width_vid, height_vid, frames_vid, steps_vid, seed_vid, rand_vid, filename_vid],
                outputs=[out_vid, out_vid_path, out_vid_seed, status_vid]
            )

        # TAB 4: YuE Music Generátor
        with gr.TabItem("🎵 Hudba (YuE Model)"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_aud = gr.Textbox(
                        label="Text písně / Lyrics", 
                        lines=4, 
                        value="[Verse]\nWalking down the neon street\nFeel the bass under my feet\n[Chorus]\nElectric dreams taking over control!"
                    )
                    genre_aud = gr.Dropdown(
                        label="Žánr a styl hudby",
                        choices=["Cyberpunk Synthwave", "Heavy Metal", "Piano Ballad", "Lo-Fi Hip Hop", "Pop Rock"],
                        value="Cyberpunk Synthwave"
                    )
                    filename_aud = gr.Textbox(label="Název WAV souboru", placeholder="např. pesnicka")
                    btn_aud = gr.Button("🎵 Vygenerovat Hudbu skrze YuE", variant="primary")
                with gr.Column(scale=1):
                    out_aud = gr.Audio(label="Přehrávač hudby")
                    out_aud_path = gr.Textbox(label="Cesta k WAV", interactive=False)
                    status_aud = gr.Textbox(label="Stav", interactive=False)

            btn_aud.click(
                fn=generate_audio,
                inputs=[prompt_aud, genre_aud, filename_aud],
                outputs=[out_aud, out_aud_path, status_aud]
            )

if __name__ == "__main__":
    demo.launch(inbrowser=True)