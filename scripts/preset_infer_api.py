import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import shutil
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from pydantic import BaseModel

from webui.preset import Presets
from webui.utils import load_configs, get_vr_model, get_msst_model
from webui.setup import setup_webui, set_debug
from utils.constant import *
from utils.logger import get_logger

logger = get_logger()
app = FastAPI(title="MSST Preset Inference API", version="1.0.0")

class InferenceResponse(BaseModel):
    status: str
    message: str
    output_path: Optional[str] = None
    time_cost: Optional[float] = None

class AudioParams(BaseModel):
    wav_bit_depth: str = "PCM_16"
    flac_bit_depth: str = "PCM_16"
    mp3_bit_rate: str = "192k"

class LocalPresetRequest(BaseModel):
    preset_path: str
    output_format: str = "wav"
    extra_output_dir: bool = False
    audio_params: Optional[AudioParams] = None

@app.post("/infer", response_model=InferenceResponse)
async def infer_preset(
    preset_file: UploadFile = File(...),
    input_file: UploadFile = File(...),
    output_format: str = Form("wav"),
    extra_output_dir: bool = Form(False),
    wav_bit_depth: str = Form("PCM_16"),
    flac_bit_depth: str = Form("PCM_16"),
    mp3_bit_rate: str = Form("192k")
):
    try:
        # 创建临时目录存储上传的文件
        os.makedirs("temp_uploads", exist_ok=True)
        
        # 保存预设文件
        preset_path = os.path.join("temp_uploads", preset_file.filename)
        with open(preset_path, "wb") as f:
            content = await preset_file.read()
            f.write(content)
        
        # 创建输入目录并保存输入文件
        input_folder = os.path.join("temp_uploads", "input")
        os.makedirs(input_folder, exist_ok=True)
        input_file_path = os.path.join(input_folder, input_file.filename)
        with open(input_file_path, "wb") as f:
            content = await input_file.read()
            f.write(content)
        
        # 创建输出目录
        store_dir = os.path.join("temp_uploads", "output")
        os.makedirs(store_dir, exist_ok=True)

        # 加载预设配置
        preset_data = load_configs(preset_path)
        preset_version = preset_data.get("version", "Unknown version")
        if preset_version not in SUPPORTED_PRESET_VERSION:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Unsupported preset version: {preset_version}"}
            )

        # 设置输出目录
        direct_output = store_dir
        if extra_output_dir:
            os.makedirs(os.path.join(store_dir, "extra_output"), exist_ok=True)
            direct_output = os.path.join(store_dir, "extra_output")

        # 初始化预设
        preset = Presets(preset_data, force_cpu=False, use_tta=False, logger=logger)
        
        # 设置音频参数
        preset.wav_bit_depth = wav_bit_depth
        preset.flac_bit_depth = flac_bit_depth
        preset.mp3_bit_rate = mp3_bit_rate
        
        if not preset.is_exist_models()[0]:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Model {preset.is_exist_models()[1]} not found"}
            )

        # 开始处理
        start_time = time.time()
        current_step = 0
        input_to_use = input_folder
        tmp_store_dir = None
        
        for step in range(preset.total_steps):
            if current_step == 0:
                input_to_use = input_folder
                tmp_store_dir = os.path.join(TEMP_PATH, "step_1_output")
            elif preset.total_steps - 1 > current_step > 0:
                if input_to_use != input_folder:
                    shutil.rmtree(input_to_use)
                input_to_use = os.path.join(TEMP_PATH, f"step_{current_step}_output")
                tmp_store_dir = os.path.join(TEMP_PATH, f"step_{current_step + 1}_output")
            elif current_step == preset.total_steps - 1:
                input_to_use = os.path.join(TEMP_PATH, f"step_{current_step}_output")
                tmp_store_dir = store_dir
            elif preset.total_steps == 1:
                input_to_use = input_folder
                tmp_store_dir = store_dir

            data = preset.get_step(step)
            model_type = data["model_type"]
            model_name = data["model_name"]
            input_to_next = data["input_to_next"]
            output_to_storage = data["output_to_storage"]

            if model_type == "UVR_VR_Models":
                primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
                storage = {primary_stem: [], secondary_stem: []}
                storage[input_to_next].append(tmp_store_dir)
                for stem in output_to_storage:
                    storage[stem].append(direct_output)

                result = preset.vr_infer(model_name, input_to_use, storage, output_format)
                if result[0] == 0:
                    return JSONResponse(
                        status_code=500,
                        content={"status": "error", "message": f"Failed to run VR model {model_name}: {result[1]}"}
                    )
            else:
                model_path, config_path, msst_model_type, _ = get_msst_model(model_name)
                stems = load_configs(config_path).training.get("instruments", [])
                storage = {stem: [] for stem in stems}
                storage[input_to_next].append(tmp_store_dir)
                for stem in output_to_storage:
                    storage[stem].append(direct_output)

                result = preset.msst_infer(msst_model_type, config_path, model_path, input_to_use, storage, output_format)
                if result[0] == 0:
                    return JSONResponse(
                        status_code=500,
                        content={"status": "error", "message": f"Failed to run MSST model {model_name}: {result[1]}"}
                    )
            current_step += 1

        time_cost = round(time.time() - start_time, 2)
        
        return InferenceResponse(
            status="success",
            message="Inference completed successfully",
            output_path=store_dir,
            time_cost=time_cost
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
    finally:
        # 清理临时文件
        if os.path.exists(TEMP_PATH):
            shutil.rmtree(TEMP_PATH)

@app.post("/infer/local", response_model=InferenceResponse)
async def infer_preset_local(
    preset_path: str = Form(...),
    output_format: str = Form("wav"),
    extra_output_dir: bool = Form(False),
    wav_bit_depth: str = Form("PCM_16"),
    flac_bit_depth: str = Form("PCM_16"),
    mp3_bit_rate: str = Form("192k"),
    input_file: UploadFile = File(...)
):
    try:
        # 检查预设文件是否存在
        if not os.path.exists(preset_path):
            raise HTTPException(status_code=404, detail=f"Preset file not found: {preset_path}")
        
        # 创建临时目录存储上传的文件
        os.makedirs("temp_uploads", exist_ok=True)
        
        # 创建输入目录并保存输入文件
        input_folder = os.path.join("temp_uploads", "input")
        os.makedirs(input_folder, exist_ok=True)
        input_file_path = os.path.join(input_folder, input_file.filename)
        with open(input_file_path, "wb") as f:
            content = await input_file.read()
            f.write(content)
        
        # 创建输出目录
        store_dir = os.path.join("temp_uploads", "output")
        os.makedirs(store_dir, exist_ok=True)

        # 加载预设配置
        preset_data = load_configs(preset_path)
        preset_version = preset_data.get("version", "Unknown version")
        if preset_version not in SUPPORTED_PRESET_VERSION:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Unsupported preset version: {preset_version}"}
            )

        # 设置输出目录
        direct_output = store_dir
        if extra_output_dir:
            os.makedirs(os.path.join(store_dir, "extra_output"), exist_ok=True)
            direct_output = os.path.join(store_dir, "extra_output")

        # 初始化预设
        preset = Presets(preset_data, force_cpu=False, use_tta=False, logger=logger)
        
        # 设置音频参数
        preset.wav_bit_depth = wav_bit_depth
        preset.flac_bit_depth = flac_bit_depth
        preset.mp3_bit_rate = mp3_bit_rate
        
        if not preset.is_exist_models()[0]:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Model {preset.is_exist_models()[1]} not found"}
            )

        # 开始处理
        start_time = time.time()
        current_step = 0
        input_to_use = input_folder
        tmp_store_dir = None
        
        for step in range(preset.total_steps):
            if current_step == 0:
                input_to_use = input_folder
                tmp_store_dir = os.path.join(TEMP_PATH, "step_1_output")
            elif preset.total_steps - 1 > current_step > 0:
                if input_to_use != input_folder:
                    shutil.rmtree(input_to_use)
                input_to_use = os.path.join(TEMP_PATH, f"step_{current_step}_output")
                tmp_store_dir = os.path.join(TEMP_PATH, f"step_{current_step + 1}_output")
            elif current_step == preset.total_steps - 1:
                input_to_use = os.path.join(TEMP_PATH, f"step_{current_step}_output")
                tmp_store_dir = store_dir
            elif preset.total_steps == 1:
                input_to_use = input_folder
                tmp_store_dir = store_dir

            data = preset.get_step(step)
            model_type = data["model_type"]
            model_name = data["model_name"]
            input_to_next = data["input_to_next"]
            output_to_storage = data["output_to_storage"]

            if model_type == "UVR_VR_Models":
                primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
                storage = {primary_stem: [], secondary_stem: []}
                storage[input_to_next].append(tmp_store_dir)
                for stem in output_to_storage:
                    storage[stem].append(direct_output)

                result = preset.vr_infer(model_name, input_to_use, storage, output_format)
                if result[0] == 0:
                    return JSONResponse(
                        status_code=500,
                        content={"status": "error", "message": f"Failed to run VR model {model_name}: {result[1]}"}
                    )
            else:
                model_path, config_path, msst_model_type, _ = get_msst_model(model_name)
                stems = load_configs(config_path).training.get("instruments", [])
                storage = {stem: [] for stem in stems}
                storage[input_to_next].append(tmp_store_dir)
                for stem in output_to_storage:
                    storage[stem].append(direct_output)

                result = preset.msst_infer(msst_model_type, config_path, model_path, input_to_use, storage, output_format)
                if result[0] == 0:
                    return JSONResponse(
                        status_code=500,
                        content={"status": "error", "message": f"Failed to run MSST model {model_name}: {result[1]}"}
                    )
            current_step += 1

        time_cost = round(time.time() - start_time, 2)
        
        return InferenceResponse(
            status="success",
            message="Inference completed successfully",
            output_path=store_dir,
            time_cost=time_cost
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
    finally:
        # 清理临时文件
        if os.path.exists(TEMP_PATH):
            shutil.rmtree(TEMP_PATH)

@app.post("/infer/upload", response_model=InferenceResponse)
async def infer_preset_upload(
    preset_file: UploadFile = File(...),
    input_file: UploadFile = File(...),
    output_format: str = Form("wav"),
    extra_output_dir: bool = Form(False),
    wav_bit_depth: str = Form("PCM_16"),
    flac_bit_depth: str = Form("PCM_16"),
    mp3_bit_rate: str = Form("192k")
):
    try:
        # 创建临时目录存储上传的文件
        os.makedirs("temp_uploads", exist_ok=True)
        
        # 保存预设文件
        preset_path = os.path.join("temp_uploads", preset_file.filename)
        with open(preset_path, "wb") as f:
            content = await preset_file.read()
            f.write(content)
        
        # 创建输入目录并保存输入文件
        input_folder = os.path.join("temp_uploads", "input")
        os.makedirs(input_folder, exist_ok=True)
        input_file_path = os.path.join(input_folder, input_file.filename)
        with open(input_file_path, "wb") as f:
            content = await input_file.read()
            f.write(content)
        
        # 创建输出目录
        store_dir = os.path.join("temp_uploads", "output")
        os.makedirs(store_dir, exist_ok=True)

        # 加载预设配置
        preset_data = load_configs(preset_path)
        preset_version = preset_data.get("version", "Unknown version")
        if preset_version not in SUPPORTED_PRESET_VERSION:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Unsupported preset version: {preset_version}"}
            )

        # 设置输出目录
        direct_output = store_dir
        if extra_output_dir:
            os.makedirs(os.path.join(store_dir, "extra_output"), exist_ok=True)
            direct_output = os.path.join(store_dir, "extra_output")

        # 初始化预设
        preset = Presets(preset_data, force_cpu=False, use_tta=False, logger=logger)
        
        # 设置音频参数
        preset.wav_bit_depth = wav_bit_depth
        preset.flac_bit_depth = flac_bit_depth
        preset.mp3_bit_rate = mp3_bit_rate
        
        if not preset.is_exist_models()[0]:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Model {preset.is_exist_models()[1]} not found"}
            )

        # 开始处理
        start_time = time.time()
        current_step = 0
        input_to_use = input_folder
        tmp_store_dir = None
        
        for step in range(preset.total_steps):
            if current_step == 0:
                input_to_use = input_folder
                tmp_store_dir = os.path.join(TEMP_PATH, "step_1_output")
            elif preset.total_steps - 1 > current_step > 0:
                if input_to_use != input_folder:
                    shutil.rmtree(input_to_use)
                input_to_use = os.path.join(TEMP_PATH, f"step_{current_step}_output")
                tmp_store_dir = os.path.join(TEMP_PATH, f"step_{current_step + 1}_output")
            elif current_step == preset.total_steps - 1:
                input_to_use = os.path.join(TEMP_PATH, f"step_{current_step}_output")
                tmp_store_dir = store_dir
            elif preset.total_steps == 1:
                input_to_use = input_folder
                tmp_store_dir = store_dir

            data = preset.get_step(step)
            model_type = data["model_type"]
            model_name = data["model_name"]
            input_to_next = data["input_to_next"]
            output_to_storage = data["output_to_storage"]

            if model_type == "UVR_VR_Models":
                primary_stem, secondary_stem, _, _ = get_vr_model(model_name)
                storage = {primary_stem: [], secondary_stem: []}
                storage[input_to_next].append(tmp_store_dir)
                for stem in output_to_storage:
                    storage[stem].append(direct_output)

                result = preset.vr_infer(model_name, input_to_use, storage, output_format)
                if result[0] == 0:
                    return JSONResponse(
                        status_code=500,
                        content={"status": "error", "message": f"Failed to run VR model {model_name}: {result[1]}"}
                    )
            else:
                model_path, config_path, msst_model_type, _ = get_msst_model(model_name)
                stems = load_configs(config_path).training.get("instruments", [])
                storage = {stem: [] for stem in stems}
                storage[input_to_next].append(tmp_store_dir)
                for stem in output_to_storage:
                    storage[stem].append(direct_output)

                result = preset.msst_infer(msst_model_type, config_path, model_path, input_to_use, storage, output_format)
                if result[0] == 0:
                    return JSONResponse(
                        status_code=500,
                        content={"status": "error", "message": f"Failed to run MSST model {model_name}: {result[1]}"}
                    )
            current_step += 1

        time_cost = round(time.time() - start_time, 2)
        
        # 清理输入文件
        if os.path.exists(input_folder):
            shutil.rmtree(input_folder)
        
        # 清理预设文件
        if os.path.exists(preset_path):
            os.remove(preset_path)
        
        return InferenceResponse(
            status="success",
            message="Inference completed successfully",
            output_path=store_dir,
            time_cost=time_cost
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
    finally:
        # 清理临时文件
        if os.path.exists(TEMP_PATH):
            shutil.rmtree(TEMP_PATH)

@app.get("/download/{filename}")
async def download_file(filename: str):
    """下载推理结果文件"""
    # 检查所有可能的输出目录
    possible_paths = [
        os.path.join("temp_uploads", "output", filename),
        os.path.join("temp_uploads", "output", "extra_output", filename)
    ]
    
    for file_path in possible_paths:
        if os.path.exists(file_path):
            return FileResponse(file_path, filename=filename)
    
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/list_outputs")
async def list_output_files():
    """列出所有可用的输出文件"""
    output_dir = os.path.join("temp_uploads", "output")
    extra_output_dir = os.path.join(output_dir, "extra_output")
    
    files = []
    
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith((".wav", ".mp3")):
                files.append({
                    "name": file,
                    "path": os.path.join("output", file)
                })
    
    if os.path.exists(extra_output_dir):
        for file in os.listdir(extra_output_dir):
            if file.endswith((".wav", ".mp3")):
                files.append({
                    "name": file,
                    "path": os.path.join("output", "extra_output", file)
                })
    
    return {"files": files}

@app.get("/presets")
async def list_presets():
    """列出所有可用的预设文件"""
    try:
        # 假设预设文件存储在presets目录下
        presets_dir = "presets"
        if not os.path.exists(presets_dir):
            return JSONResponse(
                status_code=404,
                content={"status": "error", "message": "Presets directory not found"}
            )
        
        preset_files = []
        for file in os.listdir(presets_dir):
            if file.endswith(".json"):
                preset_files.append(file)
        
        return {"status": "success", "presets": preset_files}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.on_event("startup")
async def startup_event():
    # 初始化必要的配置
    if not os.path.exists("configs"):
        shutil.copytree("configs_backup", "configs")
    if not os.path.exists("data"):
        shutil.copytree("data_backup", "data")
    setup_webui()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000) 
