"""
語音觸發 LeRobot 推論
說「開始」→ 執行推論
說「結束」→ 停止推論
"""

import sounddevice as sd
import numpy as np
import subprocess
import threading
import signal
import os
import sys
from faster_whisper import WhisperModel

# ══════════════════════════════════════════════
#  設定區（依需求修改這裡）
# ══════════════════════════════════════════════

WHISPER_MODEL  = "small"
WHISPER_DEVICE = "cuda"
WHISPER_DTYPE  = "float16"

LEROBOT_ENV    = "lerobot"

START_KEYWORDS = ["開始", "开始"]
STOP_KEYWORDS  = ["結束", "结束"]

RECORD_SECONDS = 2
SAMPLE_RATE    = 16000
MIC_DEVICE     = None

TEST_MODE      = False   # True=只印訊息不動手臂，False=正式執行

# ══════════════════════════════════════════════

LEROBOT_ARGS = [
    "lerobot-rollout",
    "--robot.type=so101_follower",
    "--robot.port=/dev/ttyACM0",
    "--robot.cameras={ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: YUYV} }",
    "--robot.id=my_awesome_follower_arm",
    "--display_data=true",
    "--dataset.repo_id=seeed/rollout_eval_test123",
    "--dataset.single_task=Put lego brick into the transparent box",
    "--policy.path=/home/pc-04/lerobot/outputs/train/act_so105_test/checkpoints/last/pretrained_model",
    "--strategy.type=sentry",
    "--duration=60",
    "--dataset.push_to_hub=false",
]

# ── 幻覺過濾 ──
HALLUCINATION_FILTER = [
    "謝謝", "谢谢", "谢谢你", "謝謝你",
    "大人", "字幕", "請訂閱", "訂閱",
    "bye", "掰掰", "好的", "好",
    "噢", "喔", "嗯", "啊",
]

# ══════════════════════════════════════════════

print("載入 Whisper 模型中，請稍候...")
model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_DTYPE)
print("✅ 模型載入完成！")

current_process = None
process_lock    = threading.Lock()

def record_audio() -> np.ndarray:
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MIC_DEVICE,
    )
    sd.wait()
    return audio.flatten()

def is_silence(audio: np.ndarray, threshold=0.01) -> bool:
    return np.abs(audio).mean() < threshold

def transcribe(audio: np.ndarray) -> str:
    segments, _ = model.transcribe(
        audio,
        language="zh",
        beam_size=1,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )
    text = "".join(s.text for s in segments).strip()
    if text in HALLUCINATION_FILTER:
        return ""
    return text

def start_lerobot():
    global current_process

    if TEST_MODE:
        print("\n" + "─"*50)
        print("🧪 [測試模式] 偵測到「開始」！")
        print("🧪 [測試模式] 準備執行的指令：")
        print("   " + " ".join(LEROBOT_ARGS))
        print("🧪 [測試模式] 手臂不會動，測試完成！")
        print("─"*50 + "\n")
        return

    import shutil
    conda_bin = shutil.which("conda") or os.path.expanduser("~/miniforge3/bin/conda")
    cmd = [conda_bin, "run", "-n", LEROBOT_ENV, "--no-capture-output"] + LEROBOT_ARGS

    print(f"\n▶ 啟動推論...\n{'─'*50}")
    with process_lock:
        # ✅ os.setsid() 讓整個程序群組都可以被一起殺掉
        current_process = subprocess.Popen(cmd, preexec_fn=os.setsid)
    current_process.wait()
    print(f"\n{'─'*50}\n✅ 推論結束，繼續監聽...")
    with process_lock:
        current_process = None

def stop_lerobot():
    global current_process
    with process_lock:
        if current_process and current_process.poll() is None:
            try:
                # ✅ 殺掉整個程序群組（conda run + lerobot-rollout + 所有子程序）
                pgid = os.getpgid(current_process.pid)
                os.killpg(pgid, signal.SIGKILL)
                print("\n⛔ 已終止整個推論程序群組，手臂停止...")
            except ProcessLookupError:
                print("\n（程序已自行結束）")
        else:
            print("\n（目前沒有推論在執行）")

def is_running() -> bool:
    if TEST_MODE:
        return False
    with process_lock:
        return current_process is not None and current_process.poll() is None

# ── 主迴圈 ──────────────────────────────────
mode_label = "🧪 測試模式" if TEST_MODE else "🤖 正式模式"
print(f"\n{mode_label} | 說「{'或'.join(START_KEYWORDS)}」觸發，說「{'或'.join(STOP_KEYWORDS)}」停止\n")

try:
    while True:
        audio = record_audio()

        if is_silence(audio):
            continue

        text = transcribe(audio)

        if text:
            print(f"辨識：{text}")

        if any(kw in text for kw in START_KEYWORDS):
            if is_running():
                print("⚠️  推論已在執行中，忽略此次觸發")
            else:
                t = threading.Thread(target=start_lerobot, daemon=True)
                t.start()

        elif any(kw in text for kw in STOP_KEYWORDS):
            stop_lerobot()

except KeyboardInterrupt:
    print("\n\n使用者中斷（Ctrl+C），停止所有程序...")
    stop_lerobot()
    sys.exit(0)
