# LeRobot Voice Trigger (語音觸發機械手臂推論)

## ⚙️ 架構總覽

本系統採用雙 `conda` 虛擬環境隔離設計，確保語音監聽與機械手臂硬體控制的依賴套件互不干擾：


```

麥克風輸入
│
▼
[stt_env] 語音監聽主體 (faster-whisper / openWakeWord)
│
├─► 偵測到「開始」 ──► subprocess (conda run -n lerobot) ──► 執行手臂推論 (lerobot-rollout)
└─► 偵測到「結束」 ──► os.killpg(pgid, SIGKILL) ────────► 強制中止程序群組

```

| Conda 環境 | 用途 | 主要套件 |
| :--- | :--- | :--- |
| `stt_env` | 語音辨識、即時監聽、跨環境觸發腳本 | `faster-whisper`、`sounddevice`、`openwakeword`、`torch (CUDA)` |
| `lerobot` | 機械手臂控制、資料採集與訓練、模型推論 | `lerobot`、`pynput`、`opencv-python` |

---

## 🛠️ 前置準備與硬體配置

在執行語音腳本前，請確保你的 Ubuntu 系統與 SO101 機械手臂已完成以下基本配置：

### 1. 硬體連接與連接埠賦權
將 Follower（從動臂）與 Leader（領導臂）插入電腦（注意：先插上的 USB 通常為 `ttyACM0`），並賦予 Linux 序列埠存取權限：
```bash
sudo chmod 666 /dev/ttyACM*

```

### 2. 音訊環境依賴安裝 (Ubuntu)

不論系統運作於 PulseAudio 或 PipeWire，請先在系統層級補齊音訊驅動與 FFmpeg 工具：

```bash
sudo apt update && sudo apt install -y \
    portaudio19-dev python3-dev ffmpeg pulseaudio alsa-utils libasound2-dev

```

*驗證麥克風是否正常抓到：*

```bash
arecord -l
# 測試錄音 3 秒並播放確認
arecord -d 3 -f cd test.wav && aplay test.wav

```

### 3. 確認 GPU 運算環境

確認 NVIDIA 驅動與 CUDA Toolkit 已正確安裝：

```bash
nvidia-smi
nvcc --version

```

---

## 🚀 語音環境建置 (stt_env)

請依照下列步驟建立專屬的語音主控環境：

```bash
# 1. 建立並進入虛擬環境
conda create -n stt_env python=3.10 -y
conda activate stt_env

# 2. 安裝對應 CUDA 版本的 PyTorch (以 CUDA 12.1 為例)
pip install torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 3. 安裝語音核心套件
pip install faster-whisper sounddevice numpy openwakeword

```

---

## 💻 使用說明

### Faster-Whisper

此模式每 2 秒錄製一段音訊並送入 Whisper 模型辨識。適合不想預先錄製聲音樣本、快速測試的場景。

1. **設定腳本：** 檢查 `voice_trigger.py` 中的 `LEROBOT_ARGS` 是否與你的 ACT Checkpoint 模型路徑相符。
2. **執行監聽：**
```bash
conda activate stt_env
python voice_trigger.py

```


3. **語音指令：**
* 對麥克風說 **「開始」** ➔ 背景將自動啟動 `lerobot-rollout` 開始執行積木任務。
* 對麥克風說 **「結束」** ➔ 立即發送全域 `SIGKILL` 訊號安全中止手臂動作。
* 在終端機隨時按下 `Ctrl+C` ➔ 安全釋放所有硬體與環境程序並退出。





* **中文「開始」常被誤認成別的錯字？**
1. 可將 `START_KEYWORDS` 修改加入英文關鍵詞 `"start"`，效果更為穩定。
2. 將模型等級調升至 `WHISPER_MODEL = "medium"`（需消耗較多 VRAM）。
3. 推薦直接切換至 **模式二 (openWakeWord)** 進行聲音微調，可徹底解決誤判問題。


