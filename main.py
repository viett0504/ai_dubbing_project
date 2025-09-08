# main.py — chạy cấu hình cố định, không cần CLI
from transcribe import transcribe_file

# ----- CẤU HÌNH CỐ ĐỊNH -----
AUDIO        = "video_demo.mp4"   # đổi thành file bạn muốn
MODEL        = "large-v3"            # "small" / "medium" / "large-v3"
LANGUAGE     = "vi"               # None = autodetect
DEVICE       = "cpu"
COMPUTE_TYPE = "int8"             # CPU: int8; GPU: float16 / float32
VAD          = True
THREADS      = 0                  # 0 = auto
OUT_PREFIX   = None               # None => cùng tên với AUDIO
BEAM_SIZE    = 5
BEST_OF      = 5
# -----------------------------

def main():
    info = transcribe_file(
        AUDIO,
        model_name=MODEL,
        language=LANGUAGE,
        task="transcribe",
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        vad=VAD,
        threads=THREADS,
        out_prefix=OUT_PREFIX,
        beam_size=BEAM_SIZE,
        best_of=BEST_OF
    )
    print("[DONE]", info)

if __name__ == "__main__":
    main()
