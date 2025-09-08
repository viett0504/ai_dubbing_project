# transcribe.py
from faster_whisper import WhisperModel
from pathlib import Path
import argparse, sys, traceback

def srt_ts(t):
    if t is None: t = 0.0
    h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int(round((t-int(t))*1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def vtt_ts(t):
    if t is None: t = 0.0
    h=int(t//3600); m=int((t%3600)//60); s=int(t%60); ms=int(round((t-int(t))*1000))
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def transcribe_file(
    src_path,
    *,
    model_name="small",
    language=None,                  # "vi" nếu muốn cố định
    task="transcribe",
    device="cpu",
    compute_type="int8",
    vad=True,
    threads=0,
    out_prefix=None,
    beam_size=5,
    best_of=5,
    progress_every=10
):
    """Nhận dạng và xuất .srt/.vtt/.txt. Trả về dict chứa đường dẫn & info."""
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Không thấy file: {src}")

    out_prefix = out_prefix or src.with_suffix("").name
    srt_path = Path(out_prefix + ".srt")
    vtt_path = Path(out_prefix + ".vtt")
    txt_path = Path(out_prefix + ".txt")

    print(f"[INFO] Load model={model_name} device={device} compute={compute_type}", flush=True)
    model = WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=threads)

    print(f"[INFO] Transcribe: {src}", flush=True)
    gen, info = model.transcribe(
        str(src),
        language=language,
        task=task,
        vad_filter=vad,
        beam_size=beam_size,
        best_of=best_of
    )

    segs, n = [], 0
    for s in gen:
        n += 1
        if progress_every and n % progress_every == 0:
            print(f"[INFO] ...đã nhận {n} segments", flush=True)
        segs.append(s)

    # Xuất
    print(f"[INFO] Writing: {srt_path}", flush=True)
    with srt_path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{srt_ts(s.start)} --> {srt_ts(s.end)}\n{s.text.strip()}\n\n")

    print(f"[INFO] Writing: {vtt_path}", flush=True)
    with vtt_path.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for s in segs:
            f.write(f"{vtt_ts(s.start)} --> {vtt_ts(s.end)}\n{s.text.strip()}\n\n")

    print(f"[INFO] Writing: {txt_path}", flush=True)
    with txt_path.open("w", encoding="utf-8") as f:
        for s in segs:
            f.write(s.text.strip()+"\n")

    print(f"[OK] xong. language={getattr(info,'language',None)} p={getattr(info,'language_probability',None)}", flush=True)

    return {
        "srt": str(srt_path),
        "vtt": str(vtt_path),
        "txt": str(txt_path),
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "segments_count": len(segs),
    }

# ---- CLI giữ nguyên để vẫn có thể chạy trực tiếp ----
def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("src", help="đường dẫn audio/video (mp3, wav, mp4, m4a, ...)")
    p.add_argument("--model", default="small")
    p.add_argument("--language", default=None)
    p.add_argument("--task", default="transcribe", choices=["transcribe","translate"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--compute-type", dest="compute_type", default="int8")
    p.add_argument("--vad", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--out-prefix", dest="out_prefix", default=None)
    args = p.parse_args()

    try:
        transcribe_file(
            args.src,
            model_name=args.model,
            language=args.language,
            task=args.task,
            device=args.device,
            compute_type=args.compute_type,
            vad=args.vad,
            threads=args.threads,
            out_prefix=args.out_prefix
        )
    except Exception:
        print("[ERR] Exception xảy ra:\n" + traceback.format_exc(), file=sys.stderr, flush=True)
        sys.exit(2)

if __name__ == "__main__":
    _cli()
