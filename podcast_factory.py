#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏭 PODCAST FACTORY — "أسرار و خفايا" automated podcast maker
================================================================
Inputs:
  script.txt            -> نص البودكاست بالدارجة (تلصقو هنا قبل التشغيل)
  voice_ref.wav         -> مرجع الصوت (مقطع من البودكاست الأول) — الصوت غايتستنسخ منو
  assets/live_final_bg.png + assets/fonts/*.ttf   -> الهوية البصرية (نفس البودكاست الأول)

ENV (optional):
  BG_URL        رابط تصويرة خلفية جديدة (اختياري — إلا خاوي كيستعمل الخلفية الأصلية)
  CHANNEL       اسم القناة (default: أسرار و خفايا)
  XTTS_SPEED    سرعة التكلم (default: 1.0)
  ONLY          tts | analyze | render | all  (default: all)
  MAX_CHARS     حد أقصى للنص (تجربة قصيرة)  default: 0 = بلا حد

Output (فمجلد out/):
  podcast_final.mp4   الفيديو الكامل 1080x1080 (faststart)
  master.mp3          الصوت النهائي
  subs.srt            الترجمة
  preview_*.jpg       تصاور للفحص
================================================================
"""
import os, re, re as _re, sys, math, time, json, difflib, pickle, subprocess, bisect
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'out')
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault('COQUI_TOS_AGREED', '1')

import imageio_ffmpeg
FF = imageio_ffmpeg.get_ffmpeg_exe()

FPS, W, H = 24, 1080, 1080
SR = 22050
OUTRO_S = 10
CHANNEL = os.environ.get('CHANNEL', 'أسرار و خفايا')
SPEED = float(os.environ.get('XTTS_SPEED', '1.0'))
ONLY = os.environ.get('ONLY', 'all')
MAX_CHARS = int(os.environ.get('MAX_CHARS', '0'))
BG_URL = os.environ.get('BG_URL', '').strip()
TTS_DIR = os.path.join(BASE, 'chunks_tts')
os.makedirs(TTS_DIR, exist_ok=True)
CACHE = os.path.join(BASE, 'factory_cache.pkl')

def log(*a):
    print('[%7.1fs]' % (time.time() - T0), *a, flush=True)
T0 = time.time()

# =====================================================================
# STAGE 1 — SCRIPT -> CHUNKS -> XTTS (voice clone of the first podcast)
# =====================================================================
def read_script():
    p = os.path.join(BASE, 'script.txt')
    t = open(p, encoding='utf-8').read()
    t = _re.sub(r'\s+', ' ', t).strip()
    if MAX_CHARS: t = t[:MAX_CHARS]
    if len(t) < 20:
        sys.exit('!! script.txt khawi — lsa9 nass dial podcast dakhel script.txt')
    log('script: %d chars (~%.1f min @16 ch/s)' % (len(t), len(t) / 16.0 / 60))
    return t

def tts_chunks(text, maxc=230):
    """قسّم النص لجمل صغيرة (XTTS 2 كيتقب جمل قصيرة مزيان)"""
    sents = _re.split(r'(?<=[.!؟…?])\s+', text)
    pieces = []
    for s in sents:
        s = s.strip()
        if not s: continue
        if len(s) <= maxc:
            pieces.append(s); continue
        for part in _re.split(r'(?<=[،,؛:])\s*', s):
            part = part.strip()
            if not part: continue
            while len(part) > maxc:
                ws = part.split(); acc = ''
                while ws and len((acc + ' ' + ws[0]).strip()) <= maxc:
                    acc = (acc + ' ' + ws.pop(0)).strip()
                pieces.append(acc); part = ' '.join(ws)
            if part: pieces.append(part)
    log('tts chunks: %d (avg %.0f chars)' % (len(pieces), sum(map(len, pieces)) / max(1, len(pieces))))
    return pieces

def stage_tts():
    import soundfile as sf
    from TTS.api import TTS
    text = read_script()
    chunks = tts_chunks(text)
    json.dump(chunks, open(os.path.join(TTS_DIR, 'chunks.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('load XTTS v2 (first run downloads ~2GB)...')
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    tts = TTS(model_name='tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)
    ref = os.path.join(BASE, 'voice_ref.wav')
    assert os.path.exists(ref), 'voice_ref.wav مفقود!'
    t_tts = time.time()
    for i, c in enumerate(chunks):
        fp = os.path.join(TTS_DIR, 'chunk_%04d.wav' % i)
        if os.path.exists(fp) and os.path.getsize(fp) > 2000:
            continue
        for attempt in (1, 2, 3):
            try:
                tts.tts_to_file(text=c, speaker_wav=ref, language='ar',
                                file_path=fp, speed=SPEED, split_sentences=False)
                break
            except Exception as e:
                log('chunk %d attempt %d failed: %s' % (i, attempt, str(e)[:120]))
                if attempt == 3: raise
                time.sleep(5)
        if i % 10 == 0 or i == len(chunks) - 1:
            el = time.time() - t_tts
            eta = el / max(1, i + 1) * (len(chunks) - i - 1)
            log('TTS %d/%d | elapsed %.0fs | ETA ~%.0fs' % (i + 1, len(chunks), el, eta))
    # assemble master
    sr_tts, pieces_audio = None, []
    gap = None
    for i in range(len(chunks)):
        y, sr = sf.read(os.path.join(TTS_DIR, 'chunk_%04d.wav' % i))
        if y.ndim > 1: y = y.mean(axis=1)
        y = y.astype(np.float32)
        if sr_tts is None:
            sr_tts = sr; gap = np.zeros(int(sr * 0.55), np.float32)
        assert sr == sr_tts
        pieces_audio += [y, gap]
    head = np.zeros(int(sr_tts * 0.8), np.float32)
    au = np.concatenate([head] + pieces_audio[:-1])
    peak = np.max(np.abs(au)) + 1e-9
    au = au / peak * 0.95
    raw_wav = os.path.join(OUT, 'master_raw.wav')
    sf.write(raw_wav, au, sr_tts)
    log('raw master: %.1fs @%dHz' % (len(au) / sr_tts, sr_tts))
    # loudnorm -> master.mp3 (exactly like our podcasts)
    mp3 = os.path.join(OUT, 'master.mp3')
    r = subprocess.run([FF, '-y', '-v', 'error', '-i', raw_wav,
                        '-af', 'loudnorm=I=-14:TP=-1.5:LRA=9',
                        '-ar', '44100', '-ac', '2', '-b:a', '128k', mp3])
    assert r.returncode == 0, 'loudnorm failed'
    log('master.mp3 ok -> %s' % mp3)
    return mp3

# =====================================================================
# STAGE 2 — WHISPER + ALIGNMENT + SPECTRUM REF (same math as podcast #1)
# =====================================================================
MAXC = 36
DIAC = re.compile(r'[\u064B-\u0652\u0670\u0640]')
NONL = re.compile(r'[^\u0621-\u064Aa-z0-9 ]')
def norm(s):
    s = DIAC.sub('', s)
    for a in ('\u0623', '\u0625', '\u0622', '\u0671'):
        s = s.replace(a, '\u0627')
    s = s.replace('\u0649', '\u064a').replace('\u0629', '\u0647')
    for d, w in zip('\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669', '0123456789'):
        s = s.replace(d, w)
    s = NONL.sub(' ', s.lower())
    out = []
    for t in s.split():
        if len(t) >= 5 and t[0] in '\u0648\u0641':
            out += [t[0], t[1:]]
        elif len(t) >= 6 and t[0] in '\u0628\u0644\u0643':
            out += [t[0], t[1:]]
        else:
            out.append(t)
    return out

def to_units(text):
    units = []
    for s in [s.strip() for s in _re.split(r'(?<=[.\u061f!\u2026?])\s+', text) if s.strip()]:
        if len(s) <= MAXC:
            units.append(s); continue
        for p in _re.split(r'(?<=[\u060c,؛:])\s*', s):
            p = p.strip()
            if not p: continue
            while len(p) > MAXC:
                ws = p.split(); part = ''
                while ws and len((part + ' ' + ws[0]).strip()) <= MAXC:
                    part = (part + ' ' + ws.pop(0)).strip()
                units.append(part); p = ' '.join(ws)
            if p: units.append(p)
    merged = []
    for u in units:
        if merged and len(u) < 10 and len(merged[-1] + ' ' + u) <= 40:
            merged[-1] = (merged[-1] + ' ' + u).strip()
        else:
            merged.append(u)
    return merged

def decode(mp3, ss, t):
    cmd = [FF, '-v', 'error']
    if ss: cmd += ['-ss', '%.6f' % ss]
    cmd += ['-i', mp3, '-t', str(t), '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-']
    return np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.float32)

def stage_analyze(mp3):
    from faster_whisper import WhisperModel
    text = read_script()
    units = to_units(text)
    n = len(units)
    log('whisper transcribe (base, cpu)...')
    wm = WhisperModel('base', device='cpu', compute_type='int8')
    segs, _ = wm.transcribe(mp3, language='ar', word_timestamps=True,
                            vad_filter=True, beam_size=5)
    words = []
    for sg in segs:
        for w in (sg.words or []):
            words.append({'text': w.word.strip(), 'start': float(w.start), 'end': float(w.end)})
            if not words[-1]['text']: words.pop()
    log('whisper words: %d' % len(words))
    audio = decode(mp3, 0.0, 40000)
    TOTAL = len(audio) / SR
    TOTAL_FRAMES = int(round(TOTAL * FPS))
    log('audio: %.2fs | frames %d' % (TOTAL, TOTAL_FRAMES))

    wtx = lambda w: w['text']; ws_ = np.array([w['start'] for w in words]); we_ = np.array([w['end'] for w in words])
    wtok, twrd = [], []
    for wi, w in enumerate(words):
        for p in norm(wtx(w)):
            wtok.append(p); twrd.append(wi)
    stok, suni = [], []
    for ui, u in enumerate(units):
        for p in norm(u):
            stok.append(p); suni.append(ui)
    log('units: %d | script toks: %d | whisper toks: %d' % (n, len(stok), len(wtok)))

    def sim(a, b):
        if a == b: return 1.0
        if min(len(a), len(b)) < 3: return 0.0
        return difflib.SequenceMatcher(None, a, b).ratio()

    sm_ = difflib.SequenceMatcher(None, stok, wtok, autojunk=False)
    pairs = {}
    for tag, a0, a1, b0, b1 in sm_.get_opcodes():
        if tag == 'equal':
            for i in range(a1 - a0): pairs[a0 + i] = (b0 + i, 1.0)
        elif tag == 'replace':
            la, lb = a1 - a0, b1 - b0
            if 0 < la <= 250 and 0 < lb <= 250:
                SA, SB = stok[a0:a1], wtok[b0:b1]
                S = np.zeros((la, lb))
                for ii in range(la):
                    for jj in range(lb):
                        S[ii, jj] = sim(SA[ii], SB[jj])
                GAP = 0.9
                D = np.zeros((la + 1, lb + 1)); ptr = np.zeros((la + 1, lb + 1), np.int8)
                D[:, 0] = np.arange(la + 1) * GAP; D[0, :] = np.arange(lb + 1) * GAP
                ptr[:, 0] = 2; ptr[0, :] = 3
                for ii in range(1, la + 1):
                    for jj in range(1, lb + 1):
                        c = (D[ii-1, jj-1] + (1 - S[ii-1, jj-1]), D[ii-1, jj] + GAP, D[ii, jj-1] + GAP)
                        m = int(np.argmin(c)); D[ii, jj] = c[m]; ptr[ii, jj] = m + 1
                ii, jj = la, lb
                while ii > 0 or jj > 0:
                    p = ptr[ii, jj]
                    if p == 1:
                        if S[ii-1, jj-1] >= 0.55:
                            pairs[a0 + ii - 1] = (b0 + jj - 1, float(S[ii-1, jj-1]))
                        ii -= 1; jj -= 1
                    elif p == 2: ii -= 1
                    else: jj -= 1
    log('token pairs: %d (%.0f%% of script)' % (len(pairs), 100.0 * len(pairs) / max(1, len(stok))))

    tw = np.array(twrd)
    pin_t, tokpos = {}, {}
    for i, ui in enumerate(suni):
        if ui not in tokpos: tokpos[ui] = [i, i]
        tokpos[ui][1] = i
    for ui in range(n):
        if ui not in tokpos: continue
        f, l = tokpos[ui]
        toks = list(range(f, l + 1))
        strong = sorted(pairs[i] for i in toks if i in pairs and pairs[i][1] >= 0.78)
        if not strong: continue
        cov = len(strong) / max(1, len(toks))
        if cov >= 0.4 or (len(toks) <= 3 and len(strong) >= max(1, len(toks) - 1)):
            pin_t[ui] = ws_[tw[strong[0][0]]] - 0.15
    ks = sorted(pin_t); arr = np.array([pin_t[k] for k in ks])
    for i, k in enumerate(ks):
        lo, hi = max(0, i - 3), min(len(ks), i + 4)
        med = np.median(np.r_[arr[lo:i], arr[i + 1:hi]]) if hi - lo > 1 else arr[i]
        if abs(pin_t[k] - med) > 4: del pin_t[k]

    pins = []; last = -1e9
    for k in sorted(pin_t):
        t = pin_t[k]
        if t >= last - 0.05:
            tt = max(t, last + 0.3 if pins else 0.3)
            if tt - t <= 1.5:
                pins.append([k, tt]); last = tt
    log('pins: %d/%d' % (len(pins), n))
    if len(pins) < 2:
        sys.exit('!! alignment failed — whisper ma lqatch match me3a script.')

    lens = np.array([max(4, len(u)) for u in units], float)
    span = pins[-1][1] - pins[0][1]
    r = sum(lens[pins[0][0]:pins[-1][0] + 1]) / max(span, 1.0)
    exp_d = lens / r
    t0 = np.full(n, np.nan)
    for k, tk in pins: t0[k] = tk
    for a in range(len(pins) - 1):
        ka, ta = pins[a]; kb, tb = pins[a + 1]
        if kb == ka + 1: continue
        E = exp_d[ka:kb].sum(); Wd = tb - ta
        sc = Wd / E if E > 0 else 1.0
        cur = ta
        for u in range(ka, kb):
            if u == ka: continue
            cur += exp_d[u - 1] * sc
            t0[u] = cur
    kf, tf = pins[0]; cur = tf
    for u in range(kf - 1, -1, -1):
        cur -= exp_d[u + 1]; t0[u] = max(0.3, cur)
    kl, tl = pins[-1]; cur = tl
    for u in range(kl + 1, n):
        cur += exp_d[u - 1]; t0[u] = cur

    tend = np.zeros(n)
    for ui in range(n):
        if ui in tokpos and tokpos[ui][1] in pairs:
            tend[ui] = we_[tw[pairs[tokpos[ui][1]][0]]] + 0.45
        else:
            tend[ui] = t0[ui] + exp_d[ui] + 0.25
    t1 = np.minimum(np.r_[t0[1:], TOTAL - 1.0] - 0.02, tend)
    t1 = np.maximum(t0 + 0.6, np.minimum(t1, t0 + 8.0))
    subs = [(float(t0[k]), float(t1[k]), units[k]) for k in range(n) if t0[k] < TOTAL]
    log('subs ready: %d (med spacing %.2fs)' % (len(subs), float(np.median(np.diff([s[0] for s in subs]))) if len(subs) > 2 else 0))

    # spectrum ref (97th percentile)
    WIN = 2048
    win = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    BANDS = 80
    edges = np.exp(np.linspace(math.log(70), math.log(8000), BANDS + 1))
    idxf = np.searchsorted(freqs, edges)
    IDXL = idxf[:-1].astype(np.int64); WID = (idxf[1:] - idxf[:-1]).astype(np.int64); LASTF = int(idxf[-1])
    SRF = SR / FPS
    offs = np.round(np.arange(TOTAL_FRAMES) * SRF).astype(np.int64)
    db = np.zeros((TOTAL_FRAMES, BANDS), np.float32)
    for j in range(TOTAL_FRAMES):
        s = audio[offs[j]:offs[j] + WIN]
        if len(s) < WIN: s = np.pad(s, (0, WIN - len(s)))
        mag = np.abs(np.fft.rfft(s * win))
        sums = np.add.reduceat(mag[:LASTF], IDXL)
        band = np.where(WID > 0, sums / np.maximum(WID, 1), mag[IDXL])
        db[j] = 20 * np.log10(band + 1e-6)
    ref = np.percentile(db, 97, axis=0).astype(np.float32)
    pickle.dump({'subs': subs, 'ref': ref, 'total_frames': TOTAL_FRAMES}, open(CACHE, 'wb'))
    # SRT export
    def ts(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
        return '%02d:%02d:%06.3f' % (h, m, s)
    with open(os.path.join(OUT, 'subs.srt'), 'w', encoding='utf-8') as f:
        for i, s in enumerate(subs, 1):
            f.write('%d\n%s --> %s\n%s\n\n' % (i, ts(s[0]).replace('.', ','), ts(s[1]).replace('.', ','), s[2]))
    log('analyze ok -> cache + subs.srt')
    return subs, ref, TOTAL_FRAMES

# =====================================================================
# STAGE 3 — RENDER (EXACT visual identity of the first podcast)
# =====================================================================
def load_bg():
    from PIL import Image, ImageFilter, ImageDraw, ImageOps
    if BG_URL:
        import requests, io
        log('download bg from BG_URL...')
        r = requests.get(BG_URL, timeout=60)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert('RGBA')
        img = ImageOps.fit(img, (W, H), Image.LANCZOS)
    else:
        img = Image.open(os.path.join(BASE, 'assets', 'live_final_bg.png')).convert('RGBA')
        # signature touches of the original studio bg
        img.paste(img.crop((300, 140, 518, 206)), (36, 34))
        img.paste(img.crop((28, 26, 262, 108)).filter(ImageFilter.GaussianBlur(7)), (28, 26))
        img.paste(img.crop((375, 165, 795, 217)), (375, 99))
        ImageDraw.Draw(img).line((330, 103, 982, 103), fill=(216, 163, 88, 235), width=3)
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    CX, CY, R0 = 545, 585, 252
    ring = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(ring)
    for rr, a in [(R0 - 6, 60), (R0 + 2, 95)]:
        rd.ellipse((CX - rr, CY - rr, CX + rr, CY + rr), outline=(244, 214, 130, a), width=2)
    return Image.alpha_composite(img, ring)

def stage_render(mp3):
    from PIL import Image, ImageDraw, ImageFont
    subs, ref, TOTAL_FRAMES = stage_load_cache(mp3)
    DUR = TOTAL_FRAMES / FPS
    OUTRO_T0 = DUR - OUTRO_S

    log('decode full audio + per-frame spectrum...')
    WIN = 2048
    win = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    BANDS = 80
    edges = np.exp(np.linspace(math.log(70), math.log(8000), BANDS + 1))
    idxf = np.searchsorted(freqs, edges)
    IDXL = idxf[:-1].astype(np.int64); WID = (idxf[1:] - idxf[:-1]).astype(np.int64); LASTF = int(idxf[-1])
    SRF = SR / FPS
    au = decode(mp3, 0.0, DUR + 1.0)
    offs = np.round(np.arange(TOTAL_FRAMES) * SRF).astype(np.int64)
    db = np.zeros((TOTAL_FRAMES, BANDS), np.float32)
    for j in range(TOTAL_FRAMES):
        s = au[offs[j]:offs[j] + WIN]
        if len(s) < WIN: s = np.pad(s, (0, WIN - len(s)))
        mag = np.abs(np.fft.rfft(s * win))
        sums = np.add.reduceat(mag[:LASTF], IDXL)
        band = np.where(WID > 0, sums / np.maximum(WID, 1), mag[IDXL])
        db[j] = 20 * np.log10(band + 1e-6)
    norm_ = np.clip((db - (ref - 38)) / 38, 0, 1) ** 1.4
    sm = np.zeros_like(norm_); sm[0] = norm_[0]
    for j in range(1, TOTAL_FRAMES):
        dv = norm_[j] - sm[j - 1]
        sm[j] = sm[j - 1] + np.where(dv > 0, dv * 0.5, dv * 0.18)
    sm = np.clip(sm * 1.18, 0, 1)
    del db, norm_, au
    log('spectrum ok')

    base_img = load_bg()
    CX, CY, R0 = 545, 585, 252
    BASE_NP = np.asarray(base_img.convert('RGBA'), dtype=np.float32)
    GOLD = np.array([250, 222, 132, 255], np.float32)
    GOLD_HI = np.array([255, 240, 190, 255], np.float32)
    ang = np.linspace(0, 2 * math.pi, BANDS, endpoint=False) - math.pi / 2
    ux, uy = np.cos(ang), np.sin(ang)
    LM = 185
    rrv = np.arange(R0 + 8, R0 + 8 + LM)
    XS = np.rint(CX + np.outer(ux, rrv)).astype(np.int32)
    YS = np.rint(CY + np.outer(uy, rrv)).astype(np.int32)

    FONTS = os.path.join(BASE, 'assets', 'fonts')
    fb = ImageFont.truetype(FONTS + '/Tajawal-Bold.ttf', 52)
    ft = ImageFont.truetype(FONTS + '/Tajawal-Bold.ttf', 74)
    fm = ImageFont.truetype(FONTS + '/Tajawal-Medium.ttf', 40)
    AR = dict(anchor='mm', direction='rtl', language='ar')
    SUB_Y, SUB_FILL = 174, (255, 243, 200)

    def like_chip():
        w, h = 190, 74
        c = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(c)
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=37, fill=(8, 11, 22, 210), outline=(232, 200, 120, 255), width=3)
        d.rounded_rectangle((18, 34, 36, 58), radius=4, fill=(250, 222, 132))
        d.polygon([(36, 36), (52, 20), (58, 24), (52, 40), (64, 40), (66, 50), (58, 58), (36, 58)], fill=(250, 222, 132))
        fchip = ImageFont.truetype(FONTS + '/Tajawal-Bold.ttf', 40)
        d.text((122, h / 2), 'لايك', font=fchip, fill=(255, 233, 170), **AR)
        return c
    CHIP = like_chip()
    CHIP_NP = np.asarray(CHIP, dtype=np.float32)
    CHIP_POS = (W - CHIP.width - 18, 668)

    def end_card():
        c = Image.new('RGBA', (W, H), (5, 8, 18, 255))
        from PIL import ImageFilter
        g = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(g).ellipse((W / 2 - 320, 200, W / 2 + 320, 840), fill=(120, 90, 30, 255))
        c.alpha_composite(g.filter(ImageFilter.GaussianBlur(140)))
        d = ImageDraw.Draw(c)
        d.text((W / 2, 330), CHANNEL, font=ft, fill=(243, 220, 150), **AR)
        d.text((W / 2, 420), 'شكرا على الاستماع من القلب', font=fm, fill=(222, 206, 168), **AR)
        pw = 560
        d.rounded_rectangle((W / 2 - pw / 2, 500, W / 2 + pw / 2, 608), radius=54, fill=(206, 32, 40, 255))
        d.text((W / 2, 554), 'اشترك في القناة', font=fb, fill=(255, 255, 255), **AR)
        d.rounded_rectangle((W / 2 - 330, 650, W / 2 + 330, 734), radius=42, outline=(232, 200, 120, 255), width=3)
        d.text((W / 2, 692), 'و دير لايك و فعّل الجرس', font=fm, fill=(255, 233, 170), **AR)
        d.text((W / 2, 830), 'بودكاست بالدارجة المغربية', font=fm, fill=(170, 155, 118), **AR)
        return c
    CARD_NP = np.asarray(end_card(), dtype=np.float32)

    starts = np.array([s[0] for s in subs]); ends_ = np.array([s[1] for s in subs])
    sub_cache = {}
    def sub_tile(ix):
        if ix in sub_cache: return sub_cache[ix]
        layer = Image.new('RGBA', (W, 110), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        s = subs[ix][2]
        tw = d.textlength(s, font=fb, direction='rtl', language='ar')
        d.rounded_rectangle((W / 2 - tw / 2 - 34, 8, W / 2 + tw / 2 + 34, 110 - 8), radius=24, fill=(4, 6, 14, 155))
        d.text((W / 2, 55), s, font=fb, fill=SUB_FILL + (255,), stroke_width=5, stroke_fill=(0, 0, 0, 255), **AR)
        sub_cache[ix] = np.asarray(layer, dtype=np.float32)
        return sub_cache[ix]

    def put_rgba(dst, src_np, x0, y0):
        hh, ww = src_np.shape[:2]
        a = src_np[..., 3:4] / 255.0
        roi = dst[y0:y0 + hh, x0:x0 + ww]
        roi[...] = roi * (1 - a) + src_np[..., :4] * a

    out = os.path.join(OUT, 'podcast_final.mp4')
    log('render %d frames (%.1f min) -> %s' % (TOTAL_FRAMES, DUR / 60, out))
    proc = subprocess.Popen([FF, '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', '%dx%d' % (W, H),
                             '-r', str(FPS), '-i', '-', '-i', mp3,
                             '-map', '0:v', '-map', '1:a', '-c:v', 'libx264',
                             '-preset', 'veryfast', '-crf', '28', '-pix_fmt', 'yuv420p',
                             '-c:a', 'aac', '-b:a', '128k', '-shortest',
                             '-movflags', '+faststart', out],
                            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    FADE_IN = 20
    t_start = time.time()
    prev = np.asarray([os.path.join(OUT, 'preview_a.jpg'), os.path.join(OUT, 'preview_b.jpg'), os.path.join(OUT, 'preview_c.jpg')])
    prev_at = [120, TOTAL_FRAMES // 2, TOTAL_FRAMES - int(6 * FPS)]
    for f in range(TOTAL_FRAMES):
        t = f / FPS
        fnp = BASE_NP.copy()
        if f < FADE_IN:
            fnp *= (f / FADE_IN)
            fnp[..., 3] = 255
        if t >= OUTRO_T0 - 1.0:
            p = min(1.0, (t - (OUTRO_T0 - 1.0)) / 1.0)
            p = 1 - (1 - p) ** 3
            alpha = (90 + 165 * p) / 255.0
            fnp[...] = fnp * (1 - alpha) + CARD_NP * alpha
        else:
            mags = sm[f]
            Ls = (6 + mags ** 1.2 * 175).astype(np.int32)
            gacc = np.zeros((H, W, 4), np.float32)
            cacc = np.zeros((H, W, 4), np.float32)
            for i in range(BANDS):
                L = min(Ls[i], LM)
                xs = XS[i, :L]; ys = YS[i, :L]
                col = GOLD_HI if mags[i] > 0.72 else GOLD
                gacc[ys, xs] += col
                cacc[ys, xs] = col
                cacc[np.clip(ys + 1, 0, H - 1), xs] = col
                cacc[ys, np.clip(xs + 1, 0, W - 1)] = col
            fnp = fnp + gacc * 0.38 + cacc
            ph = t % 30.0
            scp = 1.0 + 0.16 * math.sin(math.pi * ph / 1.1) ** 2 if ph < 1.1 else 1.0
            if scp > 1.0:
                ch = CHIP.resize((int(CHIP.width * scp), int(CHIP.height * scp)), Image.BILINEAR)
                put_rgba(fnp, np.asarray(ch, dtype=np.float32), CHIP_POS[0] - (ch.width - CHIP.width) // 2, CHIP_POS[1] - (ch.height - CHIP.height) // 2)
            else:
                put_rgba(fnp, CHIP_NP, *CHIP_POS)
            ix = bisect.bisect_right(starts, t) - 1
            if ix >= 0 and t <= ends_[ix]:
                put_rgba(fnp, sub_tile(ix), 0, SUB_Y - 55)
        u8 = np.clip(fnp, 0, 255).astype(np.uint8)
        proc.stdin.write(u8.tobytes())
        for pi, pf in enumerate(prev_at):
            if f == pf:
                Image.fromarray(u8[..., :3]).save(prev[pi], quality=88)
        if f % 4800 == 0:
            el = time.time() - t_start
            eta = el / max(1, f + 1) * (TOTAL_FRAMES - f - 1)
            log('frame %d/%d | elapsed %.0fs | ETA ~%.0fs' % (f, TOTAL_FRAMES, el, eta))
    proc.stdin.close()
    rc = proc.wait()
    assert rc == 0, 'encoder failed'
    sz = os.path.getsize(out) / 1e6
    log('DONE -> %s (%.1f MB) in %.0fs' % (out, sz, time.time() - t_start))

def stage_load_cache(mp3):
    if os.path.exists(CACHE):
        c = pickle.load(open(CACHE, 'rb'))
        return c['subs'], c['ref'], c['total_frames']
    return stage_analyze(mp3)

# =====================================================================
if __name__ == '__main__':
    mp3 = os.path.join(OUT, 'master.mp3')
    if ONLY in ('all', 'tts'):
        mp3 = stage_tts()
    if ONLY in ('all', 'analyze'):
        stage_analyze(mp3)
    if ONLY in ('all', 'render'):
        stage_render(mp3)
    log('🏭 مصنع كمل! المخرجات فـ out/ — الله يبارك 🎉')
