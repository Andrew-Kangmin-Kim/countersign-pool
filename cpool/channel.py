"""Acoustic channels: TTS -> noise -> forced-choice listener.

Two channels behind one interface (the held-out one shares NO model with the
fit one):
  original  edge-tts (Microsoft neural, InJoon) -> Whisper-small, teacher-forced
            log P(word | audio), mean over word tokens (the notebook's channel)
  heldout   Google TTS (gTTS)                   -> a non-Whisper CTC model chosen
            by scripts/pilot_heldout.py; CTC log-likelihood per label

Forced choice (not free transcription): a sentry decides which expected word
was heard.  Two decision rules are recorded for every trial:
  raw   argmax score
  corr  argmax (score - prior), prior = mean score on pure-noise audio
        (removes the recogniser's preference for some strings)

Fixes relative to the notebook:
  * SNR is set against ACTIVE speech power (leading/trailing silence trimmed),
    so nominal SNR is the SNR the listener actually faces.
  * Noise for (channel, word, snr, trial, seed) comes from a hash-derived seed,
    so results do not depend on cache state or call order.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

SR = 16000


def _device(pref: str):
    import torch
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little")


def active_power(y: np.ndarray, top_db: float = 40.0) -> float:
    import librosa
    yt, _ = librosa.effects.trim(y, top_db=top_db)
    yt = yt if len(yt) > SR // 20 else y
    return float(np.mean(yt.astype(np.float64) ** 2))


def add_noise(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(y)).astype(np.float32)
    p_sig = active_power(y)
    noise *= np.sqrt(p_sig / (np.mean(noise ** 2) * 10 ** (snr_db / 10.0)))
    z = y + noise
    return (z / (np.abs(z).max() + 1e-9)).astype(np.float32)


# ===================================================================== TTS
class TTS:
    name = "base"

    def __init__(self, cache_dir: Path):
        self.dir = Path(cache_dir) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, w):
        return self.dir / f"{w}.npy"

    def synth(self, words: list[str]) -> dict[str, np.ndarray]:
        todo = [w for w in words if not self._path(w).exists()]
        if todo:
            print(f"[{self.name}] synthesizing {len(todo)} words", flush=True)
            t0 = time.time()
            self._synth_many(todo)
            print(f"[{self.name}] done in {time.time()-t0:.0f}s", flush=True)
        out = {}
        for w in words:
            y = np.load(self._path(w))
            out[w] = (y / (np.abs(y).max() + 1e-9)).astype(np.float32)
        return out

    def _synth_many(self, words):
        raise NotImplementedError


class EdgeTTS(TTS):
    name = "edge"

    def __init__(self, cache_dir, voice="ko-KR-InJoonNeural"):
        self.voice = voice
        self.name = f"edge_{voice}"
        super().__init__(cache_dir)

    def _synth_many(self, words):
        import edge_tts
        import librosa
        tmp = self.dir / "_mp3"
        tmp.mkdir(exist_ok=True)

        async def run():
            sem = asyncio.Semaphore(8)

            async def one(w):
                async with sem:
                    for attempt in range(5):
                        try:
                            await edge_tts.Communicate(w, self.voice).save(str(tmp / f"{w}.mp3"))
                            return
                        except Exception:
                            await asyncio.sleep(2 * (attempt + 1))
                    raise RuntimeError(f"edge-tts failed for {w}")
            await asyncio.gather(*(one(w) for w in words))

        asyncio.run(run())
        for w in words:
            y, _ = librosa.load(str(tmp / f"{w}.mp3"), sr=SR)
            np.save(self._path(w), y.astype(np.float32))


class GoogleTTS(TTS):
    """Google TTS via gTTS (Google Translate's unofficial endpoint; network).
    Native Hangul input.  Chosen as the held-out TTS after MMS-TTS-kor failed
    the smoke test (romanized input, 0.4–0.6 s clipped utterances, 4/8 clean
    self-recognition vs 8/8 here)."""
    name = "gtts_ko"

    def _synth_many(self, words):
        import io
        import librosa
        from gtts import gTTS
        for k, w in enumerate(words):
            for attempt in range(6):
                try:
                    b = io.BytesIO()
                    gTTS(w, lang="ko").write_to_fp(b)
                    b.seek(0)
                    y, _ = librosa.load(b, sr=SR)
                    break
                except Exception:
                    time.sleep(3 * (attempt + 1))
            else:
                raise RuntimeError(f"gTTS failed for {w}")
            np.save(self._path(w), y.astype(np.float32))
            if k % 50 == 49:
                time.sleep(1.0)          # stay polite to the endpoint


# ===================================================================== ASR
class Scorer:
    """score(audio) -> array of per-candidate scores (higher = more likely)."""
    name = "base"

    def prepare(self, candidates: list[str]):
        raise NotImplementedError

    def score(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def prior(self, n=8, seed=0) -> np.ndarray:
        acc = None
        for k in range(n):
            rng = np.random.RandomState(stable_seed("prior", self.name, seed, k))
            z = rng.randn(SR).astype(np.float32)
            z /= np.abs(z).max() + 1e-9
            s = self.score(z)
            acc = s if acc is None else acc + s
        return acc / n


class WhisperScorer(Scorer):
    def __init__(self, model_name="small", device="cpu", chunk=256):
        import whisper
        self.name = f"whisper_{model_name}"
        self.device = device
        self.chunk = chunk
        self.model = whisper.load_model(model_name, device=device).eval()
        self.tok = whisper.tokenizer.get_tokenizer(
            self.model.is_multilingual, num_languages=self.model.num_languages,
            language="ko", task="transcribe")

    def prepare(self, candidates):
        import torch
        sot = list(self.tok.sot_sequence_including_notimestamps)
        seqs = [sot + self.tok.encode(w) + [self.tok.eot] for w in candidates]
        L = max(len(s) for s in seqs)
        toks = torch.full((len(seqs), L), self.tok.eot, dtype=torch.long)
        msk = torch.zeros((len(seqs), L), dtype=torch.bool)
        for i, s in enumerate(seqs):
            toks[i, :len(s)] = torch.tensor(s)
            msk[i, len(sot):len(s)] = True
        self.toks, self.msk = toks.to(self.device), msk.to(self.device)

    def score(self, y):
        import torch
        import whisper
        from contextlib import nullcontext
        amp = torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" \
            else nullcontext()
        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(y),
                                              n_mels=self.model.dims.n_mels).to(self.device)
            with amp:
                feats = self.model.encoder(mel.unsqueeze(0))
            out = []
            for i in range(0, self.toks.shape[0], self.chunk):
                t, m = self.toks[i:i + self.chunk], self.msk[i:i + self.chunk]
                xa = feats.expand(t.shape[0], -1, -1)
                with amp:
                    logits = self.model.decoder(t[:, :-1], xa)
                lp = torch.log_softmax(logits.float(), -1)
                tlp = lp.gather(-1, t[:, 1:].unsqueeze(-1)).squeeze(-1)
                mm = m[:, 1:]
                out.append(((tlp * mm).sum(1) / mm.sum(1).clamp(min=1)).cpu().numpy())
        return np.concatenate(out)


class CTCScorer(Scorer):
    """wav2vec2 CTC.  Score = -CTC loss / label length (mean log-lik per label).
    Candidates are tokenized with the model's own vocabulary; if a syllable is
    out of vocabulary the candidate is tried as NFD jamo; failure is an error."""

    def __init__(self, model_name, device="cpu", chunk=256):
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.name = "ctc_" + model_name.replace("/", "_")
        self.device = device
        self.chunk = chunk
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
        self.blank = self.model.config.pad_token_id

    def _labels(self, w):
        import unicodedata
        tk = self.proc.tokenizer
        unk = tk.unk_token_id
        from .hangul import decompose
        compat = "".join("".join(p for p in decompose(ch)) for ch in w)   # ㄱㅏ… jamo
        for form in (w, unicodedata.normalize("NFD", w), compat):
            ids = tk(form, add_special_tokens=False).input_ids
            # a tokenizer that silently drops characters is not representing w
            if len(ids) < len(form) and form is not w:
                continue
            ids = [i for i in ids if i != tk.word_delimiter_token_id] \
                if hasattr(tk, "word_delimiter_token_id") else ids
            if ids and unk not in ids:
                return ids
        return None

    def representable(self, w) -> bool:
        return self._labels(w) is not None

    def prepare(self, candidates):
        """Words with a syllable outside the model's vocabulary cannot be scored;
        they get -inf (never chosen) and are listed in self.unrepresentable."""
        labs = [self._labels(w) for w in candidates]
        self.unrepresentable = [w for w, l in zip(candidates, labs) if l is None]
        self.valid = np.array([l is not None for l in labs])
        self.labels = [l for l in labs if l is not None]

    def score(self, y):
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            inp = self.proc(y, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
            lp = torch.log_softmax(self.model(inp).logits.float(), -1)[0]   # (T, V)
            T = lp.shape[0]
            out = []
            for i in range(0, len(self.labels), self.chunk):
                labs = self.labels[i:i + self.chunk]
                B = len(labs)
                tgt = torch.cat([torch.tensor(l) for l in labs]).to(self.device)
                tl = torch.tensor([len(l) for l in labs])
                loss = F.ctc_loss(lp.unsqueeze(1).expand(T, B, -1), tgt,
                                  torch.full((B,), T, dtype=torch.long), tl,
                                  blank=self.blank, reduction="none", zero_infinity=True)
                out.append((-loss.cpu() / tl).numpy())
        s = np.full(len(self.valid), -np.inf)
        s[self.valid] = np.concatenate(out) if out else []
        return s


class SenseVoiceScorer(Scorer):
    """FunAudioLLM SenseVoice-Small (non-autoregressive CTC, BPE units; not Whisper).
    Replicates its inference input (language=ko, no ITN, event/emotion queries),
    drops the 4 query frames, and scores each candidate by CTC log-likelihood
    per BPE token."""

    def __init__(self, model_name="iic/SenseVoiceSmall", device="cpu", chunk=256):
        import torch
        from funasr import AutoModel
        self.name = "sensevoice_small"
        self.device = device
        self.chunk = chunk
        am = AutoModel(model=model_name, device=device, disable_update=True, log_level="ERROR")
        self.m = am.model.eval()
        self.frontend = am.kwargs["frontend"]
        self.tok = am.kwargs["tokenizer"]
        self.blank = self.m.blank_id
        import torch as _t
        self._t = _t

    def representable(self, w) -> bool:
        ids = self.tok.encode(w)
        return bool(ids) and self.tok.decode(ids).replace(" ", "") == w

    def prepare(self, candidates):
        labs = [self.tok.encode(w) if self.representable(w) else None for w in candidates]
        self.valid = np.array([l is not None for l in labs])
        self.labels = [l for l in labs if l is not None]
        self.unrepresentable = [w for w, l in zip(candidates, labs) if l is None]

    def _logprobs(self, y):
        from funasr.utils.load_utils import extract_fbank
        torch = self._t
        m = self.m
        with torch.no_grad():
            speech, lens = extract_fbank([torch.from_numpy(y)], data_type="sound",
                                         frontend=self.frontend)
            speech, lens = speech.to(self.device), lens.to(self.device)
            q = lambda ids: m.embed(torch.LongTensor([ids]).to(self.device))
            speech = torch.cat((q([m.textnorm_dict["woitn"]]), speech), dim=1)
            speech = torch.cat((q([m.lid_dict["ko"]]), q([1, 2]), speech), dim=1)
            lens = lens + 4
            enc, enc_lens = m.encoder(speech, lens)
            if isinstance(enc, tuple):
                enc = enc[0]
            lp = m.ctc.log_softmax(enc)[0, 4: int(enc_lens[0]), :]
        return lp.float()

    def score(self, y):
        import torch.nn.functional as F
        torch = self._t
        lp = self._logprobs(y)
        T = lp.shape[0]
        out = []
        with torch.no_grad():
            for i in range(0, len(self.labels), self.chunk):
                labs = self.labels[i:i + self.chunk]
                B = len(labs)
                tgt = torch.cat([torch.tensor(l) for l in labs]).to(self.device)
                tl = torch.tensor([len(l) for l in labs])
                loss = F.ctc_loss(lp.unsqueeze(1).expand(T, B, -1), tgt,
                                  torch.full((B,), T, dtype=torch.long), tl,
                                  blank=self.blank, reduction="none", zero_infinity=True)
                out.append((-loss.cpu() / tl).numpy())
        s = np.full(len(self.valid), -np.inf)
        s[self.valid] = np.concatenate(out) if out else []
        return s


# ===================================================================== channel
class Channel:
    def __init__(self, name: str, tts: TTS, scorer: Scorer, cache_db: Path):
        self.name, self.tts, self.scorer = name, tts, scorer
        self.db = sqlite3.connect(str(cache_db))
        self.db.execute("CREATE TABLE IF NOT EXISTS fc (k TEXT PRIMARY KEY, raw TEXT, corr TEXT)")

    def _cset(self, candidates):
        return hashlib.sha256("|".join(candidates).encode("utf-8")).hexdigest()[:10]

    def gate(self, words: list[str], distractors: list[str], exclude: dict | None = None):
        """Clean-audio gate: can this TTS->ASR say and hear the word at all?
        Each word competes against fixed random distractors that NEVER include
        the words it is paired with (exclude[w]).  Testing against the pair
        partner would remove exactly the confusable pairs under study."""
        exclude = exclude or {}
        audio = self.tts.synth(words)
        ok = {}
        for w in words:
            bad = exclude.get(w, set()) | {w}
            cands = [w] + [d for d in distractors if d not in bad]
            self.scorer.prepare(cands)
            ok[w] = int(np.argmax(self.scorer.score(audio[w]))) == 0
        return ok

    def self_recognition(self, words: list[str], candidates: list[str] | None = None):
        """clean-audio forced choice; -> dict word -> bool"""
        candidates = candidates or words
        audio = self.tts.synth(words)
        self.scorer.prepare(candidates)
        ok = {}
        for w in words:
            ok[w] = candidates[int(np.argmax(self.scorer.score(audio[w])))] == w
        return ok

    def run(self, spoken: list[str], candidates: list[str], snrs, trials: int, seed: int,
            log_every=200):
        """forced choice for every (spoken word, snr, trial); candidate set fixed."""
        audio = self.tts.synth(sorted(set(spoken)))
        cs = self._cset(candidates)
        prepared = False
        prior = None
        rows = []
        total = len(spoken) * len(snrs) * trials
        t0 = time.time()
        done = 0
        for u in spoken:
            for snr in snrs:
                for t in range(trials):
                    # key pins the MODELS, not just the role: swapping a TTS or
                    # ASR must never reuse old decisions
                    k = f"{self.tts.name}|{self.scorer.name}|{cs}|{u}|{snr}|{t}|{seed}"
                    hit = self.db.execute("SELECT raw, corr FROM fc WHERE k=?", (k,)).fetchone()
                    if hit is None:
                        if not prepared:
                            self.scorer.prepare(candidates)
                            prior = self.scorer.prior(seed=seed)
                            prepared = True
                        y = add_noise(audio[u], snr, stable_seed(self.name, u, snr, t, seed))
                        s = self.scorer.score(y)
                        corr = np.where(np.isfinite(s) & np.isfinite(prior), s - prior, -np.inf)
                        hit = (candidates[int(np.argmax(s))], candidates[int(np.argmax(corr))])
                        self.db.execute("INSERT OR REPLACE INTO fc VALUES (?,?,?)", (k, *hit))
                    rows.append((u, snr, t, hit[0], hit[1]))
                    done += 1
                    if done % log_every == 0:
                        self.db.commit()
                        r = done / max(time.time() - t0, 1e-9)
                        print(f"  [{self.name}] {done:,}/{total:,}  {r:.1f}/s  "
                              f"ETA {(total-done)/max(r,1e-9)/60:.1f} min", flush=True)
        self.db.commit()
        return pd.DataFrame(rows, columns=["true_word", "snr", "trial", "pred_raw", "pred_corr"])


def make_tts(spec: dict, cache_dir: Path) -> TTS:
    if spec["tts"] == "edge":
        return EdgeTTS(cache_dir / "tts", spec.get("voice", "ko-KR-InJoonNeural"))
    if spec["tts"] == "gtts":
        return GoogleTTS(cache_dir / "tts")
    raise ValueError(spec["tts"])


def make_asr(spec: dict, device: str, chunk: int) -> Scorer:
    if spec["asr"] == "whisper":
        return WhisperScorer(spec["model"], device, chunk)
    if spec["asr"] == "ctc":
        return CTCScorer(spec["model"], device, chunk)
    if spec["asr"] == "sensevoice":
        return SenseVoiceScorer(spec.get("model", "iic/SenseVoiceSmall"), device, chunk)
    raise ValueError(spec["asr"])


def build_channel(cfg: dict, which: str, cache_dir: Path) -> Channel:
    c = cfg["channels"]
    spec = c[which]
    if not spec.get("asr"):
        raise RuntimeError(f"channel {which!r} has no ASR set — run scripts/pilot_heldout.py")
    dev = _device(c["device"])
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tts = make_tts(spec, cache_dir)
    sc = make_asr(spec, dev, c["chunk"])
    print(f"channel {which}: tts={tts.name} asr={sc.name} device={dev}", flush=True)
    return Channel(which, tts, sc, cache_dir / "trials.sqlite")
