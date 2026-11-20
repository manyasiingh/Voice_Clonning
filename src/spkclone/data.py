"""LibriSpeech download/index, audio loading, P x K dataset and a sampler that draws P DISTINCT speakers per batch."""
import hashlib
import math
import random
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import soundfile as sf
import torch

SR = 16000
EXT = {".wav", ".flac", ".mp3", ".ogg"}
OPENSLR_MIRRORS = ["https://www.openslr.org/resources/12", "https://openslr.elda.org/resources/12",
                   "https://openslr.trmal.net/resources/12"]


# --------------------------------------------------------------------------- #
# Download (resume-safe, checksummed, safe extraction)
# --------------------------------------------------------------------------- #
def download(url, dst, retries=5):
    """Download to `dst.part`, resuming with an HTTP Range request after interruptions; rename when complete."""
    dst = Path(dst)
    part = dst.with_name(dst.name + ".part")
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored Range: start over
                    have = 0
                length = r.headers.get("Content-Length")
                total = have + int(length) if length else None
                with open(part, "ab" if have else "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
            size = part.stat().st_size
            if total is not None and size != total:
                raise IOError(f"incomplete download: {size} of {total} bytes")
            part.replace(dst)
            return dst
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:  # requested range starts at end of file: already complete
                part.replace(dst)
                return dst
            print(f"  HTTP {e.code} for {url} (attempt {attempt + 1}/{retries})")
        except (urllib.error.URLError, OSError) as e:
            print(f"  download interrupted: {e} (attempt {attempt + 1}/{retries}); resuming")
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}")


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def expected_md5(filename):
    """Look up the official checksum in OpenSLR's md5sum.txt (None if no mirror answers)."""
    for base in OPENSLR_MIRRORS:
        try:
            with urllib.request.urlopen(f"{base}/md5sum.txt", timeout=30) as r:
                for line in r.read().decode().splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1] == filename:
                        return parts[0]
        except (urllib.error.URLError, OSError):
            continue
    return None


def safe_extract(tgz, dest):
    """Extract a tar archive without allowing paths outside `dest`, links or device files."""
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz) as t:
        if hasattr(tarfile, "data_filter"):  # Python >= 3.12 and patched 3.8+
            t.extractall(dest, filter="data")
            return
        members = t.getmembers()
        for m in members:
            target = (dest / m.name).resolve()
            if not (m.isfile() or m.isdir()) or not target.is_relative_to(dest):
                raise ValueError(f"refusing to extract unsafe tar member: {m.name}")
        t.extractall(dest, members=members)


def ensure_librispeech(subset, root, verify_md5=True):
    """Return root/LibriSpeech/<subset>, downloading + verifying + extracting it if needed.
    A marker file is written only after a full extraction, so an interrupted run re-extracts instead of
    silently using half a dataset."""
    root = Path(root)
    d = root / "LibriSpeech" / subset
    marker = d / ".spkclone_complete"
    if marker.exists():
        return d
    root.mkdir(parents=True, exist_ok=True)
    name = f"{subset}.tar.gz"
    tgz = root / name
    if not tgz.exists():
        print(f"Downloading LibriSpeech {subset} ...")
        download(f"{OPENSLR_MIRRORS[0]}/{name}", tgz)
    if verify_md5:
        want = expected_md5(name)
        if want is None:
            print(f"  WARNING: could not fetch md5sum.txt; only the size check was done for {name}")
        else:
            got = md5sum(tgz)
            if got != want:
                tgz.unlink()
                raise RuntimeError(f"md5 mismatch for {name}: got {got}, expected {want} (file deleted, re-run)")
            print(f"  md5 ok ({got})")
    print(f"Extracting {name} ...")
    safe_extract(tgz, root)
    marker.write_text("ok\n")
    tgz.unlink()  # save disk (Kaggle working dir is ~20 GB)
    return d


def resolve_split(directory, subset, root):
    return Path(directory) if directory else ensure_librispeech(subset, root)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
def build_index(root, min_utts=2):
    """{speaker_id: [audio files]} from a root/<speaker_id>/**/*.flac layout."""
    idx = {}
    for spk in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        fs = sorted(str(f) for f in spk.rglob("*") if f.suffix.lower() in EXT)
        if len(fs) >= min_utts:
            idx[spk.name] = fs
    print(f"  {root}: {len(idx)} speakers, {sum(map(len, idx.values()))} files")
    return idx


def session_of(path):
    """Recording session of a LibriSpeech file = its chapter directory (root/<spk>/<chapter>/<file>)."""
    return Path(path).parent.name


def assert_disjoint(splits, allow_overlap=False):
    """splits: {name: index}. Raise if any two splits share a speaker (unless allow_overlap, for smoke tests)."""
    names = list(splits)
    overlaps = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(splits[a]) & set(splits[b]))
            if common:
                overlaps[f"{a}&{b}"] = len(common)
    if overlaps:
        msg = f"speaker overlap between splits: {overlaps}"
        if not allow_overlap:
            raise ValueError(msg + " (use different speakers per split; --allow_overlap only for smoke tests)")
        print("WARNING:", msg, "- results are NOT valid held-out numbers")
    return overlaps


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def load_wav(path, max_sec=None, sr=SR):
    """Mono float32 tensor at `sr` Hz."""
    w, file_sr = sf.read(path, dtype="float32", always_2d=True)
    w = torch.from_numpy(w.mean(1))
    if file_sr != sr:
        import torchaudio

        w = torchaudio.functional.resample(w, file_sr, sr)
    return w[: int(max_sec * sr)] if max_sec else w


def load_crop(path, secs, random_crop, rng=random):
    """A `secs`-long crop: random position (training) or centred (evaluation); short files are wrap-padded."""
    n_out = int(secs * SR)
    info = sf.info(path)
    if info.samplerate == SR and info.frames >= n_out:  # fast path: read only the crop
        start = rng.randint(0, info.frames - n_out) if random_crop else (info.frames - n_out) // 2
        w, _ = sf.read(path, start=start, frames=n_out, dtype="float32", always_2d=True)
        return torch.from_numpy(w.mean(1))
    w = load_wav(path)
    if len(w) >= n_out:
        start = rng.randint(0, len(w) - n_out) if random_crop else (len(w) - n_out) // 2
        return w[start:start + n_out]
    return w.repeat(math.ceil(n_out / len(w)))[:n_out]


# --------------------------------------------------------------------------- #
# P x K batches
# --------------------------------------------------------------------------- #
class PKDataset(torch.utils.data.Dataset):
    """Item j = K random crops of speaker j (K distinct files when the speaker has >= K files)."""

    def __init__(self, index, K, crop):
        self.spk, self.index, self.K, self.crop = sorted(index), index, K, crop

    def __len__(self):
        return len(self.spk)

    def __getitem__(self, j):
        files = self.index[self.spk[j]]
        pick = random.sample(files, self.K) if len(files) >= self.K else random.choices(files, k=self.K)
        return torch.stack([load_crop(f, self.crop, True) for f in pick]), j


class DistinctSpeakerBatchSampler(torch.utils.data.Sampler):
    """Yields `n_batches` lists of P DISTINCT speaker indices. Deterministic given (seed, epoch)."""

    def __init__(self, n_speakers, P, n_batches, seed=0):
        if P > n_speakers:
            raise ValueError(f"P={P} speakers per batch but only {n_speakers} speakers available")
        self.n, self.P, self.n_batches, self.seed, self.epoch = n_speakers, P, n_batches, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = random.Random(self.seed * 100003 + self.epoch)
        for _ in range(self.n_batches):
            yield rng.sample(range(self.n), self.P)
