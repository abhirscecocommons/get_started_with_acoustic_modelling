# Getting Started with Acoustic Ecology & Passive Acoustic Monitoring

A beginner-friendly workflow for ecologists to acquire, quality-check, visualise, and classify wildlife sound recordings using open Python tools and freely available Australian bird call data.

---

## Table of Contents

1. [What is Passive Acoustic Monitoring?](#1-what-is-passive-acoustic-monitoring)
2. [Audio Signal Basics](#2-audio-signal-basics)
3. [From Waveform to Spectrogram](#3-from-waveform-to-spectrogram)
4. [Mel-Spectrograms — How Machines Hear](#4-mel-spectrograms--how-machines-hear)
5. [Signal-to-Noise Ratio in Ecological Audio](#5-signal-to-noise-ratio-in-ecological-audio)
6. [BirdNET — Deep Learning for Bird Sound ID](#6-birdnet--deep-learning-for-bird-sound-id)
7. [Confidence Scores and Thresholds](#7-confidence-scores-and-thresholds)
8. [Species Occurrence Records](#8-species-occurrence-records)
9. [Australian Bioacoustics Context](#9-australian-bioacoustics-context)
10. [Data Sources](#10-data-sources)
11. [Setup and Installation](#11-setup-and-installation)
12. [Running the Notebook](#12-running-the-notebook)
13. [Troubleshooting](#13-troubleshooting)
14. [Project Structure](#14-project-structure)
15. [References](#15-references)

---

## 1. What is Passive Acoustic Monitoring?

**Passive Acoustic Monitoring (PAM)** is the practice of deploying autonomous recording units (ARUs) in the field to continuously capture soundscapes without human presence. Unlike traditional point-count surveys, PAM:

- Does **not disturb wildlife** — the device listens without an observer nearby
- Operates **24 hours a day**, capturing nocturnal and crepuscular activity
- Produces a **permanent, verifiable record** — recordings can be re-analysed as classifiers improve
- Scales across **large areas and long time periods** at low marginal cost once deployed

### The PAM Workflow

```
Deploy ARU  →  Collect SD cards  →  Upload recordings  →  QC  →  Classify  →  Ecology
   (field)          (field)            (cloud/local)    (compute)  (AI/human)   (analysis)
```

### What can PAM detect?

| Taxon | Signal type |
|-------|-------------|
| Birds | Calls, songs, wingbeats |
| Bats | Echolocation ultrasound (needs >96 kHz sample rate) |
| Frogs | Advertisement calls (most active at night) |
| Insects | Stridulation (crickets, cicadas) |
| Marine mammals | Vocalisations, clicks |
| Soundscape | Biophony, geophony, anthrophony (acoustic indices) |

### ARU Hardware

Common devices used in Australian ecological surveys:

- **Wildlife Acoustics Song Meter** (SM4, SM Mini) — industry standard for birds
- **Open Acoustics Devices AudioMoth** — open-source, cheap (~AU$90), popular for rapid deployment
- **Frontier Labs BAR-LT** — Australian-made, durable, designed for tropical conditions
- **Cornell Lab Swift** — solar-powered, long-duration deployments

---

## 2. Audio Signal Basics

### Digital Audio Fundamentals

Sound is a physical pressure wave. A microphone converts it to a continuous voltage, and an **analogue-to-digital converter (ADC)** turns that into a sequence of numbers stored in a WAV or FLAC file.

#### Sample Rate

The **sample rate** (e.g. 44,100 Hz) is how many amplitude measurements are taken per second.

**Nyquist–Shannon sampling theorem:**

> To faithfully represent a signal containing frequencies up to *f* Hz, you must sample at *at least* 2*f* Hz.

| Application | Typical sample rate | Max frequency captured |
|-------------|--------------------|-----------------------|
| Bird surveys | 22,050–48,000 Hz | 11,025–24,000 Hz |
| Bat echolocation | 192,000–384,000 Hz | 96,000–192,000 Hz |
| Human speech | 8,000–16,000 Hz | 4,000–8,000 Hz |
| CD audio | 44,100 Hz | 22,050 Hz |

BirdNET operates at **48,000 Hz** (48 kHz), so recordings are resampled to this rate before classification.

#### Bit Depth

The **bit depth** determines how many discrete amplitude values are possible:

- **16-bit**: 65,536 levels — standard for most field recorders (~96 dB dynamic range)
- **24-bit**: 16,777,216 levels — used when high dynamic range is needed

#### Mono vs Stereo

Most bird call analysis uses **mono** (single channel). Stereo recordings are down-mixed before analysis. Some researchers use stereo or multi-channel arrays to estimate sound direction.

---

## 3. From Waveform to Spectrogram

### The Waveform

The raw audio is a one-dimensional array of amplitude values over time — this is the **waveform**. It is the direct output of the ADC.

```
Amplitude
   |  /\      /\  /\
   | /  \    /  \/  \
   |/    \  /         ...
   +------\/----------> Time
```

The waveform shows *when* energy is present but doesn't easily reveal *which frequencies* are present.

### Short-Time Fourier Transform (STFT)

The **Fourier Transform** decomposes a signal into its constituent frequencies. A bird call at 4000 Hz will appear as a spike at 4000 Hz in the frequency domain.

The problem: a plain Fourier Transform throws away all time information. We can't tell *when* the bird called.

The **Short-Time Fourier Transform (STFT)** solves this by:

1. Dividing the signal into short overlapping **windows** (e.g. 25 ms each)
2. Computing the Fourier Transform of each window
3. Stacking the results into a 2D matrix: **time × frequency**

```
STFT parameters (librosa defaults):
  n_fft      = 2048 samples  (window size, ~46 ms at 44.1 kHz)
  hop_length = 512 samples   (step between windows, ~12 ms)
  window     = Hann
```

The result is a **complex matrix**. Taking the magnitude and (usually) converting to decibels gives the **power spectrogram**:

```
S_dB[f, t] = 10 * log10( |STFT[f, t]|² )
```

### Reading a Spectrogram

```
Frequency (Hz)
  ^
  |         ___
  |        /   \    <-- bird call (energy at ~4000 Hz from t=1 to t=2)
  |       /     \
  |______/       \________
  +----------------------------> Time (s)
```

- **Horizontal axis**: time
- **Vertical axis**: frequency
- **Colour**: intensity (bright = loud, dark = quiet)
- **Colour maps**: `magma` (dark background) shows faint signals well; `viridis` is perceptually uniform

---

## 4. Mel-Spectrograms — How Machines Hear

### The Mel Scale

Human hearing is **not linear in frequency**. We are better at distinguishing pitches at low frequencies than at high frequencies. The **mel scale** is a psychoacoustic scale that approximates human pitch perception:

```
mel = 2595 * log10(1 + Hz/700)
```

In practical terms:
- At low frequencies (e.g. 500 Hz → 2000 Hz), each mel step covers a small frequency range
- At high frequencies (e.g. 5000 Hz → 10000 Hz), each mel step covers a large frequency range

### Mel Filter Banks

A **mel filterbank** consists of triangular filters spaced evenly on the mel scale. Each filter averages the energy in a mel-frequency band. This converts the STFT (which has linear frequency bins) into a compact representation that emphasises the frequency ranges most important for sound recognition.

```
STFT (linear bins, e.g. 1025 bins)  →  Mel filterbank (128 mel bins)
```

### Why CNNs Use Mel-Spectrograms

Deep learning models like BirdNET treat the mel-spectrogram as a **grayscale image** and apply Convolutional Neural Networks (CNNs) — the same architecture used for photo classification. This works because:

1. **Translation invariance**: a CNN recognises a pattern (e.g. a frequency sweep) regardless of where it appears in the image
2. **Dimensionality reduction**: 128 mel bins capture the important structure while discarding uninformative high-frequency detail
3. **Biological motivation**: the mel scale mimics the cochlea's frequency resolution

---

## 5. Signal-to-Noise Ratio in Ecological Audio

### What is SNR?

**Signal-to-Noise Ratio (SNR)** measures how much louder the target signal is compared to background noise.

```
SNR (dB) = 10 * log10( P_signal / P_noise )
```

A **higher SNR** means the signal is clearer relative to noise. A **negative SNR** means the noise drowns out the signal.

### Why SNR Matters for PAM

Field recordings are contaminated by many noise sources:

| Noise source | Category | Example |
|---|---|---|
| Wind | Geophony | Low-frequency rumble in open sites |
| Rain | Geophony | Broadband crackling, especially in tropics |
| Insects | Biophony | Cicada choruses (can mask birds entirely) |
| Traffic | Anthrophony | Low-frequency hum near roads |
| Aircraft | Anthrophony | Broadband sweep |
| Electrical hum | Anthrophony | 50/60 Hz tone + harmonics |

### SNR Estimation (RMS Percentile Method)

The notebook estimates SNR using RMS energy percentiles:

```python
rms = librosa.feature.rms(y=y)[0]
signal_rms = np.percentile(rms, 95)   # loud frames ≈ signal
noise_rms  = np.percentile(rms, 10)   # quiet frames ≈ noise
snr_db = 20 * np.log10(signal_rms / (noise_rms + 1e-9))
```

**Interpretation:**

| SNR (dB) | Quality | Notes |
|----------|---------|-------|
| > 20 | Excellent | Clear signal, confident detection |
| 10–20 | Good | Most classifiers work well |
| 0–10 | Poor | Marginal, some false negatives |
| < 0 | Very poor | Noise-dominated, unreliable |

---

## 6. BirdNET — Deep Learning for Bird Sound ID

### What is BirdNET?

**BirdNET** is a deep neural network developed by the Cornell Lab of Ornithology and Chemnitz University of Technology (Kahl et al., 2021). It is trained to identify bird species from audio clips.

- **Coverage**: 6,522 bird species (as of v2.4)
- **Architecture**: EfficientNet-based CNN (an image classification architecture applied to mel-spectrograms)
- **Input**: 3-second audio clips, 48 kHz, converted to mel-spectrogram
- **Output**: probability score (0–1) for each of the 6,522 species classes

### How BirdNET Processes Audio

```
Raw WAV (any sample rate)
       │
       ▼
  Resample to 48 kHz
       │
       ▼
  Split into 3-second windows (with 50% overlap)
       │
       ▼
  Compute mel-spectrogram
  (128 mel bins, 150 Hz – 12 kHz)
       │
       ▼
  Feed into EfficientNet CNN
       │
       ▼
  Softmax over 6,522 species classes
       │
       ▼
  Filter by confidence threshold (default: 0.1–0.25)
       │
       ▼
  Output: species name, start time, end time, confidence
```

### EfficientNet Architecture

EfficientNet scales depth, width, and resolution of the network together using a compound scaling coefficient. It achieves high accuracy with fewer parameters than earlier architectures like VGG or ResNet.

BirdNET uses a variant optimised for spectrogram images. The mel-spectrogram of a 3-second clip is treated as a 2D image (time × frequency), and the CNN learns to recognise the visual patterns corresponding to each species' vocalisation.

### Training Data

BirdNET v2.4 was trained on:

- **Xeno-canto** recordings (~500,000 recordings, ~10,000 species)
- **eBird** community science data for species lists by location
- Data augmentation: pitch shifting, time stretching, noise injection, mixup

---

## 7. Confidence Scores and Thresholds

### What is a Confidence Score?

After the softmax layer, BirdNET outputs a probability for each species. The **confidence score** is the probability assigned to the top-matching species.

```
confidence = 0.85  →  "BirdNET is 85% confident this is species X in this 3-second window"
```

**Important caveats:**

1. The score is a **relative probability**, not a calibrated ecological probability of presence
2. A high score means the audio pattern *looks like* that species' training data, not that the species is definitely there
3. Some species are easily confused (similar calls, limited training data for rare species)
4. Environmental noise can cause false positives

### Recommended Thresholds

| Use case | Minimum confidence | Notes |
|---|---|---|
| Exploratory analysis | 0.10–0.25 | More detections, more false positives |
| Occurrence records | 0.50 | Reasonable balance for most species |
| High-confidence atlas data | 0.75–0.90 | Conservative, lower recall |
| Verification required | Any | Always recommended for new records |

The notebook uses:
- `MIN_CONFIDENCE = 0.25` — for the detections table (to show everything)
- `OCCURRENCE_CONFIDENCE = 0.50` — for the species occurrence CSV export

### What to Do With Low-Confidence Detections

Low-confidence windows should be:
1. Manually reviewed in a tool like Audacity or the Acoustic Workbench
2. Cross-referenced with expected species for the location/season (eBird ranges, Atlas of Living Australia)
3. Flagged as "unverified" in your data management system

---

## 8. Species Occurrence Records

### Darwin Core Standard

The notebook exports occurrence records in a simplified **Darwin Core** format. Darwin Core is the international standard for biodiversity data exchange, maintained by TDWG (Biodiversity Information Standards).

Key fields:

| Field | Description | Example |
|---|---|---|
| `species` | Scientific name | *Pachycephala pectoralis* |
| `lat` | Decimal latitude (WGS84) | -17.1234 |
| `lon` | Decimal longitude (WGS84) | 145.6789 |
| `confidence` | BirdNET score (0–1) | 0.72 |
| `recording_id` | Source file identifier | XC924845 |
| `detected_at` | Timestamp of detection | 2024-10-15T08:32:00Z |

### Important Caveats for Ecologists

**1. Location uncertainty**

Xeno-canto recordings are geotagged at the recording *location*, not the exact *perch* of the bird. GPS accuracy, habitat structure, and observer positioning all introduce uncertainty. Treat coordinates as approximate (±100 m to ±1 km depending on the recorder's notes).

**2. Presence-only data**

PAM data tell you a species *was detected*. They do **not** tell you a species was *absent*. Occupancy modelling (e.g. with `unmarked` in R) is needed to properly account for imperfect detection.

**3. False positives**

No classifier is perfect. Always cross-check unexpected records (species outside known range, records of threatened or rare species) against:
- **Atlas of Living Australia (ALA)**: [ala.org.au](https://www.ala.org.au)
- **eBird Australia**: [ebird.org/australia](https://ebird.org/australia)
- Manual spectrogram review

**4. Temporal pseudoreplication**

Multiple detections of the same species in the same recording are likely the same individual. For occurrence modelling, consider aggregating to one detection per species per site per day.

---

## 9. Australian Bioacoustics Context

### Why Australia is Unique

Australia's soundscapes are dominated by:

- **Passerines (songbirds)**: ~45% of Australian bird species — extraordinary vocal diversity
- **Parrots (Psittaciformes)**: noisy, highly recognisable calls
- **Frogmouths and nightjars**: cryptic species, most reliably detected by call
- **Frogs**: >240 species, many detectable only by PAM in wet season
- **Insects**: cicada choruses (peak in summer) can completely mask bird calls in northern Australia

### Key Habitats and Their Soundscapes

| Habitat | Key acoustic features |
|---|---|
| Wet Tropics (Qld) | Complex multi-layered dawn chorus, high frog diversity at night, heavy rain masking |
| Dry sclerophyll | Insect-dominated in summer; quieter bird activity midday |
| Spinifex grasslands | Sparse soundscape; cryptic grassland specialists (e.g. Spinifexbird) |
| Coastal mangroves | Mangrove Gerygone, wading birds, tidal noise |
| Alpine ash forest | Post-fire recovery detectable through acoustic diversity indices |

### The Acoustic Observatory

This project can connect to the **[Acoustic Observatory](https://acousticobservatory.org/)** — a network of permanently deployed ARUs across Australia, maintained by QUT and partners. The `download_audio.py` script in this repository downloads recordings from their API at `api.acousticobservatory.org`.

Key sites include:
- **Robson Creek** (Wet Tropics, north Queensland) — Dry A and Wet A sites in upland rainforest
- Multiple sites across Queensland and other states

### Best Times to Record in Australia

| Time | Activity |
|---|---|
| 30 min before sunrise to 2 hours after | Dawn chorus — peak bird activity |
| 3–8 pm | Evening chorus (less intense than dawn but useful) |
| Night (8 pm – 2 am) | Nocturnal birds (owls, frogmouths, nightjars), frogs |
| Midday | Minimal bird activity in summer heat |

---

## 10. Data Sources

### Xeno-canto via GBIF (used in this notebook)

**Xeno-canto** ([xeno-canto.org](https://www.xeno-canto.org)) is a citizen science repository with >750,000 wildlife sound recordings from around the world. Recordings are accessible through the GBIF Occurrence API:

```
https://api.gbif.org/v1/occurrence/search
  ?datasetKey=b1047888-ae52-4179-9dd5-5448ea342a24
  &scientificName=<species>
  &country=AU
```

The GBIF response includes `media` objects with direct URLs to WAV/MP3 files, plus decimal lat/lon coordinates.

**No authentication required** for read access.

### The Acoustic Observatory API

For research-grade long-term recordings from permanent sites:

```
Base URL:  https://api.acousticobservatory.org
Auth:      POST /security  →  returns auth_token
Filter:    POST /audio_recordings/filter  (JSON body)
Download:  GET  /audio_recordings/{id}/original
```

Credentials (email + password or auth token) are required. See `download_audio.py` for usage.

Store credentials in a `.env` file (never commit this file to git):
```
ACOUSTIC_USER=your.email@institution.edu.au
ACOUSTIC_TOKEN=your_token_here
```

### Other Open Sources

| Source | Coverage | Access |
|---|---|---|
| [Macaulay Library](https://www.macaulaylibrary.org) | Global, birds + more | Free browsing, no bulk API |
| [freesound.org](https://freesound.org) | General sounds, some wildlife | Free API with key |
| [Australian Acoustic Observatory](https://acousticobservatory.org) | Australian sites, long-term | Account required |
| [QUT Ecoacoustics Research Group](https://research.ecosounds.org) | Various datasets | Contact researchers |

---

## 11. Setup and Installation

### Requirements

- Python 3.9–3.12 (recommended: **3.11** for best compatibility)
- pip (included with Python)
- ~1.5 GB disk space for dependencies (mostly TensorFlow)
- Internet connection (to download audio and model weights)

> **Note on Python version**: `tflite_runtime` (BirdNET's lightweight backend) is **not available on PyPI for Python 3.12 or 3.13**. This notebook uses full `tensorflow` instead, which works on Python 3.9–3.12. Python 3.13 is not yet supported by TensorFlow.

### Step-by-Step Installation

**1. Clone the repository**

```bash
git clone https://github.com/abhirscecocommons/getting_started_with_acoustic_modelling.git
cd getting_started_with_acoustic_modelling
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate.bat     # Windows Command Prompt
# .venv\Scripts\Activate.ps1     # Windows PowerShell
```

**3. Install dependencies**

```bash
pip install librosa soundfile resampy matplotlib pandas requests numpy birdnetlib tensorflow jupyter
```

Or install from the notebook's first cell by running it.

**4. (Optional) Set up credentials for Acoustic Observatory**

Create a `.env` file in the project directory:

```bash
echo "ACOUSTIC_USER=your.email@institution.edu.au" > .env
echo "ACOUSTIC_TOKEN=your_token_here"              >> .env
```

This file is listed in `.gitignore` and will never be committed.

**5. Launch Jupyter**

```bash
jupyter notebook getting_started_acoustic_ecology.ipynb
```

---

## 12. Running the Notebook

The notebook is divided into five sections. Run cells top to bottom. Each section can be re-run independently once data is downloaded.

| Section | What it does | Runtime |
|---|---|---|
| 0 — Data Acquisition | Downloads audio files from GBIF/Xeno-canto | 2–5 min (depends on internet speed) |
| 1 — Quality Checks | Estimates SNR, sample rate, duration | ~30 sec |
| 2 — Visualisation | Waveform + spectrogram plots | ~1 min |
| 3 — BirdNET Classification | AI classification of all recordings | 2–10 min (first run downloads ~100 MB model) |
| 4 — Occurrence Export | Exports species_occurrences.csv | < 5 sec |

### Swapping the Data Source

The key design of this notebook is that **Sections 1–4 are data-source agnostic**. They all consume a `metadata_df` DataFrame with these columns:

| Column | Type | Description |
|---|---|---|
| `recording_id` | str | Unique identifier (e.g. `XC924845`) |
| `species` | str | Scientific name |
| `local_path` | str | Absolute path to the downloaded file |
| `lat` | float | Decimal latitude of recording |
| `lon` | float | Decimal longitude of recording |
| `date` | str | Recording date (ISO 8601) |
| `country` | str | Country code (e.g. `AU`) |
| `duration_s` | float | Duration in seconds |

To use your **own recordings**, replace Section 0 with a cell that builds this DataFrame from your local files.

---

## 13. Troubleshooting

### `ModuleNotFoundError: No module named 'tflite_runtime'`

**Cause**: BirdNET first tries to import `tflite_runtime` (a lightweight TensorFlow runtime). This package is **not available on PyPI for Python 3.12 or 3.13**.

**Fix**: Install full TensorFlow:

```bash
pip install tensorflow
```

TensorFlow is large (~600 MB) but is the correct solution for modern Python versions.

---

### `ModuleNotFoundError: No module named 'tensorflow'`

**Cause**: Neither `tflite_runtime` nor `tensorflow` is installed.

**Fix**: Same as above — `pip install tensorflow`.

---

### `ModuleNotFoundError: No module named 'resampy'`

**Cause**: librosa uses `resampy` for high-quality audio resampling. It is not installed by default with librosa.

**Fix**:

```bash
pip install resampy
```

---

### `librosa.load()` fails on MP3 files

**Cause**: MP3 decoding requires `ffmpeg` to be installed on your system.

**Fix**:

```bash
# macOS (Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

---

### BirdNET returns no detections

Possible causes:

1. **Recording too short**: BirdNET needs at least 3 seconds of audio
2. **High noise**: recordings with SNR < 0 dB rarely yield detections
3. **Threshold too high**: try lowering `MIN_CONFIDENCE` to 0.1
4. **Species not in BirdNET**: check [BirdNET species list](https://github.com/kahst/BirdNET-Analyzer/blob/main/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Labels.txt)

---

### `requests.exceptions.SSLError` or `504 Gateway Timeout` (Acoustic Observatory)

The Acoustic Observatory API may be temporarily unavailable. Wait and retry, or use the `--dry-run` flag first:

```bash
python download_audio.py --dry-run
```

---

### Jupyter notebook won't run (asyncio error)

On some systems with older Jupyter:

```bash
pip install --upgrade jupyter ipykernel
```

Or run as a script:

```bash
jupyter nbconvert --to notebook --execute getting_started_acoustic_ecology.ipynb --output output.ipynb
```

---

## 14. Project Structure

```
acoustic_model/
├── getting_started_acoustic_ecology.ipynb   # Main notebook
├── download_audio.py                        # Script to download from Acoustic Observatory
├── README.md                                # This file
├── .env                                     # Credentials — NEVER commit! (in .gitignore)
├── .gitignore
├── audio_data/                              # Downloaded audio files
│   ├── XC924845.wav
│   └── ...
└── results/                                 # Generated outputs
    ├── metadata.csv
    ├── quality_checks.csv
    ├── detections.csv
    ├── species_occurrences.csv              # Main output: species, lat, lon
    ├── species_occurrences_full.csv
    ├── *_visualisation.png
    ├── species_comparison.png
    └── occurrence_map.png
```

---

## 15. References

### Core Software

- **BirdNET-Analyzer**: Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep learning solution for avian diversity monitoring. *Ecological Informatics*, 61, 101236. https://doi.org/10.1016/j.ecoinf.2021.101236

- **birdnetlib**: Python wrapper for BirdNET-Analyzer. https://github.com/joeweiss/birdnetlib

- **librosa**: McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. *Proceedings of the 14th Python in Science Conference*. https://librosa.org

- **TensorFlow**: Abadi, M., et al. (2015). TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems. https://tensorflow.org

### Data Sources

- **GBIF / Xeno-canto**: GBIF Occurrence Download. https://www.gbif.org — Xeno-canto Foundation & Naturalis Biodiversity Center (2023). Xeno-canto – Bird sounds from around the world. https://www.xeno-canto.org

- **Acoustic Observatory**: Queensland University of Technology. *Australian Acoustic Observatory*. https://acousticobservatory.org

### Acoustic Ecology & Bioacoustics

- **OpenSoundscape**: Lapp, S., et al. (2023). OpenSoundscape: An Open-Source Bioacoustics Analysis Package for Python. *Methods in Ecology and Evolution*. https://opensoundscape.org

- **BirdSet**: Rauch, L., et al. (2024). BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics. https://huggingface.co/datasets/DBD-research-group/BirdSet

- **Bird Recognition Review**: Mikoś, A., et al. (2021). Review of Automatic Bird Recognition. https://github.com/AgaMiko/bird-recognition-review

- **Passive Acoustic Monitoring review**: Sugai, L. S. M., et al. (2019). Terrestrial Passive Acoustic Monitoring: Review and Perspectives. *BioScience*, 69(1), 15–25. https://doi.org/10.1093/biosci/biy147

### Australian Biodiversity

- **Atlas of Living Australia**: https://www.ala.org.au

- **eBird Australia**: https://ebird.org/australia

- **Darwin Core standard**: Wieczorek, J., et al. (2012). Darwin Core: An Evolving Community-Developed Biodiversity Data Standard. *PLoS ONE*, 7(1), e29715. https://doi.org/10.1371/journal.pone.0029715

---

## Acknowledgements

This workflow was developed for the [EcoCommons Australia](https://www.ecocommons.org.au/) platform as an educational resource for ecologists. Audio data sourced from Xeno-canto contributors via GBIF under Creative Commons licences. Long-term monitoring data available via the Australian Acoustic Observatory (QUT).

---

*Last updated: May 2026 | Python 3.11+ | BirdNET v2.4*
