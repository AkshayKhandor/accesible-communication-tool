# MyVoice — Assistive Communication & Early Literacy

A browser-based assistive tool for non-verbal, mute, and autistic users, combining
a picture-based communication board, webcam gesture recognition, and a
handwriting-tracing module for learning letters and numbers.

Built as an end-to-end applied machine learning project: data collection →
feature engineering → model training → evaluation → **in-browser deployment**.

**[Live demo](https://akshaykhandor.github.io/accesible-communication-tool/)** ·
Runs in any modern browser, and installs to a phone or tablet home screen as an
offline-capable app.

---

## What it does

### Communicate
Sixteen large picture cards covering everyday needs (water, food, bathroom, pain,
medicine, go outside, yes/no, emotions). Tapping a card speaks a full sentence
aloud via the Web Speech API and logs the request. A side panel surfaces live
usage analytics: total requests, most-frequent need, and a frequency distribution.

### Learn Letters (A–Z, a–z) and Numbers (1–10)
A handwriting module that reveals a character one movement at a time. A toggle
switches between uppercase and lowercase letter sets.

1. Only a single arrow is shown — the very next movement, with a numbered marker
   where the stroke begins. The finished letter is never shown in advance.
2. The learner traces that arrow with a finger, mouse or stylus. Reaching its tip
   unlocks the next arrow, so the character is built stroke by stroke rather than
   copied from a finished outline.
3. Arrows follow the true curve of each stroke, so round characters (C, O, S, 8)
   are guided smoothly, and C, G, O, Q and 0 start at the top and run
   counter-clockwise, matching standard handwriting instruction.
4. On completion it names the character aloud, pauses, then offers praise.
5. The guide arrows then clear away, leaving only the learner's own writing, and
   it asks whether they want to learn the character again — repeating is offered,
   never imposed.
6. Learners choose their own pencil colour from eight options.

Nothing is timed, scored or marked wrong. This is a deliberate accessibility
decision for autistic learners.

### Body Parts
Two illustrated figures (a man and a woman) with sixteen tappable body parts.
Tapping a part highlights it with a bright outline so the learner can see
exactly which region is meant. Two modes:

- **What is this?** — tapping a part speaks its name, for vocabulary learning.
- **It hurts here** — tapping speaks "my knee hurts" and records it to the
  usage log, so a non-verbal user can report pain by pointing.

The figures are drawn in a friendly cartoon style, fully clothed and
non-anatomical: this is a naming and pain-reporting aid, so it covers only
outwardly visible everyday body parts. Clothing is chosen so that legs, knees
and feet remain visible and tappable on both figures.

### Gestures
Recognises eight one-handed signs via webcam and maps them to communication
needs, for users who cannot reliably use a touchscreen. Includes a data
collection mode for building a labelled training set, and can load a trained
model to run inference client-side.

---

## Machine learning pipeline

| Stage | Implementation | Concept |
|---|---|---|
| Perception | MediaPipe Hands extracts 21 hand landmarks per frame | Transfer learning / pretrained models |
| Feature engineering | Wrist-centred translation + hand-span scaling | Position and scale invariance |
| Baseline | Rule-based classifier over finger-extension geometry | Interpretable baseline |
| Models | K-Nearest Neighbours, Random Forest (scikit-learn) | Model selection |
| Evaluation | Accuracy, confusion matrix, per-class precision/recall, feature importance | Rigorous evaluation |
| Deployment | Model exported to JSON, inference reimplemented in JavaScript | Closing the training→serving loop |

### Feature engineering result

Raw landmark coordinates encode *where* the hand was and *how large* it appeared,
so a model trained on them partly memorises the recording setup rather than the
gesture. Normalising (translate to wrist origin, scale by hand span) fixes this.

Tested by applying a position shift and scale change to held-out samples,
simulating a different recording distance and camera placement:

| Features | KNN | Random Forest |
|---|---|---|
| Raw coordinates | 100% | **60%** |
| Normalised | 100% | **100%** |

Random Forest degrades badly on raw coordinates under realistic variation and
recovers fully once features are normalised.

### Training→serving parity

The exported model is re-implemented in JavaScript so it runs in the browser.
Verified on 150 samples: **150/150 predictions identical to scikit-learn**, for
both Random Forest and KNN. The app displays the rule-based baseline and the
trained model predicting side by side on the same frame, with a running
agreement rate.

---

## Usage

### Run the app
Open `index.html` in Chrome, Edge, or Safari. Everything runs client-side;
only Gesture Mode requires a network connection (to load the hand-tracking model)
and camera permission.

### Train a gesture model

```bash
pip3 install pandas scikit-learn joblib numpy

# single dataset (auto train/test split)
python3 train_gesture_classifier.py gesture_dataset.csv

# recommended: train on one session, test on a separate one
python3 train_gesture_classifier.py session1.csv session2.csv

# report raw-vs-normalised feature comparison
python3 train_gesture_classifier.py session1.csv session2.csv --compare
```

Collect data via the app: **Gestures → Data collection mode →** hold each pose,
click its label, then download the CSV. Aim for 30–50 samples per gesture.

The script writes `gesture_model.joblib` (Python) and `gesture_model.json`
(browser). Load the JSON via **Gestures → Use trained model**.

### A note on evaluation

Training and testing on a single recording session yields ~100% accuracy, but
this reflects memorisation of a fixed scene rather than genuine generalisation.
Collecting a second session under different lighting, distance, or camera angle
and passing both files gives an honest estimate. The gap between the two numbers
is the meaningful result.

---

## Tech stack

- Vanilla JavaScript, HTML5 Canvas, Web Speech API, Pointer Events
- MediaPipe Hands for landmark detection
- Python: scikit-learn, pandas, NumPy, joblib
- No build step, no framework, no backend

---

## Limitations

- Gesture recognition assumes a single hand, reasonable lighting, and an
  uncluttered background.
- Stroke data covers uppercase A–Z, lowercase a–z, and 0–10; no cursive.
- Tracing tolerance is a single fixed radius; curved characters (S, 6, 8) are
  objectively harder to trace than straight ones and would benefit from
  per-character tuning.
- The reported feature-engineering result uses synthetically perturbed data;
  validation on genuinely independent recording sessions is still needed.
- **No evaluation with target users has been conducted.** Any real deployment
  would require testing alongside speech-language therapists and educators.

## Ethical considerations

Usage logs and handwriting-progress data about a child are sensitive. In this
implementation all data stays in the browser session and nothing is transmitted
or persisted. Any production version would need explicit consent, clear data
retention limits, and caregiver-controlled access. This is an assistive aid
intended to complement, never replace, human support and professional therapy.

## License

MIT — see [LICENSE](LICENSE).
