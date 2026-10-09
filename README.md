# Ptosis Evaluator

A browser-based AR tool for measuring eyelid position with a phone camera. The phone torch creates the corneal light reflex. Face landmarks (MediaPipe Face Landmarker, 478 points including the iris) place the measurement markers, and the clinician can correct any marker before recording.

There is no install and no build step. Everything runs on the device and no images are uploaded.

## Parameters

| Parameter | How it is measured |
|---|---|
| **MRD1** | Corneal light reflex → upper lid margin |
| **MRD2** | Corneal light reflex → lower lid margin |
| **Palpebral fissure height** | MRD1 + MRD2 |
| **Palpebral fissure width** | Medial → lateral canthus |
| **Lid crease height (MCD)** | Upper lid margin → crease (found automatically from the skin-fold shadow, then confirm it) |
| **Reflex → brow** | Reflex → lower brow margin |
| **Corneal coverage** | Iris radius − MRD1 (how much cornea the upper lid covers) |
| **Inferior scleral show** | MRD2 − iris radius |
| **Levator function** | Upper-lid excursion from down-gaze to up-gaze (two captures, brow fixed) |
| **Ptosis grade** | Ptosis only when MRD1 ≤ 2.5 mm or the lid is ≥ 2 mm lower than the fellow eye; MRD1 2.5–3.5 mm is reported as low-normal. Severity is the shortfall from normal MRD1 (default 4.5 mm) or the fellow eye: mild ≤ 2, moderate ≤ 3.5, severe > 3.5 mm |
| **LF grade** | Excellent ≥ 12, Good 8–11, Fair 5–7, Poor ≤ 4 mm |

All vertical distances are measured perpendicular to the inter-pupillary line, so a slight head tilt does not inflate the values.

You can also record the phenylephrine test, Bell's phenomenon, jaw-winking and fatigability. The interpretation section uses them for teaching pointers (frontalis sling, levator resection, levator advancement or MMCR, Hering's dependence, myasthenia work-up).

## Calibration (pixels → mm)

* **HVID (default):** the limbus (iris–sclera edge) is detected automatically in each eye by scanning horizontal rows across the lower half of the cornea, and the I markers are placed on it (they are draggable). The mean of the two eyes is scaled to the HVID value, 11.7 mm for an average adult. Enter the measured value if you have it, and use a smaller value for children. A photo alone cannot give the absolute corneal size; use the ruler mode to measure HVID in mm.
* **Ruler or sticker:** stick a mm scale or a sticker of known size on the forehead. In Review, choose *Ruler* and drag the two ◆ markers onto its ends.
* The down-gaze and up-gaze captures are scaled by the intercanthal distance from the primary capture. Vertical gaze changes the apparent shape of the iris, so the iris is not used to scale those captures.

## Using it

1. Open the site on the phone over **HTTPS** (the camera needs a secure context).
2. **Rear camera + torch (recommended):** the examiner holds the phone 25–30 cm away at the patient's eye level, and the patient looks at the torch.
   If the torch is not available (for example on iOS Safari), use a pen-torch held beside the lens, or the front camera with the **Ring light** button, which turns the screen white.
3. Keep the forehead, both eyes and the nose in the frame. The live view shows only an AR alignment guide (eye brackets, iris rings, reflex dots, level line). When the checks (distance, level, facing, reflex) stay green, the app takes the photo automatically. It takes a short burst and keeps the frame with the eyes most open, so a blink is never measured. Turn **Auto** off to use the shutter button instead; tap the preview to toggle eye zoom.
   The app copes with a phone held sideways or upside-down and with tight close-ups. If it still finds no face, press the shutter anyway: the markers start in default positions for you to drag into place. The two **I** markers go on the nasal and temporal limbus; they set the mm scale.
4. In **Review**, pinch (or use + / −) to zoom the photo up to 10× and drag empty space to pan. Drag markers to correct them; a loupe magnifies the area while you drag, and the arrow pad nudges the selected marker by about 0.1 mm. On the camera screen, pinch to zoom the camera.
5. For levator function, open **Down-gaze** and capture, then **Up-gaze** and capture. Fix the brow with your thumb for both.
6. Copy, share or save the report, or save an annotated image. History is kept only on this phone.

You can also analyse an existing photo with **Analyse a photo** or **Upload photo**. A flash photo gives a usable corneal reflex.

### Marker colours

`R` reflex (yellow) · `U` upper lid (cyan) · `L` lower lid (green) · `C` crease (magenta) · `B` brow (orange) · `M`/`T` medial/lateral canthus

## How the lid margins and limbus are found

The face model (MediaPipe) only gives starting guesses; its lid points are often 1 to 2 mm off. Each captured photo then goes through an image-based detector (`js/eyeseg.js`):

1. Each eye is resampled into an upright patch at a fixed scale (iris radius 32 px), so detection works the same at any camera distance.
2. **Pupil and limbus:** Daugman's integro-differential operator. It finds the circle where the mean brightness along the side arcs (which the lids rarely cover) jumps most from dark iris to bright sclera. A clear pupil anchors the search, since the limbus is near-concentric with it. If the two eyes' corneal diameters differ by more than 10%, the clearer fit sets the size.
3. **Lower lid, then upper lid:** a dynamic-programming path across the eye opening follows the margin edge (globe to lid for the lower lid, lashes/lid to globe for the upper lid), stays smooth, and ignores the reflex and the inside of the pupil. A robust quadratic fit gives the margin height exactly at the reflex column. The upper margin must stay above the lower one, and above the reflex when the reflex is visible.
4. Where the detector is not confident, the face-model position is kept and Review shows "check U" or "check L". The traced edges are drawn as dotted lines in Review.

### Benchmark

`node test/bench.mjs 200` renders synthetic eyes with known truth (MRD1 −1 to 5.5 mm, four iris colours, five skin tones, lashes, blur, noise, head roll) and compares the face model alone with the detector. See the latest numbers in the commit history. Synthetic eyes check the method, not clinical accuracy; use the clinical-comparison panel for that.

## Learning from your corrections

Every capture keeps the detector's original (raw) marker positions. When you press **Save & learn**, the app records, for each eye, how far you moved the U, L and C markers (mm along the vertical axis) and how much you resized the limbus (I markers). Once 5 eyes are recorded, new captures are pre-corrected by the median of the last 40 corrections, shrunk towards zero while there are few samples (factor n / (n + 5)). The applied correction is shown under Results and in the report.

* Corrections are always measured against the raw detector output, so they don't compound.
* Crease corrections are learned only where the crease was detected, or where you moved the marker.
* Turn learning off, or reset it, under **Accuracy & learning** on the home screen.

## Validation against clinical measurements

In Review, enter your own ruler or slit-lamp MRD1, MRD2, PFH, MCD and LF (all optional) before saving. **Accuracy & learning** then shows, per parameter, n, bias (mean app − clinical), 95% limits of agreement, mean absolute error and % within ±1 mm, plus a Bland–Altman plot for MRD1. **Export CSV** gives one row per eye for your own analysis.

All learning and validation data is stored on the phone only (browser local storage). Clearing Safari website data deletes it, so export the CSV regularly.

## Deploy (GitHub Pages)

Settings → Pages → *Deploy from a branch* → choose the branch and `/ (root)`. Open the Pages URL on the phone. To install it like an app, use *Add to Home screen*.

To run it locally: `python3 -m http.server 8000`, then open `http://localhost:8000`. Browsers allow the camera on localhost. On another device you need HTTPS, for example through a tunnel.

## Files

```
DESIGN.md             Visual direction (palette, type, glass, motion) with the reason for each choice
index.html            UI (home, live AR camera, review/editor)
css/style.css         Mobile-first styles, entry and screen-change motion (off under reduced motion)
js/glass.js           Liquid-glass refraction for controls floating over the camera image or photo
js/app.js             Camera, AR overlay, capture, marker editor, report, history
js/analysis.js        Landmark geometry, reflex/crease/limbus detection, measurements, grading
js/eyeseg.js          Image-based limbus and lid-margin detector (pure JS, also runs in Node)
test/                 Synthetic-eye renderer and benchmark (node test/bench.mjs)
js/learn.js           On-device learning from corrections; validation stats and CSV export
```

## Accuracy notes and limitations

* The results are photogrammetric estimates. The camera should be at eye level and the patient's gaze in the primary position. A camera above or below the eyes changes the MRD values.
* Corneal size varies (about ±0.5 mm), and the HVID scale error passes straight into every measurement. Use the ruler mode when you need precise values.
* The lid detector can still be misled by heavy lashes over the cornea, mascara, a lid covering the pupil, or a blurred photo. Always check the U and L markers in Review, zoomed in.
* If the lid covers the pupil and no reflex is visible, MRD1 is estimated from the iris centre and marked *reflex est.* In that case, lift the lid and measure clinically.
* This is a decision-support and teaching tool. It is not a certified medical device. Confirm values clinically before surgical planning.
