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
| **Ptosis grade** | Shortfall from normal MRD1 (default 4.5 mm): mild ≤ 2, moderate ≤ 3.5, severe > 3.5 mm |
| **LF grade** | Excellent ≥ 12, Good 8–11, Fair 5–7, Poor ≤ 4 mm |

All vertical distances are measured perpendicular to the inter-pupillary line, so a slight head tilt does not inflate the values.

You can also record the phenylephrine test, Bell's phenomenon, jaw-winking and fatigability. The interpretation section uses them for teaching pointers (frontalis sling, levator resection, levator advancement or MMCR, Hering's dependence, myasthenia work-up).

## Calibration (pixels → mm)

* **HVID (default):** uses the corneal diameter (white-to-white), 11.7 mm for an average adult. Enter the measured value if you have it. Use a smaller value for children.
* **Ruler or sticker:** stick a mm scale or a sticker of known size on the forehead. In Review, choose *Ruler* and drag the two ◆ markers onto its ends.
* The down-gaze and up-gaze captures are scaled by the intercanthal distance from the primary capture. Vertical gaze changes the apparent shape of the iris, so the iris is not used to scale those captures.

## Using it

1. Open the site on the phone over **HTTPS** (the camera needs a secure context).
2. **Rear camera + torch (recommended):** the examiner holds the phone 25–30 cm away at the patient's eye level, and the patient looks at the torch.
   If the torch is not available (for example on iOS Safari), use a pen-torch held beside the lens, or the front camera with the **Ring light** button, which turns the screen white.
3. Keep the forehead, both eyes and the nose in the frame. Wait for the checks (distance, level, facing, reflex) to turn green, then press **Capture**.
   The app copes with a phone held sideways or upside-down and with tight close-ups. If it still finds no face, press **Capture** anyway: the markers start in default positions for you to drag into place. The two **I** markers go on the nasal and temporal limbus; they set the mm scale.
4. In **Review**, drag markers to correct them. A loupe magnifies the area while you drag, and the arrow pad nudges the selected marker by about 0.1 mm.
5. For levator function, open **Down-gaze** and capture, then **Up-gaze** and capture. Fix the brow with your thumb for both.
6. Copy, share or save the report, or save an annotated image. History is kept only on this phone.

You can also analyse an existing photo with **Analyse a photo** or **Upload photo**. A flash photo gives a usable corneal reflex.

### Marker colours

`R` reflex (yellow) · `U` upper lid (cyan) · `L` lower lid (green) · `C` crease (magenta) · `B` brow (orange) · `M`/`T` medial/lateral canthus

## Deploy (GitHub Pages)

Settings → Pages → *Deploy from a branch* → choose the branch and `/ (root)`. Open the Pages URL on the phone. To install it like an app, use *Add to Home screen*.

To run it locally: `python3 -m http.server 8000`, then open `http://localhost:8000`. Browsers allow the camera on localhost. On another device you need HTTPS, for example through a tunnel.

## Files

```
index.html            UI (home, live AR camera, review/editor)
css/style.css         Mobile-first styles
js/app.js             Camera, AR overlay, capture, marker editor, report, history
js/analysis.js        Landmark geometry, reflex/crease detection, measurements, grading
```

## Accuracy notes and limitations

* The results are photogrammetric estimates. The camera should be at eye level and the patient's gaze in the primary position. A camera above or below the eyes changes the MRD values.
* Corneal size varies (about ±0.5 mm), and the HVID scale error passes straight into every measurement. Use the ruler mode when you need precise values.
* Landmark lid margins can sit slightly inside the lash line. Always check the U and L markers in Review.
* If the lid covers the pupil and no reflex is visible, MRD1 is estimated from the iris centre and marked *reflex est.* In that case, lift the lid and measure clinically.
* This is a decision-support and teaching tool. It is not a certified medical device. Confirm values clinically before surgical planning.
