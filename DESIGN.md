# Design direction

Source: the owner's reference spec (a warm-dark, iOS-style profile screen with liquid-glass controls and staggered entry motion), applied to this clinical tool. Coffee content from that spec is not used; only its visual language.

Dial: ENERGY 2 / RHYTHM 2 / MOTION 2

## Palette

| Token | Value | Reason |
|---|---|---|
| `--bg` | `#180a06` | Owner's base. A dark screen also keeps the phone from throwing a second bright reflection onto the patient's cornea. |
| `--card` | `rgba(255,255,255,0.06)` | Owner's card surface; separates groups without borders. |
| `--text` | `#ede4d8` | Owner's warm off-white; softer than pure white in a dim exam room. |
| `--muted` | `rgba(235,220,205,0.62)` | Owner's muted tone, raised from 0.55 to keep 4.5:1 contrast on card surfaces. |
| `--accent` | `#f0a35a` (torch amber) | The one accent. It stands for the light that makes the corneal reflex, so it appears only where the light or the main action is: the primary button, the shutter ring, focus rings and the reflex glint in the mark. |

Status colours (ok / warn / bad) and the marker colours on photos (reflex, lids, crease, brow, canthi, limbus) are data encodings, not palette. They stay distinct so markers read on any skin tone.

## Type

System stack: `-apple-system, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Inter, "Segoe UI", Roboto, sans-serif`. Reason: the owner's direction is iOS-native; the app is used on iPhones in clinic, and a system face needs no download. Numbers use tabular figures so columns of mm values line up. Labels are sentence case at 13px, weight 500, never wide-tracked capitals.

## Shape

- Cards 24px radius (owner spec). Inputs and segmented controls 14px. Primary action is a 54px pill. Floating controls are 58 × 44 glass circles (owner spec). Radius steps mark hierarchy: content (24), fields (14), actions (pill).

## Glass

Glass (`.glass`, with the liquid displacement filter where the browser supports it) is used only on controls that float over imagery: the camera view and the photo editor. Reason: they must stay legible over a live image without hiding it. Everything else is a solid surface.

## Motion

- Home entry (first load only): the eye mark reveals, then identity, then the action, then data, staggered 0.5 to 1.0 s with `cubic-bezier(0.16, 1, 0.3, 1)`. Purpose: sets the reading order.
- The mark draws its lid line and MRD1 bracket once. Purpose: shows what the app measures.
- Screen changes slide up 12px and fade in. Purpose: continuity between the three steps (set up, capture, review).
- Loading shows a pupil constricting; it stops when loading ends. Purpose: progress feedback.
- No endless decorative loops. Everything is disabled under `prefers-reduced-motion`.

## Identity motif

The corneal light reflex: a ring with a small bright dot. It appears in the app mark, the shutter and the loading state.
