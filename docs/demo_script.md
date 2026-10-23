# Demo script (about 2 minutes)

Numbers for the talking points come from the generated summaries; open them before presenting
and read the values from there rather than from memory:
[`results/sim_summary.md`](../results/sim_summary.md) and
[`results/m5_summary.md`](../results/m5_summary.md).

## Before the audience arrives (5 minutes)

1. Terminal: `cd ~/Desktop/PCVGaze && make check` (camera opens, models load).
2. Close other camera apps. Sit about 50–60 cm from the screen, face evenly lit.
3. Open `results/figures/sim_rho_line_vs_pitch.png`, `sim_rho_para_vs_para_height.png` and
   `m5_error_vs_k.png` in Preview, ready to switch to.
4. Terminal: `make app` (the browser opens; leave it on the intro page).

## Live part (about 70 s)

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the intro page | "A normal laptop webcam, no eye tracker. Nothing is recorded: only gaze numbers are logged." |
| 0:10 | **Enter**, follow the 9 white dots, then the 5 green ones | "Nine dots fit a small regression from face/iris features to the screen. The five green dots are held out to *measure* the error." |
| 0:35 | Point at the HUD (σ in px) and look around: the red cursor follows | "This σ is today's measured error. It's tens of pixels to a few hundred, which is why we never claim word-level." |
| 0:45 | **Space**, read the first page normally, **Space** through the rest | "The page is paged, not scrolled, so every line has a known position." |
| 1:05 | Heatmap appears: show **P** (paragraph), then **S** (scanpath, coarse) | "The app chose paragraph level by itself, because σ is larger than half the line spacing." |

## Results part (about 50 s)

| Time | Show | Say |
|---|---|---|
| 1:10 | `sim_rho_line_vs_pitch.png` | "Simulation through our own analysis code: at our measured error, line-level heatmaps don't become reliable at any line spacing that fits the screen." (read the threshold row from `sim_summary.md`) |
| 1:25 | `sim_rho_para_vs_para_height.png` | "Paragraph-level works once paragraphs have about six lines: that's our design guideline." |
| 1:40 | `m5_error_vs_k.png` | "On 15 people from MPIIFaceGaze, held out one at a time: error in millimetres versus number of calibration points, landmark method vs L2CS-Net." (read the k = 9 row from `m5_summary.md`) |
| 1:55 | Close | "Webcam gaze is coarse but measurable; designed for paragraphs, it gives a usable attention map." |

## If something goes wrong

- **Camera doesn't open:** run from Terminal (not an IDE), check System Settings → Privacy &
  Security → Camera, restart Terminal.
- **Calibration fails ("only N dots…")** or σ is very large: press **C** to recalibrate, sitting
  still and a bit further back.
- **No camera at all:** `make app SOURCE=synthetic` shows the full flow with fake gaze (the HUD
  says SYNTHETIC). Say so out loud; it's not a measurement.
