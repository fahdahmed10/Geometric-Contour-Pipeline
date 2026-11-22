# Screenshots Index

All screenshots in this directory were captured directly from the running **Contour Lab** application (`FastAPI` backend + `React`/`Vite` frontend) against the real project sample assets in `docs/images/`.

| File | Viewport / Scale | Module / View | Description |
| --- | --- | --- | --- |
| [`01-hero.png`](./01-hero.png) | `1440×900` (`1x`) | Workspace / Initial | Clean initial dark-theme state before image upload (`01 Canny Edges`, empty slots, History `0`). |
| [`02-canny-input.png`](./02-canny-input.png) | `353×780` (`@2x`, `706×1560`) | Canny / Input Slot | Original `736×978` bicycle photo (`circle_detection.jpg`) loaded in the interactive input slot. |
| [`03-canny-sidebar.png`](./03-canny-sidebar.png) | `330×848` (`@2x`, `660×1696`) | Canny / Sidebar | Canny run action, Slot 2 target selector, sliders (`7, 2, 80, 200`), and History (`3`). |
| [`04-canny-io.png`](./04-canny-io.png) | `1440×900` (`1x`) | Canny / I/O Slots | Dual-slot Canny edge comparison: Output 1 (`sigma=1`) vs Output 2 (`sigma=2`). |
| [`05-canny-stages.png`](./05-canny-stages.png) | `1440×900` (`1x`) | Canny / Stages | Six intermediate pipeline maps (`Grayscale` through `Edges`) for Output 1. |
| [`06-canny-compare.png`](./06-canny-compare.png) | `1440×900` (`1x`) | Canny / OpenCV Check | Side-by-side comparison of custom Canny and `cv2.Canny` using identical blur and thresholds. |
| [`07-hough-io.png`](./07-hough-io.png) | `1440×900` (`1x`) | Hough / I/O Slots | Dual-slot Hough line detection on `736×981` skylight (`12` lines / `150` votes vs `10` lines / `100` votes). |
| [`08-hough-stages.png`](./08-hough-stages.png) | `1440×900` (`1x`) | Hough / Stages | Hough 3-stage pipeline: `Input Edges`, polar `(rho, theta)` `Accumulator`, and `Line Overlay`. |
| [`09-hough-compare.png`](./09-hough-compare.png) | `1440×900` (`1x`) | Hough / OpenCV Check | Custom accumulator peaks (red) vs OpenCV `cv2.HoughLines` (blue) on Output 1. |
| [`10-ellipse-io.png`](./10-ellipse-io.png) | `1440×900` (`1x`) | Ellipse / I/O Slots | Concentric ellipses (`388×288`) comparing `a_min=10px` (Output 1) vs `a_min=55px` (Output 2). |
| [`11-ellipse-stages.png`](./11-ellipse-stages.png) | `1440×900` (`1x`) | Ellipse / Stages | Ellipse `Input Edges` and `Ellipse Overlay` stages with full parameter footer. |
| [`12-ellipse-compare.png`](./12-ellipse-compare.png) | `1440×900` (`1x`) | Ellipse / OpenCV Check | Custom direct conic + RANSAC ellipse fit vs OpenCV `cv2.fitEllipse`. |
| [`13-circle-input.png`](./13-circle-input.png) | `353×780` (`@2x`, `706×1560`) | Circle / Diameter Guides | Interactive `min ø 215px` and `max ø 250px` guides overlaid on bicycle Canny edges. |
| [`14-circle-sidebar.png`](./14-circle-sidebar.png) | `330×848` (`@2x`, `660×1696`) | Circle / Sidebar | Circular Hough Transform controls (`Min ø 215px`, `Max ø 250px`, `Min votes 250`, `Coverage 0.8`, `NMS win 21`). |
| [`15-circle-io.png`](./15-circle-io.png) | `1440×900` (`1x`) | Circle / I/O Slots | Dual-slot circle detection: Output 1 (`Circular Hough`, `2 CIRCLES`) vs Output 2 (`RANSAC`, `2 CIRCLES`). |
| [`16-circle-stages.png`](./16-circle-stages.png) | `1440×900` (`1x`) | Circle / Stages | CHT 3-stage pipeline showing `Input Edges`, dual-peak wheel hub `Accumulator`, and `Circle Overlay`. |
| [`17-circle-compare.png`](./17-circle-compare.png) | `1440×900` (`1x`) | Circle / OpenCV Check | Linked `HoughCircles` (`2 CIRCLES`) vs Custom Circular Hough (`2 CIRCLES`) with synchronized parameter pills. |
| [`18-snake-input.png`](./18-snake-input.png) | `353×780` (`@2x`, `706×1560`) | Snake / Input Contour | Interactive 24-point closed polygon placed around the skylight (`ellipse_detectio_noise.jpg`). |
| [`19-snake-sidebar.png`](./19-snake-sidebar.png) | `330×848` (`@2x`, `660×1696`) | Snake / Sidebar | Quick Init controls, Snake energy sliders (`Alpha 1`, `Beta 1`, `Gamma 1.2`, `Window 5`), and History (`8`). |
| [`20-snake-io.png`](./20-snake-io.png) | `1440×900` (`1x`) | Snake / I/O Slots | Initial (amber) vs converged (mint) contour overlay with live metrics strip (`Iter: 11`, `Perim: 1872.2`, `Area: 219412`, `Pick's: Verified`). |
| [`21-snake-evolution.png`](./21-snake-evolution.png) | `1440×900` (`1x`) | Snake / Evolution & Timing | Iteration playback (`Frame 11 / 11`) beside the 10-stage wall-clock breakdown (`5.05 s` total). |
| [`22-snake-metrics.png`](./22-snake-metrics.png) | `1440×900` (`1x`) | Snake / Chain Code & Metrics | Six metric cards (`1872.24 px`, `219412 px²`, `220229 px²`, `1632 px`, `Verified ✓`, `11 converged`) and 1,632-step Freeman chain code. |
| [`23-light-theme.png`](./23-light-theme.png) | `1440×900` (`1x`) | Theme & History | Light theme (`html.light`) workspace with `12` persisted IndexedDB history entries. |
| [`24-mobile.png`](./24-mobile.png) | `390×844` (`@2x`, `780×1688`) | Responsive Mobile | Stacked single-column mobile layout (`scrollWidth = 390 px`, zero horizontal overflow). |
