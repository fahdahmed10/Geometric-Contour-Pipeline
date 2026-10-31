import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = "http://localhost:8000/api";
const initialCanny = { kernel_size: 7, sigma: 1, low_threshold: 80, high_threshold: 200 };
const initialHough = { radius_res: 1, theta_res: 1, max_lines: 10, vote_threshold: 20 };
const initialEllipse = { a_min: 10, a_max: 150, vote_threshold: 40, fit_tol: 2, min_support: 0.5, max_ellipses: 3, ransac_iters: 300 };
const initialCircle = {
  cht: { min_diam: 45, max_diam: 125, min_votes: 48, min_coverage: 0.6, perim_frac: 0.4, smooth: 3, nms_win: 7, align_cos: 0.7, refine: true },
  ransac: { min_diam: 45, max_diam: 125, tol: 2, max_iter: 1000, min_inliers: 48, perim_frac: 0.4, min_coverage: 0.6, refine: true, seed: 0 },
  opencv: { dp: 1, min_dist: 19.1, param1: 200, param2: 28.8, min_radius: 19.1, max_radius: 71.9 },
};

function Slider({ label, name, value, min, max, step = 1, onChange, suffix = "" }) {
  return <label className="control"><span>{label}<b>{value}{suffix}</b></span><input type="range" name={name} value={value} min={min} max={max} step={step} onChange={onChange} /></label>;
}

function SlotCard({ slot, value, source, onClick, selected, summary = null }) {
  return <button className={`slot-card ${value ? "filled" : ""} ${selected ? "active" : ""}`} onClick={onClick}>
    <span className="slot-index">{slot}</span><span className="slot-label">{source || "Available slot"}</span>
    {value ? <img src={value} alt={`${source} in Output ${slot}`} /> : <span className="slot-empty">Drop result here</span>}
    {summary}
  </button>;
}

function StagePipeline({ title, steps, names, selection, onSelectionChange, options }) {
  return <section className="stage-panel"><div className="section-heading"><span className="section-number">STAGE PIPELINE</span><h2>{title}</h2><label className="pipeline-select">Visualize Pipeline For:<select value={selection} onChange={(event) => onSelectionChange(event.target.value)}>{options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}</select></label></div>
    <div className="stage-grid">{names.map((name) => {
      const step = steps?.[name];
      return <figure className={step ? "stage available" : "stage"} key={name}>{step ? <img src={step.image} alt={name} /> : <div className="stage-placeholder">Awaiting run</div>}<figcaption>{name.replaceAll("_", " ")}</figcaption></figure>;
    })}</div>
  </section>;
}

function ComparisonPanel({ title, custom, opencv, metric }) {
  if (!custom || !opencv) return null;
  return <section className="comparison-panel"><div className="comparison-heading"><span className="eyebrow">REFERENCE CHECK</span><h3>{title} vs OpenCV</h3><span className="comparison-metric">{metric}</span></div><div className="comparison-grid"><figure><img src={custom} alt={`${title} custom implementation`} /><figcaption>Custom from scratch</figcaption></figure><figure><img src={opencv} alt={`${title} OpenCV implementation`} /><figcaption>OpenCV reference</figcaption></figure></div></section>;
}

function ParameterList({ parameters }) {
  if (!parameters) return null;
  return <div className="circle-parameter-list">{Object.entries(parameters).map(([name, value]) => <span key={name}><b>{name}</b><em>{typeof value === "number" ? Number(value.toFixed(3)) : String(value)}</em></span>)}</div>;
}

function CircleRunSummary({ slot, run }) {
  if (!run) return null;
  const customCount = run.circles?.length ?? 0;
  const opencvCount = run.opencv_circles?.length ?? 0;
  return <section className="circle-run-summary"><div className="circle-summary-card custom-summary"><div className="circle-summary-heading"><div><span className="eyebrow">OUTPUT {slot} · CUSTOM</span><h3>{run.algorithm === "cht" ? "Circular Hough Transform" : "RANSAC circle detection"}</h3></div><strong>{customCount}<small>circles detected</small></strong></div><ParameterList parameters={run.custom_parameters} /></div>{run.opencv_parameters && <div className="circle-summary-card opencv-summary"><div className="circle-summary-heading"><div><span className="eyebrow">OUTPUT {slot} · OPENCV</span><h3>HoughCircles reference</h3></div><strong>{opencvCount}<small>circles detected</small></strong></div><ParameterList parameters={run.opencv_parameters} /></div>}</section>;
}

function CircleSlotSummary({ run }) {
  if (!run) return null;
  return <div className="slot-circle-summary"><div className="slot-circle-count"><strong>{run.circles?.length ?? 0}</strong><span>custom circles detected</span></div><ParameterList parameters={run.custom_parameters} />{run.opencv_parameters && <div className="slot-opencv-summary"><div className="slot-circle-count"><strong>{run.opencv_circles?.length ?? 0}</strong><span>OpenCV circles detected</span></div><ParameterList parameters={run.opencv_parameters} /></div>}</div>;
}

function CircleOpenCVControls({ values, onChange }) {
  return <section className="circle-opencv-controls"><div><span className="eyebrow">OPENCV REFERENCE CONTROLS</span><h3>Comparison parameters</h3><p>These controls affect only the optional OpenCV comparison.</p></div><div className="opencv-control-grid"><Slider label="dp (accumulator resolution)" name="dp" value={values.dp} min="0.5" max="3" step="0.1" onChange={onChange} /><Slider label="Minimum centre distance" name="min_dist" value={values.min_dist} min="1" max="500" step="1" onChange={onChange} suffix=" px" /><Slider label="param1 (Canny high)" name="param1" value={values.param1} min="1" max="1000" onChange={onChange} /><Slider label="param2 (accumulator threshold)" name="param2" value={values.param2} min="1" max="1000" onChange={onChange} /><Slider label="Minimum radius" name="min_radius" value={values.min_radius} min="1" max="1000" step="1" onChange={onChange} suffix=" px" /><Slider label="Maximum radius" name="max_radius" value={values.max_radius} min="1" max="1000" step="1" onChange={onChange} suffix=" px" /></div></section>;
}

function CircleSummaryBlock({ label, title, count, parameters, className = "" }) {
  return <div className={`circle-inline-summary ${className}`}><div className="circle-inline-heading"><div><span className="eyebrow">{label}</span><h3>{title}</h3></div><strong>{count}<small>circles</small></strong></div><ParameterList parameters={parameters} /></div>;
}

function CircleComparisonDetails({ run, title }) {
  if (!run?.opencv_output) return null;
  return <section className="comparison-panel circle-comparison-details"><div className="comparison-heading"><span className="eyebrow">REFERENCE CHECK</span><h3>{title} vs OpenCV</h3><span className="comparison-metric">Same selected image and comparison settings</span></div><div className="comparison-grid"><figure><img src={run.output} alt="Custom circle detection result" /><figcaption>Custom from scratch</figcaption><CircleSummaryBlock label="CUSTOM RESULT" title={run.algorithm === "cht" ? "Circular Hough Transform" : "RANSAC circle detection"} count={run.circles?.length ?? 0} parameters={run.custom_parameters} /></figure><figure><img src={run.opencv_output} alt="OpenCV circle detection result" /><figcaption>OpenCV reference</figcaption><CircleSummaryBlock label="OPENCV RESULT" title="HoughCircles reference" count={run.opencv_circles?.length ?? 0} parameters={run.opencv_parameters} className="opencv-inline-summary" /></figure></div></section>;
}

function CircleDetectionView({
  imageId, busy, circleAlgorithm, setCircleAlgorithm, circle, updateCircle, toggleCircleRefine,
  circleInput, setCircleInput, selectedCircleInput, circleInputReady, runCircle,
  circleSlots, circleSlotSources, circleSlot, setCircleSlot, circleRuns,
  compareCircle, setCompareCircle, circleComparisonSlot, setCircleComparisonSlot,
  selectedCircleRun, circlePipelineSelection, setCirclePipelineSelection,
  updateOpenCV,
}) {
  const activeCircle = circle[circleAlgorithm];
  const opencvBoundsValid = circle.opencv.min_radius <= circle.opencv.max_radius;
  const comparisonRun = circleRuns[circleComparisonSlot];
  const customSummary = (run) => {
    if (!run) return null;
    const parameters = run.custom_parameters || {};
    const compactNames = run.algorithm === "cht"
      ? ["min_diam", "max_diam", "min_votes", "min_coverage", "perim_frac"]
      : ["min_diam", "max_diam", "tol", "max_iter", "min_inliers"];
    return <CircleSummaryBlock className="slot-custom-summary" label="CUSTOM RESULT" title={run.algorithm === "cht" ? "Circular Hough" : "RANSAC"} count={run.circles?.length ?? 0} parameters={Object.fromEntries(compactNames.filter((name) => parameters[name] !== undefined).map((name) => [name, parameters[name]]))} />;
  };

  return <div className="circle-layout"><section className="algorithm-section hough-section circle-section circle-section-new"><div className="section-intro"><span className="section-number">02 / CIRCLE DETECTION</span><h2>Trace the roundness.</h2></div><div className="algorithm-body"><aside className="controls"><h3>Algorithm</h3><select className="input-select algorithm-select" value={circleAlgorithm} onChange={(event) => setCircleAlgorithm(event.target.value)}><option value="cht">Circular Hough Transform (from scratch)</option><option value="ransac">RANSAC 3-point fitting (from scratch)</option></select><h3 className="circle-parameter-heading">Parameters</h3><Slider label="Minimum diameter" name="min_diam" value={activeCircle.min_diam} min="4" max="1000" onChange={updateCircle} suffix=" px" /><Slider label="Maximum diameter" name="max_diam" value={activeCircle.max_diam} min="4" max="1000" onChange={updateCircle} suffix=" px" />{circleAlgorithm === "cht" ? <><Slider label="Minimum votes" name="min_votes" value={activeCircle.min_votes} min="5" max="1000" onChange={updateCircle} /><Slider label="Minimum coverage (360°)" name="min_coverage" value={activeCircle.min_coverage} min="0.1" max="1" step="0.05" onChange={updateCircle} /><Slider label="Minimum perimeter share" name="perim_frac" value={activeCircle.perim_frac} min="0" max="1" step="0.05" onChange={updateCircle} /><Slider label="Accumulator smooth" name="smooth" value={activeCircle.smooth} min="1" max="9" step="2" onChange={updateCircle} /><Slider label="Peak NMS window" name="nms_win" value={activeCircle.nms_win} min="3" max="21" step="2" onChange={updateCircle} /><Slider label="Gradient alignment" name="align_cos" value={activeCircle.align_cos} min="0" max="0.95" step="0.05" onChange={updateCircle} /><label className="check-control"><input type="checkbox" checked={activeCircle.refine} onChange={toggleCircleRefine} /> Least-squares refine</label></> : <><Slider label="Inlier tolerance" name="tol" value={activeCircle.tol} min="0.5" max="6" step="0.1" onChange={updateCircle} suffix=" px" /><Slider label="Iterations / round" name="max_iter" value={activeCircle.max_iter} min="100" max="10000" step="100" onChange={updateCircle} /><Slider label="Minimum inliers" name="min_inliers" value={activeCircle.min_inliers} min="5" max="1000" onChange={updateCircle} /><Slider label="Minimum perimeter share" name="perim_frac" value={activeCircle.perim_frac} min="0" max="1" step="0.05" onChange={updateCircle} /><Slider label="Minimum coverage (360°)" name="min_coverage" value={activeCircle.min_coverage} min="0.1" max="1" step="0.05" onChange={updateCircle} /><Slider label="Random seed" name="seed" value={activeCircle.seed} min="0" max="99" onChange={updateCircle} /><label className="check-control"><input type="checkbox" checked={activeCircle.refine} onChange={toggleCircleRefine} /> Least-squares refine</label></>}<button className="run" disabled={!imageId || busy || !circleInputReady || activeCircle.min_diam > activeCircle.max_diam || (compareCircle && !opencvBoundsValid)} onClick={runCircle}>{busy === "circle" ? "Running Circle Detection..." : "Run Circle Detection →"}</button>{activeCircle.min_diam > activeCircle.max_diam && <p className="validation-message">Minimum diameter must be less than or equal to maximum diameter.</p>}{compareCircle && !opencvBoundsValid && <p className="validation-message">OpenCV minimum radius must be less than or equal to maximum radius.</p>}</aside><div className="result-area"><div className="route-row"><div><h3>Input source</h3><p>Use the latest Canny result or either saved Canny output slot.</p></div><select className="input-select" value={circleInput} onChange={(event) => setCircleInput(event.target.value)}><option value="canny">Latest Canny edges</option><option value="output1">Canny Output 1</option><option value="output2">Canny Output 2</option></select></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">CANNY INPUT</span>{selectedCircleInput ? <CircleDiameterPreview source={selectedCircleInput} minDiam={activeCircle.min_diam} maxDiam={activeCircle.max_diam} /> : <div className="viewer-empty">Run Canny or populate a Canny output</div>}</div><SlotCard slot="1" value={circleSlots["1"]} source={circleSlotSources["1"]} selected={circleSlot === "1"} onClick={() => setCircleSlot("1")} summary={customSummary(circleRuns["1"])} /><SlotCard slot="2" value={circleSlots["2"]} source={circleSlotSources["2"]} selected={circleSlot === "2"} onClick={() => setCircleSlot("2")} summary={customSummary(circleRuns["2"])} /></div><div className="route-row output-route"><div><h3>Save Circle result to</h3><p>Choose the destination before running Circle Detection.</p></div><div className="slot-toggle"><button className={circleSlot === "1" ? "chosen" : ""} onClick={() => setCircleSlot("1")}>Output 1</button><button className={circleSlot === "2" ? "chosen" : ""} onClick={() => setCircleSlot("2")}>Output 2</button></div></div></div></div></section>{compareCircle && <aside className="circle-opencv-sidebar"><CircleOpenCVControls values={circle.opencv} onChange={updateOpenCV} /></aside>}<div className="circle-comparison-area"><div className="compare-toggle"><label><input type="checkbox" checked={compareCircle} onChange={(event) => setCompareCircle(event.target.checked)} /> Compare with OpenCV</label>{compareCircle && <select className="comparison-select" value={circleComparisonSlot} onChange={(event) => setCircleComparisonSlot(event.target.value)}><option value="1">Compare Output 1</option><option value="2">Compare Output 2</option></select>}</div>{compareCircle && <CircleComparisonDetails run={comparisonRun} title={`${circleAlgorithm === "cht" ? "Circular Hough" : "RANSAC"} circle overlay`} />}</div><StagePipeline title="Circle stages" steps={selectedCircleRun?.steps} names={circleAlgorithm === "cht" ? ["input_edges", "accumulator", "circle_overlay"] : ["input_edges", "circle_overlay"]} selection={circlePipelineSelection} onSelectionChange={setCirclePipelineSelection} options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }, { value: "input", label: "Input" }]} /></div>;
}

function SnakeContourInput({ source, points, onAddPoint }) {
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const [imageBox, setImageBox] = useState(null);
  const updateImageBox = () => {
    const container = containerRef.current;
    const image = imageRef.current;
    if (!container || !image?.naturalWidth || !image.naturalHeight) return;
    const containerWidth = container.clientWidth;
    const containerHeight = 350;
    const scale = Math.min(containerWidth / image.naturalWidth, containerHeight / image.naturalHeight);
    const width = image.naturalWidth * scale;
    const height = image.naturalHeight * scale;
    setImageBox({ left: (containerWidth - width) / 2, top: (containerHeight - height) / 2, width, height });
  };
  useEffect(() => {
    updateImageBox();
    const observer = new ResizeObserver(updateImageBox);
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [source]);
  const handleClick = (event) => {
    const container = containerRef.current;
    if (!container || !imageBox) return;
    const bounds = container.getBoundingClientRect();
    const x = event.clientX - bounds.left - imageBox.left;
    const y = event.clientY - bounds.top - imageBox.top;
    if (x < 0 || y < 0 || x > imageBox.width || y > imageBox.height) return;
    onAddPoint({
      x: x * imageRef.current.naturalWidth / imageBox.width,
      y: y * imageRef.current.naturalHeight / imageBox.height,
    });
  };
  return <div ref={containerRef} className="snake-contour-input" onClick={handleClick}>
    {source ? <><img ref={imageRef} src={source} onLoad={updateImageBox} alt="Click around the object to initialize the Snake contour" />{imageBox && <svg className="snake-contour-overlay" style={{ left: imageBox.left, top: imageBox.top, width: imageBox.width, height: imageBox.height }} viewBox={`0 0 ${imageRef.current.naturalWidth} ${imageRef.current.naturalHeight}`} aria-hidden="true">
      {points.length > 1 && <polyline points={points.map((point) => `${point.x},${point.y}`).join(" ")} />}
      {points.map((point, index) => <circle key={`${point.x}-${point.y}-${index}`} cx={point.x} cy={point.y} r={Math.max(3, imageRef.current.naturalWidth / 180)} />)}
    </svg>}</> : <div className="viewer-empty">Upload an image</div>}
  </div>;
}

function SnakeDetectionView({ imageId, original, busy, snake, setSnake, snakePoints, setSnakePoints, snakeSlot, setSnakeSlot, snakeSlots, snakeSlotSources, runSnake, snakeResult, snakeMetrics, displayThickness, setDisplayThickness, showVertices, setShowVertices }) {
  const updateSnake = (event) => setSnake((state) => ({ ...state, [event.target.name]: Number(event.target.value) }));
  const addPoint = (point) => { if (snakePoints.length < snake.n_points) setSnakePoints((current) => [...current, point]); };
  return <section className="algorithm-section hough-section snake-section"><div className="section-intro"><span className="section-number">02 / ACTIVE CONTOUR SNAKE</span><h2>Follow the boundary.</h2><p>Click exactly the selected number of samples around the object. The last sample closes to the first automatically.</p></div><div className="algorithm-body"><aside className="controls"><h3>Parameters</h3><Slider label="Contour samples" name="n_points" value={snake.n_points} min="3" max="100" onChange={updateSnake} /><Slider label="Alpha (continuity)" name="alpha" value={snake.alpha} min="0" max="5" step="0.1" onChange={updateSnake} /><Slider label="Beta (curvature)" name="beta" value={snake.beta} min="0" max="5" step="0.1" onChange={updateSnake} /><Slider label="Gamma (image)" name="gamma" value={snake.gamma} min="0" max="5" step="0.1" onChange={updateSnake} /><Slider label="Window size" name="window_size" value={snake.window_size} min="3" max="11" step="2" onChange={updateSnake} /><Slider label="Maximum iterations" name="max_iterations" value={snake.max_iterations} min="1" max="500" onChange={updateSnake} /><h3 className="circle-parameter-heading">Display</h3><Slider label="Contour thickness" name="displayThickness" value={displayThickness} min="1" max="8" onChange={(event) => setDisplayThickness(Number(event.target.value))} /><label className="check-control"><input type="checkbox" checked={showVertices} onChange={(event) => setShowVertices(event.target.checked)} /> Show final vertices</label><button className="run" disabled={!imageId || busy || snakePoints.length !== snake.n_points} onClick={runSnake}>{busy === "snake" ? "Running Snake..." : "Run Snake →"}</button></aside><div className="result-area"><div className="route-row"><div><h3>Initial contour</h3><p>{snakePoints.length} / {snake.n_points} samples selected. The closing edge is implicit.</p></div><button className="slot-toggle button-reset" type="button" onClick={() => setSnakePoints([])}>Clear points</button></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">INPUT / INITIAL CONTOUR</span>{original ? <SnakeContourInput source={original} points={snakePoints} onAddPoint={addPoint} /> : <div className="viewer-empty">Upload an image</div>}</div><SlotCard slot="1" value={snakeSlots["1"]} source={snakeSlotSources["1"]} selected={snakeSlot === "1"} onClick={() => setSnakeSlot("1")} /><SlotCard slot="2" value={snakeSlots["2"]} source={snakeSlotSources["2"]} selected={snakeSlot === "2"} onClick={() => setSnakeSlot("2")} /></div><div className="route-row output-route"><div><h3>Save Snake result to</h3><p>Choose the destination before running Snake.</p></div><div className="slot-toggle"><button className={snakeSlot === "1" ? "chosen" : ""} onClick={() => setSnakeSlot("1")}>Output 1</button><button className={snakeSlot === "2" ? "chosen" : ""} onClick={() => setSnakeSlot("2")}>Output 2</button></div></div>{snakeResult && <p className="snake-result-summary">Iterations: {snakeResult.iterations_run} · Stop reason: {snakeResult.stop_reason} · Final points: {snakeResult.final_contour?.length ?? 0}</p>}{snakeMetrics && <div className="snake-metrics"><b>Chain code</b><code>{snakeMetrics.chain_code.slice(0, 240)}{snakeMetrics.chain_code.length > 240 ? `… (${snakeMetrics.chain_code.length} digits total)` : ""}</code><span>Perimeter: {Number(snakeMetrics.perimeter.toFixed(2))}</span><span>Shoelace area: {Number(snakeMetrics.shoelace_area.toFixed(2))}</span><span>Pixel fill: {snakeMetrics.pixel_fill_area} · Boundary pixels: {snakeMetrics.boundary_pixels}</span><span>Pick's theorem: {snakeMetrics.pick_theorem_valid ? "verified" : "failed"}</span></div>}</div></div></section>;
}

function AxisReferencePreview({ source, aMin, aMax }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !source) return undefined;
    const image = new Image();
    image.onload = () => {
      const width = image.naturalWidth;
      const height = image.naturalHeight;
      const resizeScale = Math.min(1, 400 / Math.max(width, height));
      const context = canvas.getContext("2d");
      canvas.width = width;
      canvas.height = height;
      context.drawImage(image, 0, 0, width, height);
      const centerX = width / 2;
      const centerY = height / 2;
      const drawGuide = (radius, color, label, labelOffset) => {
        context.beginPath();
        context.arc(centerX, centerY, radius / resizeScale, 0, Math.PI * 2);
        context.strokeStyle = color;
        context.lineWidth = Math.max(2, width / 220);
        context.stroke();
        context.fillStyle = color;
        context.font = `700 ${Math.max(12, width / 35)}px monospace`;
        context.fillText(label, centerX + labelOffset, centerY - radius / resizeScale - 8);
      };
      drawGuide(aMin, "#ffe600", `a_min ${aMin}px`, 8);
      drawGuide(aMax, "#ff50be", `a_max ${aMax}px`, 8);
    };
    image.src = source;
    return () => { image.onload = null; };
  }, [source, aMin, aMax]);

  return source ? <canvas ref={canvasRef} role="img" aria-label={`Axis reference guides: a_min ${aMin} pixels and a_max ${aMax} pixels`} /> : <div className="viewer-empty">Upload an image to preview the axis limits</div>;
}

function CircleDiameterPreview({ source, minDiam, maxDiam }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !source) return undefined;
    const image = new Image();
    image.onload = () => {
      const width = image.naturalWidth;
      const height = image.naturalHeight;
      const context = canvas.getContext("2d");
      canvas.width = width;
      canvas.height = height;
      context.drawImage(image, 0, 0, width, height);

      // Match the notebook's diameter-to-radius search bounds exactly.
      const minRadius = 0.85 * minDiam / 2;
      const maxRadius = 1.15 * maxDiam / 2;
      const centerX = width / 2;
      const centerY = height / 2;
      const drawGuide = (radius, color, label, labelOffset) => {
        context.beginPath();
        context.arc(centerX, centerY, radius, 0, Math.PI * 2);
        context.strokeStyle = color;
        context.lineWidth = Math.max(2, width / 320);
        context.setLineDash([8, 6]);
        context.stroke();
        context.setLineDash([]);
        context.fillStyle = color;
        context.font = `700 ${Math.max(12, width / 42)}px monospace`;
        context.fillText(label, centerX + labelOffset, centerY - radius - 8);
      };
      drawGuide(minRadius, "#ff5a5f", `min ø ${minDiam}px`, 8);
      drawGuide(maxRadius, "#ffe600", `max ø ${maxDiam}px`, 8);
      context.beginPath();
      context.moveTo(centerX - 7, centerY);
      context.lineTo(centerX + 7, centerY);
      context.moveTo(centerX, centerY - 7);
      context.lineTo(centerX, centerY + 7);
      context.strokeStyle = "#ffffff";
      context.lineWidth = 1.5;
      context.stroke();
    };
    image.src = source;
    return () => { image.onload = null; };
  }, [source, minDiam, maxDiam]);

  return source ? <canvas ref={canvasRef} role="img" aria-label={`Circle diameter guides: minimum ${minDiam} pixels and maximum ${maxDiam} pixels`} /> : <div className="viewer-empty">Run Canny or populate a Canny output</div>;
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("contour-lab-theme") || "dark");
  const [imageId, setImageId] = useState(null);
  const [original, setOriginal] = useState(null);
  const [canny, setCanny] = useState(initialCanny);
  const [hough, setHough] = useState(initialHough);
  const [ellipse, setEllipse] = useState(initialEllipse);
  const [circle, setCircle] = useState(initialCircle);
  const [snake, setSnake] = useState({ alpha: 1, beta: 1, gamma: 1.2, n_points: 20, window_size: 5, max_iterations: 200 });
  const [displayThickness, setDisplayThickness] = useState(3);
  const [showVertices, setShowVertices] = useState(false);
  const [snakePoints, setSnakePoints] = useState([]);
  const [detectionAlgorithm, setDetectionAlgorithm] = useState("hough");
  const [circleAlgorithm, setCircleAlgorithm] = useState("cht");
  const [cannySlot, setCannySlot] = useState("1");
  const [cannyComparisonSlot, setCannyComparisonSlot] = useState("1");
  const [houghInput, setHoughInput] = useState("canny");
  const [houghSlot, setHoughSlot] = useState("2");
  const [houghComparisonSlot, setHoughComparisonSlot] = useState("2");
  const [ellipseInput, setEllipseInput] = useState("canny");
  const [ellipseSlot, setEllipseSlot] = useState("2");
  const [ellipseComparisonSlot, setEllipseComparisonSlot] = useState("2");
  const [circleInput, setCircleInput] = useState("canny");
  const [circleSlot, setCircleSlot] = useState("2");
  const [snakeSlot, setSnakeSlot] = useState("2");
  const [circleComparisonSlot, setCircleComparisonSlot] = useState("2");
  const [cannySlots, setCannySlots] = useState({ "1": null, "2": null });
  const [cannySlotSources, setCannySlotSources] = useState({ "1": null, "2": null });
  const [houghSlots, setHoughSlots] = useState({ "1": null, "2": null });
  const [houghSlotSources, setHoughSlotSources] = useState({ "1": null, "2": null });
  const [ellipseSlots, setEllipseSlots] = useState({ "1": null, "2": null });
  const [ellipseSlotSources, setEllipseSlotSources] = useState({ "1": null, "2": null });
  const [circleSlots, setCircleSlots] = useState({ "1": null, "2": null });
  const [circleSlotSources, setCircleSlotSources] = useState({ "1": null, "2": null });
  const [snakeSlots, setSnakeSlots] = useState({ "1": null, "2": null });
  const [snakeSlotSources, setSnakeSlotSources] = useState({ "1": null, "2": null });
  const [cannyResult, setCannyResult] = useState(null);
  const [houghResult, setHoughResult] = useState(null);
  const [ellipseResult, setEllipseResult] = useState(null);
  const [circleResult, setCircleResult] = useState(null);
  const [snakeResult, setSnakeResult] = useState(null);
  const [snakeMetrics, setSnakeMetrics] = useState(null);
  const [cannyRuns, setCannyRuns] = useState({});
  const [houghRuns, setHoughRuns] = useState({});
  const [ellipseRuns, setEllipseRuns] = useState({});
  const [circleRuns, setCircleRuns] = useState({});
  const [cannyPipelineSelection, setCannyPipelineSelection] = useState("1");
  const [houghPipelineSelection, setHoughPipelineSelection] = useState("1");
  const [ellipsePipelineSelection, setEllipsePipelineSelection] = useState("1");
  const [circlePipelineSelection, setCirclePipelineSelection] = useState("1");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [compareCanny, setCompareCanny] = useState(false);
  const [compareHough, setCompareHough] = useState(false);
  const [compareEllipse, setCompareEllipse] = useState(false);
  const [compareCircle, setCompareCircle] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("contour-lab-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!original || !snakeResult) return undefined;
    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);
      const drawContour = (points, color, width) => {
        if (!points?.length) return;
        context.beginPath();
        points.forEach((point, index) => index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y));
        context.closePath();
        context.strokeStyle = color;
        context.lineWidth = width;
        context.stroke();
      };
      drawContour(snakeResult.initial_contour, "#f4be60", Math.max(1, displayThickness - 1));
      drawContour(snakeResult.final_contour, "#7dd3c0", displayThickness);
      if (showVertices) {
        context.fillStyle = "#ffffff";
        snakeResult.final_contour?.forEach((point) => {
          context.beginPath();
          context.arc(point.x, point.y, Math.max(2, displayThickness), 0, Math.PI * 2);
          context.fill();
        });
      }
      const output = canvas.toDataURL("image/png");
      setSnakeSlots((state) => ({ ...state, [snakeSlot]: output }));
    };
    image.src = original;
    return () => { image.onload = null; };
  }, [original, snakeResult, displayThickness, showVertices, snakeSlot]);

  const update = (setter) => (event) => setter((state) => ({ ...state, [event.target.name]: Number(event.target.value) }));
  const updateCircle = (event) => setCircle((state) => ({ ...state, [circleAlgorithm]: { ...state[circleAlgorithm], [event.target.name]: Number(event.target.value) } }));
  const updateOpenCV = (event) => setCircle((state) => ({ ...state, opencv: { ...state.opencv, [event.target.name]: Number(event.target.value) } }));
  const toggleCircleRefine = () => setCircle((state) => ({ ...state, [circleAlgorithm]: { ...state[circleAlgorithm], refine: !state[circleAlgorithm].refine } }));

  function resetExpiredSession() {
    setImageId(null); setOriginal(null); setCannyResult(null); setHoughResult(null); setEllipseResult(null); setCircleResult(null);
    setCannyRuns({}); setHoughRuns({}); setEllipseRuns({}); setCircleRuns({});
    setCannySlots({ "1": null, "2": null }); setCannySlotSources({ "1": null, "2": null });
    setHoughSlots({ "1": null, "2": null }); setHoughSlotSources({ "1": null, "2": null });
    setEllipseSlots({ "1": null, "2": null }); setEllipseSlotSources({ "1": null, "2": null });
    setCircleSlots({ "1": null, "2": null }); setCircleSlotSources({ "1": null, "2": null });
    setSnakeSlots({ "1": null, "2": null }); setSnakeSlotSources({ "1": null, "2": null }); setSnakeResult(null); setSnakeMetrics(null);
    setSnakePoints([]);
    setError("The processing session expired after the backend restarted. Please upload the image again.");
  }

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(""); setCannyResult(null); setHoughResult(null); setEllipseResult(null); setCircleResult(null);
    setCannyRuns({}); setHoughRuns({}); setEllipseRuns({}); setCircleRuns({});
    setCannySlots({ "1": null, "2": null }); setCannySlotSources({ "1": null, "2": null });
    setHoughSlots({ "1": null, "2": null }); setHoughSlotSources({ "1": null, "2": null });
    setEllipseSlots({ "1": null, "2": null }); setEllipseSlotSources({ "1": null, "2": null });
    setCircleSlots({ "1": null, "2": null }); setCircleSlotSources({ "1": null, "2": null });
    setSnakeSlots({ "1": null, "2": null }); setSnakeSlotSources({ "1": null, "2": null }); setSnakeResult(null); setSnakeMetrics(null);
    setSnakePoints([]);
    const form = new FormData(); form.append("file", file);
    try {
      const uploadResponse = await fetch(`${API}/images`, { method: "POST", body: form });
      const data = await uploadResponse.json();
      if (!uploadResponse.ok) throw new Error(data.detail || "Upload failed");
      setImageId(data.image_id);
      const originalResponse = await fetch(`${API}/images/${data.image_id}`);
      setOriginal((await originalResponse.json()).original);
    } catch (caught) { setError(caught.message); }
  }

  async function runCanny() {
    if (!imageId) return;
    setBusy("canny"); setError("");
    try {
      const response = await fetch(`${API}/images/${imageId}/canny`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...canny, output_slot: cannySlot }) });
      const data = await response.json();
      if (response.status === 404) { resetExpiredSession(); return; }
      if (!response.ok) throw new Error(data.detail || "Canny processing failed");
      setCannyResult(data); setCannyRuns((state) => ({ ...state, [cannySlot]: data }));
      setCannySlots((state) => ({ ...state, [cannySlot]: data.output })); setCannySlotSources((state) => ({ ...state, [cannySlot]: "Canny edges" }));
      // Ellipse results depend on the Canny edge map. Prevent an older run
      // from masking the newly selected Canny output when the user returns to
      // the Ellipse section.
       setEllipseResult(null); setEllipseRuns({}); setCircleResult(null); setCircleRuns({});
      setEllipseSlots({ "1": null, "2": null }); setEllipseSlotSources({ "1": null, "2": null });
       setCircleSlots({ "1": null, "2": null }); setCircleSlotSources({ "1": null, "2": null });
    } catch (caught) { setError(caught.message); } finally { setBusy(""); }
  }

  async function runHough() {
    if (!imageId) return;
    setBusy("hough"); setError("");
    try {
      const inputSource = houghInput === "1" || houghInput === "2" ? `output${houghInput}` : houghInput;
      const response = await fetch(`${API}/images/${imageId}/hough`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...hough, input_source: inputSource, output_slot: houghSlot }) });
      const data = await response.json();
      if (response.status === 404) { resetExpiredSession(); return; }
      if (!response.ok) throw new Error(data.detail || "Hough processing failed");
      setHoughResult(data); setHoughRuns((state) => ({ ...state, [houghSlot]: data }));
      setHoughSlots((state) => ({ ...state, [houghSlot]: data.output })); setHoughSlotSources((state) => ({ ...state, [houghSlot]: "Hough lines" }));
    } catch (caught) { setError(caught.message); } finally { setBusy(""); }
  }

  async function runEllipse() {
    if (!imageId || ellipse.a_min > ellipse.a_max) return;
    setBusy("ellipse"); setError("");
    try {
      const response = await fetch(`${API}/images/${imageId}/ellipse`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...ellipse, input_source: ellipseInput, output_slot: ellipseSlot, compare_opencv: compareEllipse, resize_input: true, canny }) });
      const data = await response.json();
      if (response.status === 404) { resetExpiredSession(); return; }
      if (!response.ok) throw new Error(data.detail || "Ellipse processing failed");
      setEllipseResult(data); setEllipseRuns((state) => ({ ...state, [ellipseSlot]: data }));
      setEllipseSlots((state) => ({ ...state, [ellipseSlot]: data.output })); setEllipseSlotSources((state) => ({ ...state, [ellipseSlot]: "Ellipse detections" }));
    } catch (caught) { setError(caught.message); } finally { setBusy(""); }
  }

  async function runCircle() {
    if (!imageId) return;
    setBusy("circle"); setError("");
    const selectedCircle = circle[circleAlgorithm];
    try {
      const response = await fetch(`${API}/images/${imageId}/circle`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ algorithm: circleAlgorithm, ...selectedCircle, opencv_dp: circle.opencv.dp, opencv_min_dist: circle.opencv.min_dist, opencv_param1: circle.opencv.param1, opencv_param2: circle.opencv.param2, opencv_min_radius: circle.opencv.min_radius, opencv_max_radius: circle.opencv.max_radius, input_source: circleInput, output_slot: circleSlot, compare_opencv: compareCircle }) });
      const data = await response.json();
      if (response.status === 404) { resetExpiredSession(); return; }
      if (!response.ok) throw new Error(data.detail || "Circle processing failed");
      setCircleResult(data); setCircleRuns((state) => ({ ...state, [circleSlot]: data }));
      setCircleSlots((state) => ({ ...state, [circleSlot]: data.output })); setCircleSlotSources((state) => ({ ...state, [circleSlot]: `${circleAlgorithm === "cht" ? "Circular Hough" : "RANSAC"} circles` }));
    } catch (caught) { setError(caught.message); } finally { setBusy(""); }
  }

  async function runSnake() {
    if (!imageId) return;
    setBusy("snake"); setError("");
    try {
      const response = await fetch(`${API}/images/${imageId}/snake`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...snake, initial_contour: snakePoints, output_slot: snakeSlot }) });
      const data = await response.json();
      if (response.status === 404) { resetExpiredSession(); return; }
      if (!response.ok) throw new Error(data.detail || "Snake processing failed");
      setSnakeResult(data); setSnakeSlots((state) => ({ ...state, [snakeSlot]: data.output })); setSnakeSlotSources((state) => ({ ...state, [snakeSlot]: "Snake contour" }));
      const metricsResponse = await fetch(`${API}/images/${imageId}/contour-metrics`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input_source: "snake", output_slot: snakeSlot }) });
      const metricsData = await metricsResponse.json();
      if (!metricsResponse.ok) throw new Error(metricsData.detail || "Contour metrics failed");
      setSnakeMetrics(metricsData);
    } catch (caught) { setError(caught.message); } finally { setBusy(""); }
  }

  const selectedInput = houghInput === "original" ? original : houghInput === "canny" ? cannyResult?.output : cannySlots[houghInput];
  const selectedEllipseInput = ellipseResult?.steps?.input_edges?.image || (ellipseInput === "canny" ? cannyResult?.output : cannySlots[ellipseInput.replace("output", "")]);
  const selectedCannyRun = cannyPipelineSelection === "input" ? null : cannyRuns[cannyPipelineSelection];
  const selectedHoughRun = houghPipelineSelection === "input" ? null : houghRuns[houghPipelineSelection];
  const selectedEllipseRun = ellipsePipelineSelection === "input" ? null : ellipseRuns[ellipsePipelineSelection];
  const selectedCircleRun = circlePipelineSelection === "input" ? null : circleRuns[circlePipelineSelection];
  const cannyComparison = cannyRuns[cannyComparisonSlot];
  const houghComparison = houghRuns[houghComparisonSlot];
  const ellipseComparison = ellipseRuns[ellipseComparisonSlot];
  const circleComparison = circleRuns[circleComparisonSlot];
  const ellipseInputReady = ellipseInput === "canny" ? Boolean(cannyResult?.output) : Boolean(cannySlots[ellipseInput.replace("output", "")]);
  const selectedCircleInput = circleInput === "canny" ? cannyResult?.output : cannySlots[circleInput.replace("output", "")];
  const circleInputReady = circleInput === "canny" ? Boolean(cannyResult?.output) : Boolean(cannySlots[circleInput.replace("output", "")]);
  const activeCircle = circle[circleAlgorithm];

  return <main>
    <header><div><p className="eyebrow">FROM-SCRATCH COMPUTER VISION / 02</p><h1>Contour <em>lab</em></h1><p className="lede">A deliberate workspace for tracing every transformation.</p></div><div className="header-actions"><button className="theme-toggle" type="button" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`} onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}><span aria-hidden="true">{theme === "dark" ? "☼" : "☾"}</span>{theme === "dark" ? "Light mode" : "Dark mode"}</button><label className="upload">Upload image<input type="file" accept="image/*" onChange={upload} /></label></div></header>
    {error && <div className="error">{error}</div>}
    <section className="source-strip"><div><p className="eyebrow">SOURCE MATRIX</p><h2>{imageId ? "Original image loaded" : "Upload an image to begin"}</h2></div>{original ? <img src={original} alt="Original source" /> : <div className="source-empty">Your source will appear here</div>}</section>
    <div className="detection-selector"><label htmlFor="detection-algorithm">Detection algorithm</label><select id="detection-algorithm" value={detectionAlgorithm} onChange={(event) => setDetectionAlgorithm(event.target.value)}><option value="hough">Hough Line Detection</option><option value="ellipse">Ellipse Detection</option><option value="circle">Circle Detection</option><option value="snake">Snake</option></select></div>

    <section className="algorithm-section canny-section"><div className="section-intro"><span className="section-number">01 / CANNY EDGE DETECTION</span><h2>Find the edges.</h2><p>Build a binary edge map from the original source with the notebook's custom stages.</p></div><div className="algorithm-body"><aside className="controls"><h3>Parameters</h3><Slider label="Kernel size" name="kernel_size" value={canny.kernel_size} min="3" max="15" step="2" onChange={update(setCanny)} /><Slider label="Gaussian sigma" name="sigma" value={canny.sigma} min="0.1" max="10" step="0.1" onChange={update(setCanny)} /><Slider label="Low threshold" name="low_threshold" value={canny.low_threshold} min="0" max="500" onChange={update(setCanny)} /><Slider label="High threshold" name="high_threshold" value={canny.high_threshold} min="0" max="800" onChange={update(setCanny)} /><button className="run" disabled={!imageId || busy} onClick={runCanny}>{busy === "canny" ? "Running Canny..." : "Run Canny →"}</button></aside><div className="result-area"><div className="route-row"><div><h3>Input source</h3><p>Canny always starts from the original uploaded image.</p></div><select className="input-select" value="original" disabled><option value="original">Original image</option></select></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">INPUT</span>{original ? <img src={original} alt="Original Canny input" /> : <div className="viewer-empty">Upload an image</div>}</div><SlotCard slot="1" value={cannySlots["1"]} source={cannySlotSources["1"]} selected={cannySlot === "1"} onClick={() => setCannySlot("1")} /><SlotCard slot="2" value={cannySlots["2"]} source={cannySlotSources["2"]} selected={cannySlot === "2"} onClick={() => setCannySlot("2")} /></div></div></div></section>
    <div className="compare-toggle"><label><input type="checkbox" checked={compareCanny} onChange={(event) => setCompareCanny(event.target.checked)} /> Compare with OpenCV</label>{compareCanny && <select className="comparison-select" value={cannyComparisonSlot} onChange={(event) => setCannyComparisonSlot(event.target.value)}><option value="1">Compare Output 1</option><option value="2">Compare Output 2</option></select>}</div>{compareCanny && <ComparisonPanel title="Canny edges" custom={cannyComparison?.output} opencv={cannyComparison?.opencv_output} metric="Same thresholds · same blurred input" />}<StagePipeline title="Canny stages" steps={selectedCannyRun?.steps} names={["grayscale", "gaussian_blur", "gradient_magnitude", "gradient_direction", "non_maximum_suppression", "edges"]} selection={cannyPipelineSelection} onSelectionChange={setCannyPipelineSelection} options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }, { value: "input", label: "Input" }]} />

    {detectionAlgorithm === "hough" ? <>
      <section className="algorithm-section hough-section"><div className="section-intro"><span className="section-number">02 / HOUGH LINE DETECTION</span><h2>Read the geometry.</h2><p>Choose a source matrix, vote in accumulator space, then render the strongest lines.</p></div><div className="algorithm-body"><aside className="controls"><h3>Parameters</h3><Slider label="Radius resolution" name="radius_res" value={hough.radius_res} min="0.5" max="10" step="0.5" onChange={update(setHough)} /><Slider label="Theta resolution" name="theta_res" value={hough.theta_res} min="0.5" max="20" step="0.5" onChange={update(setHough)} /><Slider label="Maximum lines" name="max_lines" value={hough.max_lines} min="1" max="50" onChange={update(setHough)} /><Slider label="Vote threshold" name="vote_threshold" value={hough.vote_threshold} min="0" max="500" onChange={update(setHough)} /><button className="run" disabled={!imageId || busy || (houghInput !== "original" && houghInput !== "canny" && !cannySlots[houghInput])} onClick={runHough}>{busy === "hough" ? "Running Hough..." : "Run Hough →"}</button></aside><div className="result-area"><div className="route-row"><div><h3>Input source</h3><p>Select the original or either Canny output slot.</p></div><select className="input-select" value={houghInput} onChange={(event) => setHoughInput(event.target.value)}><option value="original">Original</option><option value="1">Canny Output 1</option><option value="2">Canny Output 2</option><option value="canny">Latest Canny edges</option></select></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">INPUT</span>{selectedInput ? <img src={selectedInput} alt="Selected Hough input" /> : <div className="viewer-empty">Select a populated source</div>}</div><SlotCard slot="1" value={houghSlots["1"]} source={houghSlotSources["1"]} selected={houghSlot === "1"} onClick={() => setHoughSlot("1")} /><SlotCard slot="2" value={houghSlots["2"]} source={houghSlotSources["2"]} selected={houghSlot === "2"} onClick={() => setHoughSlot("2")} /></div><div className="route-row output-route"><div><h3>Save Hough result to</h3><p>Choose the destination before running Hough.</p></div><div className="slot-toggle"><button className={houghSlot === "1" ? "chosen" : ""} onClick={() => setHoughSlot("1")}>Output 1</button><button className={houghSlot === "2" ? "chosen" : ""} onClick={() => setHoughSlot("2")}>Output 2</button></div></div></div></div></section>
      <div className="compare-toggle"><label><input type="checkbox" checked={compareHough} onChange={(event) => setCompareHough(event.target.checked)} /> Compare with OpenCV</label>{compareHough && <select className="comparison-select" value={houghComparisonSlot} onChange={(event) => setHoughComparisonSlot(event.target.value)}><option value="1">Compare Output 1</option><option value="2">Compare Output 2</option></select>}</div>{compareHough && <ComparisonPanel title="Hough line overlay" custom={houghComparison?.output} opencv={houghComparison?.opencv_output} metric="Same rho · theta · vote threshold" />}<StagePipeline title="Hough stages" steps={selectedHoughRun?.steps} names={["input_edges", "accumulator", "line_overlay"]} selection={houghPipelineSelection} onSelectionChange={setHoughPipelineSelection} options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }, { value: "input", label: "Input" }]} />
    </> : detectionAlgorithm === "ellipse" ? <>
      <section className="algorithm-section hough-section ellipse-section"><div className="section-intro"><span className="section-number">02 / ELLIPSE DETECTION</span><h2>Find the curves.</h2><p>Fit the custom notebook detector to an existing Canny edge map and render its strongest ellipses.</p></div><div className="algorithm-body"><aside className="controls"><h3>Parameters</h3><Slider label="a_min" name="a_min" value={ellipse.a_min} min="2" max="200" onChange={update(setEllipse)} suffix=" px" /><Slider label="a_max" name="a_max" value={ellipse.a_max} min="10" max="400" onChange={update(setEllipse)} suffix=" px" /><Slider label="Vote threshold" name="vote_threshold" value={ellipse.vote_threshold} min="10" max="500" onChange={update(setEllipse)} /><Slider label="Fit tolerance" name="fit_tol" value={ellipse.fit_tol} min="0.5" max="8" step="0.1" onChange={update(setEllipse)} suffix=" px" /><Slider label="Min. support" name="min_support" value={ellipse.min_support} min="0.1" max="1" step="0.05" onChange={update(setEllipse)} /><Slider label="Max ellipses/component" name="max_ellipses" value={ellipse.max_ellipses} min="1" max="8" onChange={update(setEllipse)} /><Slider label="RANSAC iterations" name="ransac_iters" value={ellipse.ransac_iters} min="50" max="1500" step="50" onChange={update(setEllipse)} /><button className="run" disabled={!imageId || busy || !ellipseInputReady || ellipse.a_min > ellipse.a_max} onClick={runEllipse}>{busy === "ellipse" ? "Running Ellipse..." : "Run Ellipse →"}</button>{ellipse.a_min > ellipse.a_max && <p className="validation-message">a_min must be less than or equal to a_max.</p>}</aside><div className="result-area"><div className="route-row"><div><h3>Input source</h3><p>Latest Canny reruns the current Canny on a notebook-sized input; saved Canny slots are reused and resized.</p></div><select className="input-select" value={ellipseInput} onChange={(event) => setEllipseInput(event.target.value)}><option value="canny">Latest Canny (notebook-sized)</option><option value="output1">Canny Output 1</option><option value="output2">Canny Output 2</option></select></div><div className="ellipse-reference"><div><h3>Axis reference</h3><p>Yellow shows <b>a_min</b>; pink shows <b>a_max</b>. Both guides update immediately on the original image.</p></div><AxisReferencePreview source={original} aMin={ellipse.a_min} aMax={ellipse.a_max} /></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">CANNY INPUT</span>{selectedEllipseInput ? <img src={selectedEllipseInput} alt="Selected Canny input for ellipse detection" /> : <div className="viewer-empty">Run Canny or populate a Canny output</div>}</div><SlotCard slot="1" value={ellipseSlots["1"]} source={ellipseSlotSources["1"]} selected={ellipseSlot === "1"} onClick={() => setEllipseSlot("1")} /><SlotCard slot="2" value={ellipseSlots["2"]} source={ellipseSlotSources["2"]} selected={ellipseSlot === "2"} onClick={() => setEllipseSlot("2")} /></div><div className="route-row output-route"><div><h3>Save Ellipse result to</h3><p>Choose the destination before running Ellipse Detection.</p></div><div className="slot-toggle"><button className={ellipseSlot === "1" ? "chosen" : ""} onClick={() => setEllipseSlot("1")}>Output 1</button><button className={ellipseSlot === "2" ? "chosen" : ""} onClick={() => setEllipseSlot("2")}>Output 2</button></div></div></div></div></section>
      <div className="compare-toggle"><label><input type="checkbox" checked={compareEllipse} onChange={(event) => setCompareEllipse(event.target.checked)} /> Compare with OpenCV</label>{compareEllipse && <select className="comparison-select" value={ellipseComparisonSlot} onChange={(event) => setEllipseComparisonSlot(event.target.value)}><option value="1">Compare Output 1</option><option value="2">Compare Output 2</option></select>}</div>{compareEllipse && <ComparisonPanel title="Ellipse overlay" custom={ellipseComparison?.output} opencv={ellipseComparison?.opencv_output} metric="Same edge input · same axis filters" />}<StagePipeline title="Ellipse stages" steps={selectedEllipseRun?.steps} names={["input_edges", "ellipse_overlay"]} selection={ellipsePipelineSelection} onSelectionChange={setEllipsePipelineSelection} options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }, { value: "input", label: "Input" }]} />
    </> : <>
      <section className="algorithm-section hough-section circle-section"><div className="section-intro"><span className="section-number">02 / CIRCLE DETECTION</span><h2>Trace the roundness.</h2><p>Run one of the notebook's independent from-scratch circle detectors on an existing Canny edge map.</p></div><div className="algorithm-body"><aside className="controls"><h3>Algorithm</h3><select className="input-select algorithm-select" value={circleAlgorithm} onChange={(event) => setCircleAlgorithm(event.target.value)}><option value="cht">Circular Hough Transform (from scratch)</option><option value="ransac">RANSAC 3-point fitting (from scratch)</option></select><h3 className="circle-parameter-heading">Parameters</h3><Slider label="Minimum diameter" name="min_diam" value={activeCircle.min_diam} min="4" max="1000" onChange={updateCircle} suffix=" px" /><Slider label="Maximum diameter" name="max_diam" value={activeCircle.max_diam} min="4" max="1000" onChange={updateCircle} suffix=" px" />{circleAlgorithm === "cht" ? <><Slider label="Minimum votes" name="min_votes" value={activeCircle.min_votes} min="5" max="1000" onChange={updateCircle} /><Slider label="Minimum coverage (360°)" name="min_coverage" value={activeCircle.min_coverage} min="0.1" max="1" step="0.05" onChange={updateCircle} /><Slider label="Minimum perimeter share" name="perim_frac" value={activeCircle.perim_frac} min="0" max="1" step="0.05" onChange={updateCircle} /><Slider label="Accumulator smooth" name="smooth" value={activeCircle.smooth} min="1" max="9" step="2" onChange={updateCircle} /><Slider label="Peak NMS window" name="nms_win" value={activeCircle.nms_win} min="3" max="21" step="2" onChange={updateCircle} /><Slider label="Gradient alignment" name="align_cos" value={activeCircle.align_cos} min="0" max="0.95" step="0.05" onChange={updateCircle} /><label className="check-control"><input type="checkbox" checked={activeCircle.refine} onChange={toggleCircleRefine} /> Least-squares refine</label></> : <><Slider label="Inlier tolerance" name="tol" value={activeCircle.tol} min="0.5" max="6" step="0.1" onChange={updateCircle} suffix=" px" /><Slider label="Iterations / round" name="max_iter" value={activeCircle.max_iter} min="100" max="10000" step="100" onChange={updateCircle} /><Slider label="Minimum inliers" name="min_inliers" value={activeCircle.min_inliers} min="5" max="1000" onChange={updateCircle} /><Slider label="Minimum perimeter share" name="perim_frac" value={activeCircle.perim_frac} min="0" max="1" step="0.05" onChange={updateCircle} /><Slider label="Minimum coverage (360°)" name="min_coverage" value={activeCircle.min_coverage} min="0.1" max="1" step="0.05" onChange={updateCircle} /><Slider label="Random seed" name="seed" value={activeCircle.seed} min="0" max="99" onChange={updateCircle} /><label className="check-control"><input type="checkbox" checked={activeCircle.refine} onChange={toggleCircleRefine} /> Least-squares refine</label></>}<button className="run" disabled={!imageId || busy || !circleInputReady || activeCircle.min_diam > activeCircle.max_diam} onClick={runCircle}>{busy === "circle" ? "Running Circle Detection..." : "Run Circle Detection →"}</button>{activeCircle.min_diam > activeCircle.max_diam && <p className="validation-message">Minimum diameter must be less than or equal to maximum diameter.</p>}</aside><div className="result-area"><div className="route-row"><div><h3>Input source</h3><p>Use the latest Canny result or either saved Canny output slot. Circle detection never runs its own Canny stage.</p></div><select className="input-select" value={circleInput} onChange={(event) => setCircleInput(event.target.value)}><option value="canny">Latest Canny edges</option><option value="output1">Canny Output 1</option><option value="output2">Canny Output 2</option></select></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">CANNY INPUT</span>{selectedCircleInput ? <CircleDiameterPreview source={selectedCircleInput} minDiam={activeCircle.min_diam} maxDiam={activeCircle.max_diam} /> : <div className="viewer-empty">Run Canny or populate a Canny output</div>}</div><SlotCard slot="1" value={circleSlots["1"]} source={circleSlotSources["1"]} selected={circleSlot === "1"} onClick={() => setCircleSlot("1")} /><SlotCard slot="2" value={circleSlots["2"]} source={circleSlotSources["2"]} selected={circleSlot === "2"} onClick={() => setCircleSlot("2")} /></div><div className="route-row output-route"><div><h3>Save Circle result to</h3><p>Choose the destination before running Circle Detection.</p></div><div className="slot-toggle"><button className={circleSlot === "1" ? "chosen" : ""} onClick={() => setCircleSlot("1")}>Output 1</button><button className={circleSlot === "2" ? "chosen" : ""} onClick={() => setCircleSlot("2")}>Output 2</button></div></div></div></div></section>
      <div className="compare-toggle"><label><input type="checkbox" checked={compareCircle} onChange={(event) => setCompareCircle(event.target.checked)} /> Compare with OpenCV</label>{compareCircle && <select className="comparison-select" value={circleComparisonSlot} onChange={(event) => setCircleComparisonSlot(event.target.value)}><option value="1">Compare Output 1</option><option value="2">Compare Output 2</option></select>}</div>{compareCircle && <ComparisonPanel title={`${circleAlgorithm === "cht" ? "Circular Hough" : "RANSAC"} circle overlay`} custom={circleComparison?.output} opencv={circleComparison?.opencv_output} metric="Same Canny blur · same radius bounds" />}<StagePipeline title="Circle stages" steps={selectedCircleRun?.steps} names={circleAlgorithm === "cht" ? ["input_edges", "accumulator", "circle_overlay"] : ["input_edges", "circle_overlay"]} selection={circlePipelineSelection} onSelectionChange={setCirclePipelineSelection} options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }, { value: "input", label: "Input" }]} />
    </>}
    {detectionAlgorithm === "circle" && <CircleDetectionView imageId={imageId} busy={busy} circleAlgorithm={circleAlgorithm} setCircleAlgorithm={setCircleAlgorithm} circle={circle} updateCircle={updateCircle} toggleCircleRefine={toggleCircleRefine} circleInput={circleInput} setCircleInput={setCircleInput} selectedCircleInput={selectedCircleInput} circleInputReady={circleInputReady} runCircle={runCircle} circleSlots={circleSlots} circleSlotSources={circleSlotSources} circleSlot={circleSlot} setCircleSlot={setCircleSlot} circleRuns={circleRuns} compareCircle={compareCircle} setCompareCircle={setCompareCircle} circleComparisonSlot={circleComparisonSlot} setCircleComparisonSlot={setCircleComparisonSlot} selectedCircleRun={selectedCircleRun} circlePipelineSelection={circlePipelineSelection} setCirclePipelineSelection={setCirclePipelineSelection} updateOpenCV={updateOpenCV} />}
    {detectionAlgorithm === "snake" && <SnakeDetectionView imageId={imageId} original={original} busy={busy} snake={snake} setSnake={setSnake} snakePoints={snakePoints} setSnakePoints={setSnakePoints} snakeSlot={snakeSlot} setSnakeSlot={setSnakeSlot} snakeSlots={snakeSlots} snakeSlotSources={snakeSlotSources} runSnake={runSnake} snakeResult={snakeResult} snakeMetrics={snakeMetrics} displayThickness={displayThickness} setDisplayThickness={setDisplayThickness} showVertices={showVertices} setShowVertices={setShowVertices} />}
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);
