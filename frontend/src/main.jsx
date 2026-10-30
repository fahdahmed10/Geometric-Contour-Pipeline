import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = "http://localhost:8000/api";
const initialCanny = { kernel_size: 7, sigma: 1, low_threshold: 80, high_threshold: 200 };
const initialHough = { radius_res: 1, theta_res: 1, max_lines: 10, vote_threshold: 20 };
const initialEllipse = { a_min: 10, a_max: 150, vote_threshold: 40, fit_tol: 2, min_support: 0.5, max_ellipses: 3, ransac_iters: 300 };

function Slider({ label, name, value, min, max, step = 1, onChange, suffix = "" }) {
  return <label className="control"><span>{label}<b>{value}{suffix}</b></span><input type="range" name={name} value={value} min={min} max={max} step={step} onChange={onChange} /></label>;
}

function SlotCard({ slot, value, source, onClick, selected }) {
  return <button className={`slot-card ${value ? "filled" : ""} ${selected ? "active" : ""}`} onClick={onClick}>
    <span className="slot-index">{slot}</span><span className="slot-label">{source || "Available slot"}</span>
    {value ? <img src={value} alt={`${source} in Output ${slot}`} /> : <span className="slot-empty">Drop result here</span>}
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

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("contour-lab-theme") || "dark");
  const [imageId, setImageId] = useState(null);
  const [original, setOriginal] = useState(null);
  const [canny, setCanny] = useState(initialCanny);
  const [hough, setHough] = useState(initialHough);
  const [ellipse, setEllipse] = useState(initialEllipse);
  const [detectionAlgorithm, setDetectionAlgorithm] = useState("hough");
  const [cannySlot, setCannySlot] = useState("1");
  const [cannyComparisonSlot, setCannyComparisonSlot] = useState("1");
  const [houghInput, setHoughInput] = useState("canny");
  const [houghSlot, setHoughSlot] = useState("2");
  const [houghComparisonSlot, setHoughComparisonSlot] = useState("2");
  const [ellipseInput, setEllipseInput] = useState("canny");
  const [ellipseSlot, setEllipseSlot] = useState("2");
  const [ellipseComparisonSlot, setEllipseComparisonSlot] = useState("2");
  const [cannySlots, setCannySlots] = useState({ "1": null, "2": null });
  const [cannySlotSources, setCannySlotSources] = useState({ "1": null, "2": null });
  const [houghSlots, setHoughSlots] = useState({ "1": null, "2": null });
  const [houghSlotSources, setHoughSlotSources] = useState({ "1": null, "2": null });
  const [ellipseSlots, setEllipseSlots] = useState({ "1": null, "2": null });
  const [ellipseSlotSources, setEllipseSlotSources] = useState({ "1": null, "2": null });
  const [cannyResult, setCannyResult] = useState(null);
  const [houghResult, setHoughResult] = useState(null);
  const [ellipseResult, setEllipseResult] = useState(null);
  const [cannyRuns, setCannyRuns] = useState({});
  const [houghRuns, setHoughRuns] = useState({});
  const [ellipseRuns, setEllipseRuns] = useState({});
  const [cannyPipelineSelection, setCannyPipelineSelection] = useState("1");
  const [houghPipelineSelection, setHoughPipelineSelection] = useState("1");
  const [ellipsePipelineSelection, setEllipsePipelineSelection] = useState("1");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [compareCanny, setCompareCanny] = useState(false);
  const [compareHough, setCompareHough] = useState(false);
  const [compareEllipse, setCompareEllipse] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("contour-lab-theme", theme);
  }, [theme]);

  const update = (setter) => (event) => setter((state) => ({ ...state, [event.target.name]: Number(event.target.value) }));

  function resetExpiredSession() {
    setImageId(null); setOriginal(null); setCannyResult(null); setHoughResult(null); setEllipseResult(null);
    setCannyRuns({}); setHoughRuns({}); setEllipseRuns({});
    setCannySlots({ "1": null, "2": null }); setCannySlotSources({ "1": null, "2": null });
    setHoughSlots({ "1": null, "2": null }); setHoughSlotSources({ "1": null, "2": null });
    setEllipseSlots({ "1": null, "2": null }); setEllipseSlotSources({ "1": null, "2": null });
    setError("The processing session expired after the backend restarted. Please upload the image again.");
  }

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(""); setCannyResult(null); setHoughResult(null); setEllipseResult(null);
    setCannyRuns({}); setHoughRuns({}); setEllipseRuns({});
    setCannySlots({ "1": null, "2": null }); setCannySlotSources({ "1": null, "2": null });
    setHoughSlots({ "1": null, "2": null }); setHoughSlotSources({ "1": null, "2": null });
    setEllipseSlots({ "1": null, "2": null }); setEllipseSlotSources({ "1": null, "2": null });
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
      setEllipseResult(null); setEllipseRuns({});
      setEllipseSlots({ "1": null, "2": null }); setEllipseSlotSources({ "1": null, "2": null });
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

  const selectedInput = houghInput === "original" ? original : houghInput === "canny" ? cannyResult?.output : cannySlots[houghInput];
  const selectedEllipseInput = ellipseResult?.steps?.input_edges?.image || (ellipseInput === "canny" ? cannyResult?.output : cannySlots[ellipseInput.replace("output", "")]);
  const selectedCannyRun = cannyPipelineSelection === "input" ? null : cannyRuns[cannyPipelineSelection];
  const selectedHoughRun = houghPipelineSelection === "input" ? null : houghRuns[houghPipelineSelection];
  const selectedEllipseRun = ellipsePipelineSelection === "input" ? null : ellipseRuns[ellipsePipelineSelection];
  const cannyComparison = cannyRuns[cannyComparisonSlot];
  const houghComparison = houghRuns[houghComparisonSlot];
  const ellipseComparison = ellipseRuns[ellipseComparisonSlot];
  const ellipseInputReady = ellipseInput === "canny" ? Boolean(cannyResult?.output) : Boolean(cannySlots[ellipseInput.replace("output", "")]);

  return <main>
    <header><div><p className="eyebrow">FROM-SCRATCH COMPUTER VISION / 02</p><h1>Contour <em>lab</em></h1><p className="lede">A deliberate workspace for tracing every transformation.</p></div><div className="header-actions"><button className="theme-toggle" type="button" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`} onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}><span aria-hidden="true">{theme === "dark" ? "☼" : "☾"}</span>{theme === "dark" ? "Light mode" : "Dark mode"}</button><label className="upload">Upload image<input type="file" accept="image/*" onChange={upload} /></label></div></header>
    {error && <div className="error">{error}</div>}
    <section className="source-strip"><div><p className="eyebrow">SOURCE MATRIX</p><h2>{imageId ? "Original image loaded" : "Upload an image to begin"}</h2></div>{original ? <img src={original} alt="Original source" /> : <div className="source-empty">Your source will appear here</div>}</section>
    <div className="detection-selector"><label htmlFor="detection-algorithm">Detection algorithm</label><select id="detection-algorithm" value={detectionAlgorithm} onChange={(event) => setDetectionAlgorithm(event.target.value)}><option value="hough">Hough Line Detection</option><option value="ellipse">Ellipse Detection</option></select></div>

    <section className="algorithm-section canny-section"><div className="section-intro"><span className="section-number">01 / CANNY EDGE DETECTION</span><h2>Find the edges.</h2><p>Build a binary edge map from the original source with the notebook's custom stages.</p></div><div className="algorithm-body"><aside className="controls"><h3>Parameters</h3><Slider label="Kernel size" name="kernel_size" value={canny.kernel_size} min="3" max="15" step="2" onChange={update(setCanny)} /><Slider label="Gaussian sigma" name="sigma" value={canny.sigma} min="0.1" max="10" step="0.1" onChange={update(setCanny)} /><Slider label="Low threshold" name="low_threshold" value={canny.low_threshold} min="0" max="500" onChange={update(setCanny)} /><Slider label="High threshold" name="high_threshold" value={canny.high_threshold} min="0" max="800" onChange={update(setCanny)} /><button className="run" disabled={!imageId || busy} onClick={runCanny}>{busy === "canny" ? "Running Canny..." : "Run Canny →"}</button></aside><div className="result-area"><div className="route-row"><div><h3>Input source</h3><p>Canny always starts from the original uploaded image.</p></div><select className="input-select" value="original" disabled><option value="original">Original image</option></select></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">INPUT</span>{original ? <img src={original} alt="Original Canny input" /> : <div className="viewer-empty">Upload an image</div>}</div><SlotCard slot="1" value={cannySlots["1"]} source={cannySlotSources["1"]} selected={cannySlot === "1"} onClick={() => setCannySlot("1")} /><SlotCard slot="2" value={cannySlots["2"]} source={cannySlotSources["2"]} selected={cannySlot === "2"} onClick={() => setCannySlot("2")} /></div></div></div></section>
    <div className="compare-toggle"><label><input type="checkbox" checked={compareCanny} onChange={(event) => setCompareCanny(event.target.checked)} /> Compare with OpenCV</label>{compareCanny && <select className="comparison-select" value={cannyComparisonSlot} onChange={(event) => setCannyComparisonSlot(event.target.value)}><option value="1">Compare Output 1</option><option value="2">Compare Output 2</option></select>}</div>{compareCanny && <ComparisonPanel title="Canny edges" custom={cannyComparison?.output} opencv={cannyComparison?.opencv_output} metric="Same thresholds · same blurred input" />}<StagePipeline title="Canny stages" steps={selectedCannyRun?.steps} names={["grayscale", "gaussian_blur", "gradient_magnitude", "gradient_direction", "non_maximum_suppression", "edges"]} selection={cannyPipelineSelection} onSelectionChange={setCannyPipelineSelection} options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }, { value: "input", label: "Input" }]} />

    {detectionAlgorithm === "hough" ? <>
      <section className="algorithm-section hough-section"><div className="section-intro"><span className="section-number">02 / HOUGH LINE DETECTION</span><h2>Read the geometry.</h2><p>Choose a source matrix, vote in accumulator space, then render the strongest lines.</p></div><div className="algorithm-body"><aside className="controls"><h3>Parameters</h3><Slider label="Radius resolution" name="radius_res" value={hough.radius_res} min="0.5" max="10" step="0.5" onChange={update(setHough)} /><Slider label="Theta resolution" name="theta_res" value={hough.theta_res} min="0.5" max="20" step="0.5" onChange={update(setHough)} /><Slider label="Maximum lines" name="max_lines" value={hough.max_lines} min="1" max="50" onChange={update(setHough)} /><Slider label="Vote threshold" name="vote_threshold" value={hough.vote_threshold} min="0" max="500" onChange={update(setHough)} /><button className="run" disabled={!imageId || busy || (houghInput !== "original" && houghInput !== "canny" && !cannySlots[houghInput])} onClick={runHough}>{busy === "hough" ? "Running Hough..." : "Run Hough →"}</button></aside><div className="result-area"><div className="route-row"><div><h3>Input source</h3><p>Select the original or either Canny output slot.</p></div><select className="input-select" value={houghInput} onChange={(event) => setHoughInput(event.target.value)}><option value="original">Original</option><option value="1">Canny Output 1</option><option value="2">Canny Output 2</option><option value="canny">Latest Canny edges</option></select></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">INPUT</span>{selectedInput ? <img src={selectedInput} alt="Selected Hough input" /> : <div className="viewer-empty">Select a populated source</div>}</div><SlotCard slot="1" value={houghSlots["1"]} source={houghSlotSources["1"]} selected={houghSlot === "1"} onClick={() => setHoughSlot("1")} /><SlotCard slot="2" value={houghSlots["2"]} source={houghSlotSources["2"]} selected={houghSlot === "2"} onClick={() => setHoughSlot("2")} /></div><div className="route-row output-route"><div><h3>Save Hough result to</h3><p>Choose the destination before running Hough.</p></div><div className="slot-toggle"><button className={houghSlot === "1" ? "chosen" : ""} onClick={() => setHoughSlot("1")}>Output 1</button><button className={houghSlot === "2" ? "chosen" : ""} onClick={() => setHoughSlot("2")}>Output 2</button></div></div></div></div></section>
      <div className="compare-toggle"><label><input type="checkbox" checked={compareHough} onChange={(event) => setCompareHough(event.target.checked)} /> Compare with OpenCV</label>{compareHough && <select className="comparison-select" value={houghComparisonSlot} onChange={(event) => setHoughComparisonSlot(event.target.value)}><option value="1">Compare Output 1</option><option value="2">Compare Output 2</option></select>}</div>{compareHough && <ComparisonPanel title="Hough line overlay" custom={houghComparison?.output} opencv={houghComparison?.opencv_output} metric="Same rho · theta · vote threshold" />}<StagePipeline title="Hough stages" steps={selectedHoughRun?.steps} names={["input_edges", "accumulator", "line_overlay"]} selection={houghPipelineSelection} onSelectionChange={setHoughPipelineSelection} options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }, { value: "input", label: "Input" }]} />
    </> : <>
      <section className="algorithm-section hough-section ellipse-section"><div className="section-intro"><span className="section-number">02 / ELLIPSE DETECTION</span><h2>Find the curves.</h2><p>Fit the custom notebook detector to an existing Canny edge map and render its strongest ellipses.</p></div><div className="algorithm-body"><aside className="controls"><h3>Parameters</h3><Slider label="a_min" name="a_min" value={ellipse.a_min} min="2" max="200" onChange={update(setEllipse)} suffix=" px" /><Slider label="a_max" name="a_max" value={ellipse.a_max} min="10" max="400" onChange={update(setEllipse)} suffix=" px" /><Slider label="Vote threshold" name="vote_threshold" value={ellipse.vote_threshold} min="10" max="500" onChange={update(setEllipse)} /><Slider label="Fit tolerance" name="fit_tol" value={ellipse.fit_tol} min="0.5" max="8" step="0.1" onChange={update(setEllipse)} suffix=" px" /><Slider label="Min. support" name="min_support" value={ellipse.min_support} min="0.1" max="1" step="0.05" onChange={update(setEllipse)} /><Slider label="Max ellipses/component" name="max_ellipses" value={ellipse.max_ellipses} min="1" max="8" onChange={update(setEllipse)} /><Slider label="RANSAC iterations" name="ransac_iters" value={ellipse.ransac_iters} min="50" max="1500" step="50" onChange={update(setEllipse)} /><button className="run" disabled={!imageId || busy || !ellipseInputReady || ellipse.a_min > ellipse.a_max} onClick={runEllipse}>{busy === "ellipse" ? "Running Ellipse..." : "Run Ellipse →"}</button>{ellipse.a_min > ellipse.a_max && <p className="validation-message">a_min must be less than or equal to a_max.</p>}</aside><div className="result-area"><div className="route-row"><div><h3>Input source</h3><p>Latest Canny reruns the current Canny on a notebook-sized input; saved Canny slots are reused and resized.</p></div><select className="input-select" value={ellipseInput} onChange={(event) => setEllipseInput(event.target.value)}><option value="canny">Latest Canny (notebook-sized)</option><option value="output1">Canny Output 1</option><option value="output2">Canny Output 2</option></select></div><div className="ellipse-reference"><div><h3>Axis reference</h3><p>Yellow shows <b>a_min</b>; pink shows <b>a_max</b>. Both guides update immediately on the original image.</p></div><AxisReferencePreview source={original} aMin={ellipse.a_min} aMax={ellipse.a_max} /></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">CANNY INPUT</span>{selectedEllipseInput ? <img src={selectedEllipseInput} alt="Selected Canny input for ellipse detection" /> : <div className="viewer-empty">Run Canny or populate a Canny output</div>}</div><SlotCard slot="1" value={ellipseSlots["1"]} source={ellipseSlotSources["1"]} selected={ellipseSlot === "1"} onClick={() => setEllipseSlot("1")} /><SlotCard slot="2" value={ellipseSlots["2"]} source={ellipseSlotSources["2"]} selected={ellipseSlot === "2"} onClick={() => setEllipseSlot("2")} /></div><div className="route-row output-route"><div><h3>Save Ellipse result to</h3><p>Choose the destination before running Ellipse Detection.</p></div><div className="slot-toggle"><button className={ellipseSlot === "1" ? "chosen" : ""} onClick={() => setEllipseSlot("1")}>Output 1</button><button className={ellipseSlot === "2" ? "chosen" : ""} onClick={() => setEllipseSlot("2")}>Output 2</button></div></div></div></div></section>
      <div className="compare-toggle"><label><input type="checkbox" checked={compareEllipse} onChange={(event) => setCompareEllipse(event.target.checked)} /> Compare with OpenCV</label>{compareEllipse && <select className="comparison-select" value={ellipseComparisonSlot} onChange={(event) => setEllipseComparisonSlot(event.target.value)}><option value="1">Compare Output 1</option><option value="2">Compare Output 2</option></select>}</div>{compareEllipse && <ComparisonPanel title="Ellipse overlay" custom={ellipseComparison?.output} opencv={ellipseComparison?.opencv_output} metric="Same edge input · same axis filters" />}<StagePipeline title="Ellipse stages" steps={selectedEllipseRun?.steps} names={["input_edges", "ellipse_overlay"]} selection={ellipsePipelineSelection} onSelectionChange={setEllipsePipelineSelection} options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }, { value: "input", label: "Input" }]} />
    </>}
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);
