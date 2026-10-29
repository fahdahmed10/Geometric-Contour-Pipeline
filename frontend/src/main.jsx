import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = "http://localhost:8000/api";
const initialCanny = { kernel_size: 7, sigma: 1, low_threshold: 80, high_threshold: 200 };
const initialHough = { radius_res: 1, theta_res: 1, max_lines: 10, vote_threshold: 20 };

function Slider({ label, name, value, min, max, step = 1, onChange }) {
  return <label className="control"><span>{label}<b>{value}</b></span><input type="range" name={name} value={value} min={min} max={max} step={step} onChange={onChange} /></label>;
}

function SlotCard({ slot, value, source, onClick, selected }) {
  return <button className={`slot-card ${value ? "filled" : ""} ${selected ? "active" : ""}`} onClick={onClick}>
    <span className="slot-index">{slot}</span><span className="slot-label">{source || "Available slot"}</span>
    {value ? <img src={value} alt={`${source} in Output ${slot}`} /> : <span className="slot-empty">Drop result here</span>}
  </button>;
}

function StagePipeline({ title, steps, names }) {
  return <section className="stage-panel"><div className="section-heading"><span className="section-number">STAGE PIPELINE</span><h2>{title}</h2></div>
    <div className="stage-grid">{names.map((name) => {
      const step = steps?.[name];
      return <figure className={step ? "stage available" : "stage"} key={name}>{step ? <img src={step.image} alt={name} /> : <div className="stage-placeholder">Awaiting run</div>}<figcaption>{name.replaceAll("_", " ")}</figcaption></figure>;
    })}</div>
  </section>;
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("contour-lab-theme") || "dark");
  const [imageId, setImageId] = useState(null);
  const [original, setOriginal] = useState(null);
  const [canny, setCanny] = useState(initialCanny);
  const [hough, setHough] = useState(initialHough);
  const [cannySlot, setCannySlot] = useState("1");
  const [houghInput, setHoughInput] = useState("canny");
  const [houghSlot, setHoughSlot] = useState("2");
  const [cannySlots, setCannySlots] = useState({ "1": null, "2": null });
  const [cannySlotSources, setCannySlotSources] = useState({ "1": null, "2": null });
  const [houghSlots, setHoughSlots] = useState({ "1": null, "2": null });
  const [houghSlotSources, setHoughSlotSources] = useState({ "1": null, "2": null });
  const [cannyResult, setCannyResult] = useState(null);
  const [houghResult, setHoughResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");

  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("contour-lab-theme", theme);
  }, [theme]);

  const update = (setter) => (event) => setter((state) => ({ ...state, [event.target.name]: Number(event.target.value) }));

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(""); setCannyResult(null); setHoughResult(null); setCannySlots({ "1": null, "2": null }); setCannySlotSources({ "1": null, "2": null }); setHoughSlots({ "1": null, "2": null }); setHoughSlotSources({ "1": null, "2": null });
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
      if (!response.ok) throw new Error(data.detail || "Canny processing failed");
      setCannyResult(data);
      setCannySlots((state) => ({ ...state, [cannySlot]: data.output }));
      setCannySlotSources((state) => ({ ...state, [cannySlot]: "Canny edges" }));
    } catch (caught) { setError(caught.message); } finally { setBusy(""); }
  }

  async function runHough() {
    if (!imageId) return;
    setBusy("hough"); setError("");
    try {
      const inputSource = houghInput === "1" || houghInput === "2" ? `output${houghInput}` : houghInput;
      const response = await fetch(`${API}/images/${imageId}/hough`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ...hough, input_source: inputSource, output_slot: houghSlot }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Hough processing failed");
      setHoughResult(data);
      setHoughSlots((state) => ({ ...state, [houghSlot]: data.output }));
      setHoughSlotSources((state) => ({ ...state, [houghSlot]: "Hough lines" }));
    } catch (caught) { setError(caught.message); } finally { setBusy(""); }
  }

  const selectedInput = houghInput === "original" ? original : houghInput === "canny" ? cannyResult?.output : cannySlots[houghInput];
  return <main>
    <header><div><p className="eyebrow">FROM-SCRATCH COMPUTER VISION / 02</p><h1>Contour <em>lab</em></h1><p className="lede">A deliberate workspace for tracing every transformation.</p></div><div className="header-actions"><button className="theme-toggle" type="button" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`} onClick={() => setTheme((current) => current === "dark" ? "light" : "dark")}><span aria-hidden="true">{theme === "dark" ? "☼" : "☾"}</span>{theme === "dark" ? "Light mode" : "Dark mode"}</button><label className="upload">Upload image<input type="file" accept="image/*" onChange={upload} /></label></div></header>
    {error && <div className="error">{error}</div>}
    <section className="source-strip"><div><p className="eyebrow">SOURCE MATRIX</p><h2>{imageId ? "Original image loaded" : "Upload an image to begin"}</h2></div>{original ? <img src={original} alt="Original source" /> : <div className="source-empty">Your source will appear here</div>}</section>

    <section className="algorithm-section canny-section"><div className="section-intro"><span className="section-number">01 / CANNY EDGE DETECTION</span><h2>Find the edges.</h2><p>Build a binary edge map from the original source with the notebook's custom stages.</p></div><div className="algorithm-body"><aside className="controls"><h3>Parameters</h3><Slider label="Kernel size" name="kernel_size" value={canny.kernel_size} min="3" max="15" step="2" onChange={update(setCanny)} /><Slider label="Gaussian sigma" name="sigma" value={canny.sigma} min="0.1" max="10" step="0.1" onChange={update(setCanny)} /><Slider label="Low threshold" name="low_threshold" value={canny.low_threshold} min="0" max="500" onChange={update(setCanny)} /><Slider label="High threshold" name="high_threshold" value={canny.high_threshold} min="0" max="800" onChange={update(setCanny)} /><button className="run" disabled={!imageId || busy} onClick={runCanny}>{busy === "canny" ? "Running Canny..." : "Run Canny →"}</button></aside><div className="result-area"><div className="route-row"><div><h3>Input source</h3><p>Canny always starts from the original uploaded image.</p></div><select className="input-select" value="original" disabled><option value="original">Original image</option></select></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">INPUT</span>{original ? <img src={original} alt="Original Canny input" /> : <div className="viewer-empty">Upload an image</div>}</div><SlotCard slot="1" value={cannySlots["1"]} source={cannySlotSources["1"]} selected={cannySlot === "1"} onClick={() => setCannySlot("1")} /><SlotCard slot="2" value={cannySlots["2"]} source={cannySlotSources["2"]} selected={cannySlot === "2"} onClick={() => setCannySlot("2")} /></div></div></div></section>
    <StagePipeline title="Canny stages" steps={cannyResult?.steps} names={["grayscale", "gaussian_blur", "gradient_magnitude", "gradient_direction", "non_maximum_suppression", "edges"]} />

    <section className="algorithm-section hough-section"><div className="section-intro"><span className="section-number">02 / HOUGH LINE DETECTION</span><h2>Read the geometry.</h2><p>Choose a source matrix, vote in accumulator space, then render the strongest lines.</p></div><div className="algorithm-body"><aside className="controls"><h3>Parameters</h3><Slider label="Radius resolution" name="radius_res" value={hough.radius_res} min="0.5" max="10" step="0.5" onChange={update(setHough)} /><Slider label="Theta resolution" name="theta_res" value={hough.theta_res} min="0.5" max="20" step="0.5" onChange={update(setHough)} /><Slider label="Maximum lines" name="max_lines" value={hough.max_lines} min="1" max="50" onChange={update(setHough)} /><Slider label="Vote threshold" name="vote_threshold" value={hough.vote_threshold} min="0" max="500" onChange={update(setHough)} /><button className="run" disabled={!imageId || busy || (houghInput !== "original" && houghInput !== "canny" && !cannySlots[houghInput])} onClick={runHough}>{busy === "hough" ? "Running Hough..." : "Run Hough →"}</button></aside><div className="result-area"><div className="route-row"><div><h3>Input source</h3><p>Select the original or either Canny output slot.</p></div><select className="input-select" value={houghInput} onChange={(event) => setHoughInput(event.target.value)}><option value="original">Original</option><option value="1">Canny Output 1</option><option value="2">Canny Output 2</option><option value="canny">Latest Canny edges</option></select></div><div className="viewer-grid"><div className="viewer-card input-card"><span className="viewer-label">INPUT</span>{selectedInput ? <img src={selectedInput} alt="Selected Hough input" /> : <div className="viewer-empty">Select a populated source</div>}</div><SlotCard slot="1" value={houghSlots["1"]} source={houghSlotSources["1"]} selected={houghSlot === "1"} onClick={() => setHoughSlot("1")} /><SlotCard slot="2" value={houghSlots["2"]} source={houghSlotSources["2"]} selected={houghSlot === "2"} onClick={() => setHoughSlot("2")} /></div><div className="route-row output-route"><div><h3>Save Hough result to</h3><p>Choose the destination before running Hough.</p></div><div className="slot-toggle"><button className={houghSlot === "1" ? "chosen" : ""} onClick={() => setHoughSlot("1")}>Output 1</button><button className={houghSlot === "2" ? "chosen" : ""} onClick={() => setHoughSlot("2")}>Output 2</button></div></div></div></div></section>
    <StagePipeline title="Hough stages" steps={houghResult?.steps} names={["input_edges", "accumulator", "line_overlay"]} />
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);
