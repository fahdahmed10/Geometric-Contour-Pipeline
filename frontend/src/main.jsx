import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = "http://localhost:8000/api";
const initialCanny = { kernel_size: 7, sigma: 1, low_threshold: 80, high_threshold: 200 };
const initialHough = { radius_res: 1, theta_res: 1, max_lines: 10, vote_threshold: 20 };

function Slider({ label, name, value, min, max, step = 1, onChange }) {
  return <label className="control"><span>{label}<b>{value}</b></span><input type="range" name={name} value={value} min={min} max={max} step={step} onChange={onChange} /></label>;
}

function ImageCard({ title, src, selected, onClick }) {
  return <button className={`image-card ${selected ? "selected" : ""}`} onClick={onClick}><strong>{title}</strong>{src ? <img src={src} alt={title} /> : <div className="empty">No output yet</div>}</button>;
}

function StepViewer({ title, steps }) {
  if (!steps) return null;
  return <section className="steps"><h2>{title} stages</h2><div className="step-grid">{Object.entries(steps).map(([name, data]) => <figure key={name}><img src={data.image} alt={name} /><figcaption>{name.replaceAll("_", " ")}</figcaption></figure>)}</div></section>;
}

function App() {
  const [imageId, setImageId] = useState(null);
  const [original, setOriginal] = useState(null);
  const [canny, setCanny] = useState(initialCanny);
  const [hough, setHough] = useState(initialHough);
  const [houghInput, setHoughInput] = useState("canny");
  const [result, setResult] = useState(null);
  const [selectedOutput, setSelectedOutput] = useState("canny");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(""); setResult(null);
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

  async function process() {
    if (!imageId) return;
    setBusy(true); setError("");
    try {
      const response = await fetch(`${API}/images/${imageId}/pipeline`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ canny, hough, hough_input: houghInput }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Processing failed");
      setResult(data);
    } catch (caught) { setError(caught.message); } finally { setBusy(false); }
  }

  const update = (setter) => (event) => setter((state) => ({ ...state, [event.target.name]: Number(event.target.value) }));
  const output = result?.[selectedOutput]?.output;
  return <main>
    <header><div><p className="eyebrow">FROM-SCRATCH COMPUTER VISION</p><h1>Contour <em>lab</em></h1><p className="lede">Explore every matrix in a custom Canny and Hough pipeline.</p></div><label className="upload">Upload image<input type="file" accept="image/*" onChange={upload} /></label></header>
    {error && <div className="error">{error}</div>}
    <section className="workspace">
      <aside className="panel"><h2>Pipeline controls</h2><h3>01 / Canny edge detection</h3><Slider label="Kernel size" name="kernel_size" value={canny.kernel_size} min="3" max="15" step="2" onChange={update(setCanny)} /><Slider label="Gaussian sigma" name="sigma" value={canny.sigma} min="0.1" max="10" step="0.1" onChange={update(setCanny)} /><Slider label="Low threshold" name="low_threshold" value={canny.low_threshold} min="0" max="500" onChange={update(setCanny)} /><Slider label="High threshold" name="high_threshold" value={canny.high_threshold} min="0" max="800" onChange={update(setCanny)} /><h3>02 / Hough line detection</h3><Slider label="Radius resolution" name="radius_res" value={hough.radius_res} min="0.5" max="10" step="0.5" onChange={update(setHough)} /><Slider label="Theta resolution" name="theta_res" value={hough.theta_res} min="0.5" max="20" step="0.5" onChange={update(setHough)} /><Slider label="Maximum lines" name="max_lines" value={hough.max_lines} min="1" max="50" onChange={update(setHough)} /><Slider label="Vote threshold" name="vote_threshold" value={hough.vote_threshold} min="0" max="500" onChange={update(setHough)} /><label className="select">Hough input<select value={houghInput} onChange={(event) => setHoughInput(event.target.value)}><option value="canny">Canny output</option><option value="original">Original (grayscale)</option></select></label><button className="run" disabled={!imageId || busy} onClick={process}>{busy ? "Processing..." : "Run pipeline →"}</button></aside>
      <div className="canvas"><div className="canvas-head"><div><p className="eyebrow">OUTPUT ROUTING</p><h2>{imageId ? "Select an output to inspect" : "Start with an image"}</h2></div>{result?.hough?.lines?.length > 0 && <span className="badge">{result.hough.lines.length} lines detected</span>}</div><div className="image-grid"><ImageCard title="Original" src={original} selected={selectedOutput === "original"} onClick={() => setSelectedOutput("original")} /><ImageCard title="Output 1 · Canny" src={result?.canny?.output} selected={selectedOutput === "canny"} onClick={() => setSelectedOutput("canny")} /><ImageCard title="Output 2 · Hough input" src={result?.hough?.output} selected={selectedOutput === "hough"} onClick={() => setSelectedOutput("hough")} /></div>{output && <div className="focus"><span>Selected output</span><img src={output} alt="Selected pipeline output" /></div>}</div>
    </section>
    <StepViewer title="Canny" steps={result?.canny?.steps} /><StepViewer title="Hough" steps={result?.hough?.steps} />
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);
