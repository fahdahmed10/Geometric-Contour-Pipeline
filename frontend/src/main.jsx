import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = "http://localhost:8000/api";
const initialCanny = { kernel_size: 7, sigma: 1, low_threshold: 80, high_threshold: 200 };
const initialHough = { radius_res: 1, theta_res: 1, max_lines: 100, vote_threshold: 2 };
const initialEllipse = { a_min: 10, a_max: 150, vote_threshold: 40, fit_tol: 2, min_support: 0.5, max_ellipses: 3, ransac_iters: 300 };
const initialCircle = {
  cht: { min_diam: 150, max_diam: 260, min_votes: 80, min_coverage: 0.55, perim_frac: 0.45, smooth: 3, nms_win: 11, align_cos: 0.7, refine: true },
  ransac: { min_diam: 150, max_diam: 260, tol: 2, max_iter: 1000, min_inliers: 80, perim_frac: 0.45, min_coverage: 0.65, refine: true, seed: 0 },
};
const initialSnake = { alpha: 1, beta: 1, gamma: 1.2, n_points: 20, window_size: 5, max_iterations: 200 };

const TABS = [
  { id: "canny", num: "01", label: "Canny Edges" },
  { id: "hough", num: "02", label: "Hough Lines" },
  { id: "ellipse", num: "03", label: "Ellipse" },
  { id: "circle", num: "04", label: "Circle" },
  { id: "snake", num: "05", label: "Active Snake" },
];

// --- Persistent IndexedDB Storage (survives browser close & PC restart) ---
const DB_NAME = "contour_lab_persistence_v1";
const STORE_NAME = "kv";

function openStorageDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function dbGet(key) {
  try {
    const db = await openStorageDB();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const req = tx.objectStore(STORE_NAME).get(key);
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror = () => reject(req.error);
    });
  } catch {
    return null;
  }
}

async function dbSet(key, value) {
  try {
    const db = await openStorageDB();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      tx.objectStore(STORE_NAME).put(value, key);
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => reject(tx.error);
    });
  } catch {
    return false;
  }
}

async function dataUrlToFile(dataUrl, filename = "image.png") {
  const res = await fetch(dataUrl);
  const blob = await res.blob();
  return new File([blob], filename, { type: blob.type || "image/png" });
}

function formatDateTime(ts) {
  const d = new Date(ts);
  const datePart = d.toLocaleDateString(undefined, { month: "short", day: "2-digit", year: "numeric" });
  const timePart = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  return `${datePart} · ${timePart}`;
}

function downloadDataUrl(dataUrl, flag, ts) {
  const link = document.createElement("a");
  const safeFlag = (flag || "image").toLowerCase().replace(/[^a-z0-9]+/g, "-");
  link.href = dataUrl;
  link.download = `contour-lab-${safeFlag}-${ts}.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function Slider({ label, name, value, min, max, step = 1, onChange, suffix = "" }) {
  return (
    <label className="control">
      <span>
        {label}
        <b>{value}{suffix}</b>
      </span>
      <input type="range" name={name} value={value} min={min} max={max} step={step} onChange={onChange} />
    </label>
  );
}

function PillGroup({ label, options, value, onChange }) {
  return (
    <div className="pill-row">
      {label && <span className="pill-label">{label}</span>}
      <div className="pill-group">
        {options.map((opt) => (
          <button
            key={opt.value}
            type="button"
            className={value === opt.value ? "pill active" : "pill"}
            disabled={opt.disabled}
            onClick={() => onChange(opt.value)}
            title={opt.title || opt.label}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function SlotCard({ slot, value, source, onClick, selected, summary = null }) {
  return (
    <button type="button" className={`slot-card ${value ? "filled" : ""} ${selected ? "active" : ""}`} onClick={onClick}>
      <div className="card-header">
        <span className="slot-index">OUTPUT {slot} {selected ? "· ACTIVE TARGET" : ""}</span>
        <span className="slot-label">{source || "Click to target slot"}</span>
      </div>
      <div className="card-media">
        {value ? <img src={value} alt={`${source} in Output ${slot}`} /> : <span className="slot-empty">Click Run to populate Output {slot}</span>}
      </div>
      {summary}
    </button>
  );
}

function HistoryWindow({ items, onViewProcess, onUseAsInput, onClearHistory }) {
  const [confirmClear, setConfirmClear] = useState(false);
  const sortedItems = [...items].sort((a, b) => b.timestamp - a.timestamp);

  return (
    <section className="history-panel" aria-label="Session History">
      <div className="history-header">
        <div className="history-title-group">
          <h3>History</h3>
          <span className="history-count">{sortedItems.length}</span>
        </div>
        {sortedItems.length > 0 && !confirmClear && (
          <button
            type="button"
            className="history-clear-btn"
            onClick={() => setConfirmClear(true)}
            title="Clear all saved history"
          >
            Clear history
          </button>
        )}
      </div>

      {confirmClear && (
        <div className="history-confirm-bar">
          <span>Are you sure?</span>
          <div className="history-confirm-actions">
            <button
              type="button"
              className="confirm-yes"
              onClick={() => {
                onClearHistory();
                setConfirmClear(false);
              }}
            >
              Yes
            </button>
            <button
              type="button"
              className="confirm-no"
              onClick={() => setConfirmClear(false)}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="history-scroll">
        {sortedItems.length === 0 ? (
          <div className="history-empty">Uploaded inputs & processed outputs will appear here</div>
        ) : (
          sortedItems.map((item) => (
            <div
              key={item.id}
              className={`history-card ${item.kind === "input" ? "is-input" : "is-output"}`}
              onClick={() => {
                if (item.kind === "output") onViewProcess(item);
                else onUseAsInput(item);
              }}
            >
              <img src={item.image} alt={item.flag} loading="lazy" />
              <span className={`history-flag ${item.kind === "input" ? "flag-input" : "flag-output"}`}>
                {item.flag}
              </span>

              <div className="history-hover-overlay" onClick={(e) => e.stopPropagation()}>
                <time className="history-date">{formatDateTime(item.timestamp)}</time>
                <div className="history-actions">
                  {item.kind === "output" && (
                    <button
                      type="button"
                      className="history-icon-btn"
                      onClick={() => onViewProcess(item)}
                      aria-label="View process again"
                    >
                      <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                      <span className="icon-tooltip">View process again</span>
                    </button>
                  )}
                  <button
                    type="button"
                    className="history-icon-btn"
                    onClick={() => onUseAsInput(item)}
                    aria-label="Use as input"
                  >
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M15 3h6v6" />
                      <path d="M10 14L21 3" />
                      <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5" />
                    </svg>
                    <span className="icon-tooltip">Use as input</span>
                  </button>
                  <button
                    type="button"
                    className="history-icon-btn"
                    onClick={() => downloadDataUrl(item.image, item.flag, item.timestamp)}
                    aria-label="Download"
                  >
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="7 10 12 15 17 10" />
                      <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    <span className="icon-tooltip">Download</span>
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );
}

function StagePipeline({ title, steps, names, selection, onSelectionChange, options }) {
  return (
    <section className="stage-panel">
      <div className="panel-toolbar">
        <div>
          <span className="eyebrow">STAGE PIPELINE</span>
          <h2>{title}</h2>
        </div>
        <PillGroup label="Source" options={options} value={selection} onChange={onSelectionChange} />
      </div>
      <div className="stage-grid" style={{ gridTemplateColumns: `repeat(${Math.min(names.length, 3)}, minmax(0, 1fr))` }}>
        {names.map((name) => {
          const step = steps?.[name];
          return (
            <figure className={step ? "stage available" : "stage"} key={name}>
              <div className="stage-media">
                {step ? <img src={step.image} alt={name} /> : <div className="stage-placeholder">Run algorithm to view stage</div>}
              </div>
              <figcaption>{name.replaceAll("_", " ")}</figcaption>
            </figure>
          );
        })}
      </div>
    </section>
  );
}

function ComparisonPanel({ title, custom, opencv, metric, slot, onSlotChange, extraControls = null, customSummary = null, opencvSummary = null }) {
  return (
    <section className="comparison-panel">
      <div className="panel-toolbar">
        <div>
          <span className="eyebrow">REFERENCE CHECK · {metric}</span>
          <h2>{title} vs OpenCV</h2>
        </div>
        <PillGroup
          label="Compare Slot"
          options={[
            { value: "1", label: "Output 1" },
            { value: "2", label: "Output 2" },
          ]}
          value={slot}
          onChange={onSlotChange}
        />
      </div>
      {extraControls}
      {custom && opencv ? (
        <div className="comparison-grid">
          <figure>
            <div className="comparison-media">
              <img src={custom} alt={`${title} custom implementation`} />
            </div>
            <figcaption>Custom from scratch</figcaption>
            {customSummary}
          </figure>
          <figure>
            <div className="comparison-media">
              <img src={opencv} alt={`${title} OpenCV implementation`} />
            </div>
            <figcaption>OpenCV reference</figcaption>
            {opencvSummary}
          </figure>
        </div>
      ) : (
        <div className="comparison-empty">
          <p>No comparison data in Output {slot} yet. Click <b>Run</b> in the left panel to generate both Custom and OpenCV results.</p>
        </div>
      )}
    </section>
  );
}

function ParameterList({ parameters }) {
  if (!parameters) return null;
  return (
    <div className="circle-parameter-list">
      {Object.entries(parameters).map(([name, value]) => (
        <span key={name}>
          <b>{name}</b>
          <em>{typeof value === "number" ? Number(value.toFixed(3)) : String(value)}</em>
        </span>
      ))}
    </div>
  );
}

function CircleSummaryBlock({ label, title, count, parameters, className = "" }) {
  return (
    <div className={`circle-inline-summary ${className}`}>
      <div className="circle-inline-heading">
        <div>
          <span className="eyebrow">{label}</span>
          <h3>{title}</h3>
        </div>
        <strong>
          {count}
          <small>circles</small>
        </strong>
      </div>
      <ParameterList parameters={parameters} />
    </div>
  );
}

function LinkedCircleOpenCVPanel({ algorithm, parameters }) {
  return (
    <div className="opencv-linked-panel">
      <span className="eyebrow">LINKED OPENCV PARAMETERS</span>
      <p>Derived from the selected {algorithm === "cht" ? "CHT" : "RANSAC 3-point"} settings and Canny output.</p>
      <ParameterList parameters={parameters} />
    </div>
  );
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
    return () => {
      image.onload = null;
    };
  }, [source, aMin, aMax]);

  return source ? (
    <canvas ref={canvasRef} role="img" aria-label={`Axis reference guides: a_min ${aMin} pixels and a_max ${aMax} pixels`} />
  ) : (
    <div className="viewer-empty">Click to select/upload an image</div>
  );
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

      const minRadius = (0.85 * minDiam) / 2;
      const maxRadius = (1.15 * maxDiam) / 2;
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
    return () => {
      image.onload = null;
    };
  }, [source, minDiam, maxDiam]);

  return source ? (
    <canvas ref={canvasRef} role="img" aria-label={`Circle diameter guides: minimum ${minDiam} pixels and maximum ${maxDiam} pixels`} />
  ) : (
    <div className="viewer-empty">Click to select/upload an image</div>
  );
}

function resamplePolygon(points, targetCount) {
  if (!points || points.length === 0) return [];
  if (points.length === targetCount) return points;
  if (points.length < 3) return points;
  const closed = [...points, points[0]];
  const segLengths = [];
  let totalLength = 0;
  for (let i = 0; i < points.length; i += 1) {
    const dx = closed[i + 1].x - closed[i].x;
    const dy = closed[i + 1].y - closed[i].y;
    const len = Math.hypot(dx, dy);
    segLengths.push(len);
    totalLength += len;
  }
  if (totalLength === 0) return Array.from({ length: targetCount }, () => ({ ...points[0] }));
  const step = totalLength / targetCount;
  const result = [];
  let segIndex = 0;
  let segAccum = 0;
  for (let i = 0; i < targetCount; i += 1) {
    const targetDist = i * step;
    while (segIndex < segLengths.length - 1 && segAccum + segLengths[segIndex] < targetDist) {
      segAccum += segLengths[segIndex];
      segIndex += 1;
    }
    const segLen = segLengths[segIndex] || 1;
    const t = Math.min(1, Math.max(0, (targetDist - segAccum) / segLen));
    const p0 = closed[segIndex];
    const p1 = closed[segIndex + 1];
    result.push({
      x: p0.x + t * (p1.x - p0.x),
      y: p0.y + t * (p1.y - p0.y),
    });
  }
  return result;
}

function makePresetContour(type, width, height, nPoints) {
  const cx = width / 2;
  const cy = height / 2;
  const rx = width * 0.34;
  const ry = height * 0.34;
  if (type === "box") {
    const corners = [
      { x: cx - rx, y: cy - ry },
      { x: cx + rx, y: cy - ry },
      { x: cx + rx, y: cy + ry },
      { x: cx - rx, y: cy + ry },
    ];
    return resamplePolygon(corners, nPoints);
  }
  const r = Math.min(rx, ry);
  return Array.from({ length: nPoints }, (_, idx) => {
    const angle = (2 * Math.PI * idx) / nPoints;
    return {
      x: cx + r * Math.cos(angle),
      y: cy + r * Math.sin(angle),
    };
  });
}

function SnakeContourInput({ source, points, onAddPoint, onImageLoaded }) {
  const containerRef = useRef(null);
  const imageRef = useRef(null);
  const [imageBox, setImageBox] = useState(null);

  const updateImageBox = () => {
    const container = containerRef.current;
    const image = imageRef.current;
    if (!container || !image?.naturalWidth || !image.naturalHeight) return;
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight || 320;
    const scale = Math.min(containerWidth / image.naturalWidth, containerHeight / image.naturalHeight);
    const width = image.naturalWidth * scale;
    const height = image.naturalHeight * scale;
    setImageBox({ left: (containerWidth - width) / 2, top: (containerHeight - height) / 2, width, height });
    if (onImageLoaded) onImageLoaded({ width: image.naturalWidth, height: image.naturalHeight });
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
      x: (x * imageRef.current.naturalWidth) / imageBox.width,
      y: (y * imageRef.current.naturalHeight) / imageBox.height,
    });
  };

  const closedPoints = points.length >= 3 ? [...points, points[0]] : points;

  return (
    <div ref={containerRef} className="snake-contour-input" onClick={handleClick}>
      {source ? (
        <>
          <img
            ref={imageRef}
            src={source}
            onLoad={updateImageBox}
            alt="Click around the object to initialize the Snake contour"
          />
          {imageBox && (
            <svg
              className="snake-contour-overlay"
              style={{ left: imageBox.left, top: imageBox.top, width: imageBox.width, height: imageBox.height }}
              viewBox={`0 0 ${imageRef.current.naturalWidth} ${imageRef.current.naturalHeight}`}
              aria-hidden="true"
            >
              {closedPoints.length > 1 && (
                <polyline points={closedPoints.map((point) => `${point.x},${point.y}`).join(" ")} />
              )}
              {points.map((point, index) => (
                <circle
                  key={`${point.x}-${point.y}-${index}`}
                  cx={point.x}
                  cy={point.y}
                  r={Math.max(3, imageRef.current.naturalWidth / 180)}
                />
              ))}
            </svg>
          )}
        </>
      ) : (
        <div className="viewer-empty">Click "Select Image" button above to load an image</div>
      )}
    </div>
  );
}

function SnakeEvolutionPlayer({ source, frames, showVertices, timings, totalMs }) {
  const canvasRef = useRef(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [image, setImage] = useState(null);

  useEffect(() => {
    if (!source) {
      setImage(null);
      return undefined;
    }
    const nextImage = new Image();
    nextImage.onload = () => setImage(nextImage);
    nextImage.src = source;
    return () => {
      nextImage.onload = null;
    };
  }, [source]);

  useEffect(() => {
    setFrameIndex(0);
    if (!frames?.length) return undefined;
    const timer = window.setInterval(() => {
      setFrameIndex((current) => (current + 1) % frames.length);
    }, 120);
    return () => window.clearInterval(timer);
  }, [frames]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const frame = frames?.[frameIndex];
    if (!canvas || !image || !frame?.contour?.length) return;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    context.beginPath();
    frame.contour.forEach((point, pointIndex) => {
      if (pointIndex === 0) context.moveTo(point.x, point.y);
      else context.lineTo(point.x, point.y);
    });
    context.closePath();
    context.strokeStyle = "#7dd3c0";
    context.lineWidth = Math.max(2, image.naturalWidth / 300);
    context.stroke();
    if (showVertices) {
      context.fillStyle = "#ffffff";
      frame.contour.forEach((point) => {
        context.beginPath();
        context.arc(point.x, point.y, Math.max(2, image.naturalWidth / 180), 0, Math.PI * 2);
        context.fill();
      });
    }
  }, [image, frames, frameIndex, showVertices]);

  if (!source || !frames?.length) {
    return (
      <div className="comparison-empty">
        <p>Run Snake first to watch the active contour evolution playback and stage timings.</p>
      </div>
    );
  }

  const frame = frames[frameIndex];
  const entries = timings ? Object.entries(timings).filter(([name]) => name !== "Timing") : [];

  return (
    <section className="snake-evolution-panel">
      <div className="evolution-layout">
        <figure className="snake-evolution-player">
          <div className="evolution-canvas-wrap">
            <canvas ref={canvasRef} aria-label={`Snake contour at iteration ${frame?.iteration ?? 0}`} />
          </div>
          <figcaption>
            <span>Frame {frameIndex + 1} / {frames.length}</span>
            <b>Iteration {frame?.iteration ?? 0}</b>
          </figcaption>
        </figure>
        <div className="snake-timing-panel">
          <span className="eyebrow">PERFORMANCE</span>
          <h3>Stage Timings</h3>
          <div className="snake-timing-grid">
            {entries.map(([name, duration]) => (
              <span key={name}>
                <b>{name}</b>
                <em>{Number(duration).toFixed(1)} ms</em>
              </span>
            ))}
          </div>
          {typeof totalMs === "number" && (
            <p className="total-time">
              Total Snake processing: <strong>{(totalMs / 1000).toFixed(2)} s</strong>
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

function renderSnakeOverlayDataUrl(originalSrc, snakeRes, thickness, verticesFlag) {
  return new Promise((resolve) => {
    if (!originalSrc || !snakeRes) {
      resolve(snakeRes?.output || null);
      return;
    }
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
        points.forEach((point, index) => (index ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y)));
        context.closePath();
        context.strokeStyle = color;
        context.lineWidth = width;
        context.stroke();
      };
      drawContour(snakeRes.initial_contour, "#f4be60", Math.max(1, thickness - 1));
      drawContour(snakeRes.final_contour, "#7dd3c0", thickness);
      if (verticesFlag) {
        context.fillStyle = "#ffffff";
        snakeRes.final_contour?.forEach((point) => {
          context.beginPath();
          context.arc(point.x, point.y, Math.max(2, thickness), 0, Math.PI * 2);
          context.fill();
        });
      }
      resolve(canvas.toDataURL("image/png"));
    };
    image.onerror = () => resolve(snakeRes?.output || null);
    image.src = originalSrc;
  });
}

function App() {
  const fileInputRef = useRef(null);
  const liveComparisonRequestRef = useRef(0);
  const [hydrated, setHydrated] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem("contour-lab-theme") || "dark");
  const [activeTab, setActiveTab] = useState("canny");
  const [subTab, setSubTab] = useState("io");
  const [imageId, setImageId] = useState(null);
  const [original, setOriginal] = useState(null);
  const [imageDims, setImageDims] = useState({ width: 400, height: 400 });

  const [canny, setCanny] = useState(initialCanny);
  const [hough, setHough] = useState(initialHough);
  const [ellipse, setEllipse] = useState(initialEllipse);
  const [circle, setCircle] = useState(initialCircle);
  const [snake, setSnake] = useState(initialSnake);
  const [displayThickness, setDisplayThickness] = useState(3);
  const [showVertices, setShowVertices] = useState(false);
  const [showEllipseGuides, setShowEllipseGuides] = useState(false);
  const [snakePoints, setSnakePoints] = useState([]);

  const [circleAlgorithm, setCircleAlgorithm] = useState("cht");
  const [cannySlot, setCannySlot] = useState("1");
  const [cannyComparisonSlot, setCannyComparisonSlot] = useState("1");
  const [houghInput, setHoughInput] = useState("canny");
  const [houghSlot, setHoughSlot] = useState("1");
  const [houghComparisonSlot, setHoughComparisonSlot] = useState("1");
  const [ellipseInput, setEllipseInput] = useState("canny");
  const [ellipseSlot, setEllipseSlot] = useState("1");
  const [ellipseComparisonSlot, setEllipseComparisonSlot] = useState("1");
  const [circleInput, setCircleInput] = useState("canny");
  const [circleSlot, setCircleSlot] = useState("1");
  const [circleComparisonSlot, setCircleComparisonSlot] = useState("1");
  const [snakeSlot, setSnakeSlot] = useState("1");

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

  const [historyItems, setHistoryItems] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [compareEllipse] = useState(true);
  const [compareCircle] = useState(true);

  useEffect(() => {
    document.documentElement.classList.toggle("light", theme === "light");
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("contour-lab-theme", theme);
  }, [theme]);

  // Restore saved history & latest workspace state on app startup
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [savedHistory, savedWorkspace] = await Promise.all([
        dbGet("history"),
        dbGet("workspace"),
      ]);
      if (cancelled) return;
      if (Array.isArray(savedHistory)) {
        setHistoryItems(savedHistory);
      }
      if (savedWorkspace && typeof savedWorkspace === "object") {
        if (savedWorkspace.activeTab) setActiveTab(savedWorkspace.activeTab);
        setSubTab("io");
        if (savedWorkspace.original) setOriginal(savedWorkspace.original);
        if (savedWorkspace.imageDims) setImageDims(savedWorkspace.imageDims);
        if (savedWorkspace.canny) setCanny(savedWorkspace.canny);
        if (savedWorkspace.hough) setHough(savedWorkspace.hough);
        if (savedWorkspace.ellipse) setEllipse(savedWorkspace.ellipse);
        if (savedWorkspace.circle) setCircle(savedWorkspace.circle);
        if (savedWorkspace.snake) setSnake(savedWorkspace.snake);
        if (savedWorkspace.circleAlgorithm) setCircleAlgorithm(savedWorkspace.circleAlgorithm);
        if (savedWorkspace.displayThickness) setDisplayThickness(savedWorkspace.displayThickness);
        if (typeof savedWorkspace.showVertices === "boolean") setShowVertices(savedWorkspace.showVertices);
        if (Array.isArray(savedWorkspace.snakePoints)) setSnakePoints(savedWorkspace.snakePoints);
        if (savedWorkspace.cannySlot) setCannySlot(savedWorkspace.cannySlot);
        if (savedWorkspace.houghInput) setHoughInput(savedWorkspace.houghInput);
        if (savedWorkspace.houghSlot) setHoughSlot(savedWorkspace.houghSlot);
        if (savedWorkspace.ellipseInput) setEllipseInput(savedWorkspace.ellipseInput);
        if (savedWorkspace.ellipseSlot) setEllipseSlot(savedWorkspace.ellipseSlot);
        if (savedWorkspace.circleInput) setCircleInput(savedWorkspace.circleInput);
        if (savedWorkspace.circleSlot) setCircleSlot(savedWorkspace.circleSlot);
        if (savedWorkspace.snakeSlot) setSnakeSlot(savedWorkspace.snakeSlot);
        if (savedWorkspace.cannySlots) setCannySlots(savedWorkspace.cannySlots);
        if (savedWorkspace.cannySlotSources) setCannySlotSources(savedWorkspace.cannySlotSources);
        if (savedWorkspace.houghSlots) setHoughSlots(savedWorkspace.houghSlots);
        if (savedWorkspace.houghSlotSources) setHoughSlotSources(savedWorkspace.houghSlotSources);
        if (savedWorkspace.ellipseSlots) setEllipseSlots(savedWorkspace.ellipseSlots);
        if (savedWorkspace.ellipseSlotSources) setEllipseSlotSources(savedWorkspace.ellipseSlotSources);
        if (savedWorkspace.circleSlots) setCircleSlots(savedWorkspace.circleSlots);
        if (savedWorkspace.circleSlotSources) setCircleSlotSources(savedWorkspace.circleSlotSources);
        if (savedWorkspace.snakeSlots) setSnakeSlots(savedWorkspace.snakeSlots);
        if (savedWorkspace.snakeSlotSources) setSnakeSlotSources(savedWorkspace.snakeSlotSources);
        if (savedWorkspace.cannyResult) setCannyResult(savedWorkspace.cannyResult);
        if (savedWorkspace.houghResult) setHoughResult(savedWorkspace.houghResult);
        if (savedWorkspace.ellipseResult) setEllipseResult(savedWorkspace.ellipseResult);
        if (savedWorkspace.circleResult) setCircleResult(savedWorkspace.circleResult);
        if (savedWorkspace.snakeResult) setSnakeResult(savedWorkspace.snakeResult);
        if (savedWorkspace.snakeMetrics) setSnakeMetrics(savedWorkspace.snakeMetrics);
        if (savedWorkspace.cannyRuns) setCannyRuns(savedWorkspace.cannyRuns);
        if (savedWorkspace.houghRuns) setHoughRuns(savedWorkspace.houghRuns);
        if (savedWorkspace.ellipseRuns) setEllipseRuns(savedWorkspace.ellipseRuns);
        if (savedWorkspace.circleRuns) setCircleRuns(savedWorkspace.circleRuns);
        if (savedWorkspace.cannyComparisonSlot) setCannyComparisonSlot(savedWorkspace.cannyComparisonSlot);
        if (savedWorkspace.houghComparisonSlot) setHoughComparisonSlot(savedWorkspace.houghComparisonSlot);
        if (savedWorkspace.ellipseComparisonSlot) setEllipseComparisonSlot(savedWorkspace.ellipseComparisonSlot);
        if (savedWorkspace.circleComparisonSlot) setCircleComparisonSlot(savedWorkspace.circleComparisonSlot);
        if (savedWorkspace.cannyPipelineSelection) setCannyPipelineSelection(savedWorkspace.cannyPipelineSelection);
        if (savedWorkspace.houghPipelineSelection) setHoughPipelineSelection(savedWorkspace.houghPipelineSelection);
        if (savedWorkspace.ellipsePipelineSelection) setEllipsePipelineSelection(savedWorkspace.ellipsePipelineSelection);
        if (savedWorkspace.circlePipelineSelection) setCirclePipelineSelection(savedWorkspace.circlePipelineSelection);
      }
      setHydrated(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Persist history whenever it changes
  useEffect(() => {
    if (!hydrated) return;
    dbSet("history", historyItems);
  }, [historyItems, hydrated]);

  // Persist workspace state whenever it changes
  useEffect(() => {
    if (!hydrated) return;
    dbSet("workspace", {
      activeTab,
      original,
      imageDims,
      canny,
      hough,
      ellipse,
      circle,
      snake,
      circleAlgorithm,
      displayThickness,
      showVertices,
      snakePoints,
      cannySlot,
      houghInput,
      houghSlot,
      ellipseInput,
      ellipseSlot,
      circleInput,
      circleSlot,
      snakeSlot,
      cannySlots,
      cannySlotSources,
      houghSlots,
      houghSlotSources,
      ellipseSlots,
      ellipseSlotSources,
      circleSlots,
      circleSlotSources,
      snakeSlots,
      snakeSlotSources,
      cannyResult,
      houghResult,
      ellipseResult,
      circleResult,
      snakeResult,
      snakeMetrics,
      cannyRuns,
      houghRuns,
      ellipseRuns,
      circleRuns,
      cannyComparisonSlot,
      houghComparisonSlot,
      ellipseComparisonSlot,
      circleComparisonSlot,
      cannyPipelineSelection,
      houghPipelineSelection,
      ellipsePipelineSelection,
      circlePipelineSelection,
    });
  }, [
    hydrated,
    activeTab,
    original,
    imageDims,
    canny,
    hough,
    ellipse,
    circle,
    snake,
    circleAlgorithm,
    displayThickness,
    showVertices,
    snakePoints,
    cannySlot,
    houghInput,
    houghSlot,
    ellipseInput,
    ellipseSlot,
    circleInput,
    circleSlot,
    snakeSlot,
    cannySlots,
    cannySlotSources,
    houghSlots,
    houghSlotSources,
    ellipseSlots,
    ellipseSlotSources,
    circleSlots,
    circleSlotSources,
    snakeSlots,
    snakeSlotSources,
    cannyResult,
    houghResult,
    ellipseResult,
    circleResult,
    snakeResult,
    snakeMetrics,
    cannyRuns,
    houghRuns,
    ellipseRuns,
    circleRuns,
    cannyComparisonSlot,
    houghComparisonSlot,
    ellipseComparisonSlot,
    circleComparisonSlot,
    cannyPipelineSelection,
    houghPipelineSelection,
    ellipsePipelineSelection,
    circlePipelineSelection,
  ]);

  // Change 1: Always select Input / Output Slots ("io") by default when clicking any main tab
  const selectMainTab = (tabId) => {
    setActiveTab(tabId);
    setSubTab("io");
  };

  // Trigger file picker dialog when clicking input slot (except Snake canvas)
  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  // Add or deduplicate input image in history
  const recordInputHistory = (imgDataUrl, dims) => {
    if (!imgDataUrl) return;
    setHistoryItems((prev) => {
      const existingIndex = prev.findIndex((item) => item.kind === "input" && item.image === imgDataUrl);
      if (existingIndex !== -1) {
        // Do not duplicate input image that had multiple processes; keep single entry
        return prev;
      }
      const entry = {
        id: `in-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        kind: "input",
        flag: "input",
        image: imgDataUrl,
        dims,
        timestamp: Date.now(),
      };
      return [entry, ...prev];
    });
  };

  // Add output image and full process snapshot to history
  const recordOutputHistory = (entry) => {
    setHistoryItems((prev) => [
      {
        id: `out-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        kind: "output",
        timestamp: Date.now(),
        ...entry,
      },
      ...prev,
    ]);
  };

  useEffect(() => {
    if (!original || !snakeResult) return undefined;
    let active = true;
    renderSnakeOverlayDataUrl(original, snakeResult, displayThickness, showVertices).then((output) => {
      if (active && output) {
        setSnakeSlots((state) => ({ ...state, [snakeSlot]: output }));
      }
    });
    return () => {
      active = false;
    };
  }, [original, snakeResult, displayThickness, showVertices, snakeSlot]);

  const update = (setter) => (event) => setter((state) => ({ ...state, [event.target.name]: Number(event.target.value) }));
  const updateCircle = (event) =>
    setCircle((state) => ({
      ...state,
      [circleAlgorithm]: { ...state[circleAlgorithm], [event.target.name]: Number(event.target.value) },
    }));
  const toggleCircleRefine = () =>
    setCircle((state) => ({
      ...state,
      [circleAlgorithm]: { ...state[circleAlgorithm], refine: !state[circleAlgorithm].refine },
    }));

  // Ensure the current `original` image is registered in the backend (even after app/PC restart)
  async function ensureBackendSession(overrideDataUrl = null) {
    const src = overrideDataUrl || original;
    if (!src) return null;
    if (imageId && !overrideDataUrl) {
      const check = await fetch(`${API}/images/${imageId}`);
      if (check.ok) return imageId;
    }
    const file = await dataUrlToFile(src, "restored-input.png");
    const form = new FormData();
    form.append("file", file);
    const uploadResponse = await fetch(`${API}/images`, { method: "POST", body: form });
    const data = await uploadResponse.json();
    if (!uploadResponse.ok) throw new Error(data.detail || "Failed to initialize image session");
    setImageId(data.image_id);
    return data.image_id;
  }

  async function executeCanny(targetImageId, targetSlot = cannySlot, cannyParams = canny, recordInHistory = false, origDataUrl = original, dims = imageDims) {
    const response = await fetch(`${API}/images/${targetImageId}/canny`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...cannyParams, output_slot: targetSlot }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Canny processing failed");
    setCannyResult(data);
    setCannyRuns((state) => ({ ...state, [targetSlot]: data }));
    setCannySlots((state) => ({ ...state, [targetSlot]: data.output }));
    setCannySlotSources((state) => ({ ...state, [targetSlot]: "Canny edges" }));
    if (recordInHistory) {
      recordOutputHistory({
        processTab: "canny",
        flag: "Canny Edges",
        image: data.output,
        slot: targetSlot,
        originalImage: origDataUrl,
        imageDims: dims,
        parameters: { ...cannyParams },
        runData: data,
      });
    }
    return data;
  }

  async function loadNewInputFile(file) {
    setError("");
    setCannyResult(null);
    setHoughResult(null);
    setEllipseResult(null);
    setCircleResult(null);
    setCannyRuns({});
    setHoughRuns({});
    setEllipseRuns({});
    setCircleRuns({});
    setCannySlots({ "1": null, "2": null });
    setCannySlotSources({ "1": null, "2": null });
    setHoughSlots({ "1": null, "2": null });
    setHoughSlotSources({ "1": null, "2": null });
    setEllipseSlots({ "1": null, "2": null });
    setEllipseSlotSources({ "1": null, "2": null });
    setCircleSlots({ "1": null, "2": null });
    setCircleSlotSources({ "1": null, "2": null });
    setSnakeSlots({ "1": null, "2": null });
    setSnakeSlotSources({ "1": null, "2": null });
    setSnakeResult(null);
    setSnakeMetrics(null);

    const form = new FormData();
    form.append("file", file);
    setBusy("upload");
    try {
      const uploadResponse = await fetch(`${API}/images`, { method: "POST", body: form });
      const data = await uploadResponse.json();
      if (!uploadResponse.ok) throw new Error(data.detail || "Upload failed");
      setImageId(data.image_id);
      const dims = { width: data.width || 400, height: data.height || 400 };
      setImageDims(dims);
      setSnakePoints(makePresetContour("circle", dims.width, dims.height, snake.n_points));
      const originalResponse = await fetch(`${API}/images/${data.image_id}`);
      const origDataUrl = (await originalResponse.json()).original;
      setOriginal(origDataUrl);
      recordInputHistory(origDataUrl, dims);
      await executeCanny(data.image_id, "1", canny, false, origDataUrl, dims);
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function upload(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    await loadNewInputFile(file);
  }

  // Use any history item (input or output) as the new input image
  async function handleUseAsInput(item) {
    if (!item?.image) return;
    setSubTab("io");
    const file = await dataUrlToFile(item.image, "history-input.png");
    await loadNewInputFile(file);
  }

  // Restore saved output process state WITHOUT re-processing
  function handleViewProcess(item) {
    if (!item || item.kind !== "output") return;
    setError("");
    const targetSlot = item.slot || "1";
    if (item.originalImage) {
      setOriginal(item.originalImage);
      setImageId(null); // Will transparently re-sync with backend only if user clicks Run later
    }
    if (item.imageDims) {
      setImageDims(item.imageDims);
    }

    setActiveTab(item.processTab);
    setSubTab("io");

    if (item.processTab === "canny") {
      if (item.parameters) setCanny(item.parameters);
      setCannySlot(targetSlot);
      setCannyComparisonSlot(targetSlot);
      setCannyPipelineSelection(targetSlot);
      if (item.runData) {
        setCannyResult(item.runData);
        setCannyRuns((state) => ({ ...state, [targetSlot]: item.runData }));
        setCannySlots((state) => ({ ...state, [targetSlot]: item.runData.output }));
        setCannySlotSources((state) => ({ ...state, [targetSlot]: "Canny edges" }));
      }
    } else if (item.processTab === "hough") {
      if (item.parameters) setHough(item.parameters);
      if (item.inputSource) setHoughInput(item.inputSource);
      setHoughSlot(targetSlot);
      setHoughComparisonSlot(targetSlot);
      setHoughPipelineSelection(targetSlot);
      if (item.cannySnapshot) {
        setCannyResult(item.cannySnapshot);
        setCannySlots((state) => ({ ...state, "1": item.cannySnapshot.output }));
      }
      if (item.runData) {
        setHoughResult(item.runData);
        setHoughRuns((state) => ({ ...state, [targetSlot]: item.runData }));
        setHoughSlots((state) => ({ ...state, [targetSlot]: item.runData.output }));
        setHoughSlotSources((state) => ({ ...state, [targetSlot]: "Hough lines" }));
      }
    } else if (item.processTab === "ellipse") {
      if (item.parameters) setEllipse(item.parameters);
      if (item.inputSource) setEllipseInput(item.inputSource);
      setEllipseSlot(targetSlot);
      setEllipseComparisonSlot(targetSlot);
      setEllipsePipelineSelection(targetSlot);
      if (item.cannySnapshot) {
        setCannyResult(item.cannySnapshot);
        setCannySlots((state) => ({ ...state, "1": item.cannySnapshot.output }));
      }
      if (item.runData) {
        setEllipseResult(item.runData);
        setEllipseRuns((state) => ({ ...state, [targetSlot]: item.runData }));
        setEllipseSlots((state) => ({ ...state, [targetSlot]: item.runData.output }));
        setEllipseSlotSources((state) => ({ ...state, [targetSlot]: "Ellipse detections" }));
      }
    } else if (item.processTab === "circle") {
      if (item.circleAlgorithm) setCircleAlgorithm(item.circleAlgorithm);
      if (item.parameters) setCircle(item.parameters);
      if (item.inputSource) setCircleInput(item.inputSource);
      setCircleSlot(targetSlot);
      setCircleComparisonSlot(targetSlot);
      setCirclePipelineSelection(targetSlot);
      if (item.cannySnapshot) {
        setCannyResult(item.cannySnapshot);
        setCannySlots((state) => ({ ...state, "1": item.cannySnapshot.output }));
      }
      if (item.runData) {
        setCircleResult(item.runData);
        setCircleRuns((state) => ({ ...state, [targetSlot]: item.runData }));
        setCircleSlots((state) => ({ ...state, [targetSlot]: item.runData.output }));
        setCircleSlotSources((state) => ({
          ...state,
          [targetSlot]: `${item.circleAlgorithm === "cht" ? "Circular Hough" : "RANSAC"} circles`,
        }));
      }
    } else if (item.processTab === "snake") {
      if (item.parameters?.snake) setSnake(item.parameters.snake);
      if (Array.isArray(item.parameters?.snakePoints)) setSnakePoints(item.parameters.snakePoints);
      if (item.parameters?.displayThickness) setDisplayThickness(item.parameters.displayThickness);
      if (typeof item.parameters?.showVertices === "boolean") setShowVertices(item.parameters.showVertices);
      setSnakeSlot(targetSlot);
      if (item.runData) {
        setSnakeResult(item.runData);
        setSnakeMetrics(item.runData.metrics);
        setSnakeSlots((state) => ({ ...state, [targetSlot]: item.image || item.runData.output }));
        setSnakeSlotSources((state) => ({ ...state, [targetSlot]: "Snake contour" }));
      }
    }
  }

  async function runCanny() {
    if (!original) return;
    setBusy("canny");
    setError("");
    try {
      const activeId = await ensureBackendSession();
      await executeCanny(activeId, cannySlot, canny, true, original, imageDims);
      setCannyComparisonSlot(cannySlot);
      setCannyPipelineSelection(cannySlot);
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function ensureCannyForSource(activeId, sourceKey) {
    if (sourceKey === "original") return true;
    const slotNum = sourceKey.replace("output", "");
    const targetSlot = slotNum === "2" ? "2" : "1";
    const data = await executeCanny(activeId, targetSlot, canny, false, original, imageDims);
    return Boolean(data);
  }

  async function runHough() {
    if (!original) return;
    setBusy("hough");
    setError("");
    try {
      const activeId = await ensureBackendSession();
      const inputSource = houghInput === "1" || houghInput === "2" ? `output${houghInput}` : houghInput;
      const ready = await ensureCannyForSource(activeId, inputSource);
      if (!ready) return;
      const response = await fetch(`${API}/images/${activeId}/hough`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...hough, input_source: inputSource, output_slot: houghSlot }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Hough processing failed");
      setHoughResult(data);
      setHoughRuns((state) => ({ ...state, [houghSlot]: data }));
      setHoughSlots((state) => ({ ...state, [houghSlot]: data.output }));
      setHoughSlotSources((state) => ({ ...state, [houghSlot]: "Hough lines" }));
      setHoughComparisonSlot(houghSlot);
      setHoughPipelineSelection(houghSlot);
      recordOutputHistory({
        processTab: "hough",
        flag: "Hough Lines",
        image: data.output,
        slot: houghSlot,
        inputSource: houghInput,
        originalImage: original,
        imageDims,
        cannySnapshot: cannyResult,
        parameters: { ...hough },
        runData: data,
      });
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function runEllipse() {
    if (!original || ellipse.a_min > ellipse.a_max) return;
    liveComparisonRequestRef.current += 1;
    setBusy("ellipse");
    setError("");
    try {
      const activeId = await ensureBackendSession();
      const ready = await ensureCannyForSource(activeId, ellipseInput);
      if (!ready) return;
      const response = await fetch(`${API}/images/${activeId}/ellipse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...ellipse,
          input_source: ellipseInput,
          output_slot: ellipseSlot,
          compare_opencv: compareEllipse,
          resize_input: true,
          canny,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Ellipse processing failed");
      setEllipseResult(data);
      setEllipseRuns((state) => ({ ...state, [ellipseSlot]: data }));
      setEllipseSlots((state) => ({ ...state, [ellipseSlot]: data.output }));
      setEllipseSlotSources((state) => ({ ...state, [ellipseSlot]: "Ellipse detections" }));
      setEllipseComparisonSlot(ellipseSlot);
      setEllipsePipelineSelection(ellipseSlot);
      recordOutputHistory({
        processTab: "ellipse",
        flag: "Ellipse",
        image: data.output,
        slot: ellipseSlot,
        inputSource: ellipseInput,
        originalImage: original,
        imageDims,
        cannySnapshot: cannyResult,
        parameters: { ...ellipse },
        runData: data,
      });
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function runCircle() {
    if (!original) return;
    liveComparisonRequestRef.current += 1;
    setBusy("circle");
    setError("");
    const selectedCircle = circle[circleAlgorithm];
    try {
      const activeId = await ensureBackendSession();
      const ready = await ensureCannyForSource(activeId, circleInput);
      if (!ready) return;
      const response = await fetch(`${API}/images/${activeId}/circle`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          algorithm: circleAlgorithm,
          ...selectedCircle,
          input_source: circleInput,
          output_slot: circleSlot,
          compare_opencv: compareCircle,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Circle processing failed");
      setCircleResult(data);
      setCircleRuns((state) => ({ ...state, [circleSlot]: data }));
      setCircleSlots((state) => ({ ...state, [circleSlot]: data.output }));
      setCircleSlotSources((state) => ({
        ...state,
        [circleSlot]: `${circleAlgorithm === "cht" ? "Circular Hough" : "RANSAC"} circles`,
      }));
      setCircleComparisonSlot(circleSlot);
      setCirclePipelineSelection(circleSlot);
      recordOutputHistory({
        processTab: "circle",
        flag: circleAlgorithm === "cht" ? "Circle (CHT)" : "Circle (RANSAC)",
        circleAlgorithm,
        image: data.output,
        slot: circleSlot,
        inputSource: circleInput,
        originalImage: original,
        imageDims,
        cannySnapshot: cannyResult,
        parameters: JSON.parse(JSON.stringify(circle)),
        runData: data,
      });
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  async function refreshEllipseComparison() {
    const slot = ellipseComparisonSlot;
    if (!original || !ellipseRuns[slot] || ellipse.a_min > ellipse.a_max) return;
    const requestId = ++liveComparisonRequestRef.current;
    try {
      const activeId = await ensureBackendSession();
      const payload = JSON.stringify({
        ...ellipse,
        input_source: ellipseInput,
        output_slot: slot,
        compare_opencv: true,
        resize_input: true,
        canny,
      });
      let response = await fetch(`${API}/images/${activeId}/ellipse/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
      });
      if (response.status === 409) {
        const ready = await ensureCannyForSource(activeId, ellipseInput);
        if (!ready) return;
        response = await fetch(`${API}/images/${activeId}/ellipse/compare`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
        });
      }
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Ellipse comparison failed");
      if (requestId !== liveComparisonRequestRef.current) return;
      setEllipseRuns((state) => (state[slot] ? { ...state, [slot]: { ...state[slot], ...data } } : state));
    } catch (caught) {
      if (requestId === liveComparisonRequestRef.current) setError(caught.message);
    }
  }

  async function refreshCircleComparison() {
    const slot = circleComparisonSlot;
    if (!original || !circleRuns[slot]) return;
    const requestId = ++liveComparisonRequestRef.current;
    try {
      const activeId = await ensureBackendSession();
      const payload = JSON.stringify({
        algorithm: circleAlgorithm,
        ...circle[circleAlgorithm],
        input_source: circleInput,
        output_slot: slot,
        compare_opencv: true,
      });
      let response = await fetch(`${API}/images/${activeId}/circle/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
      });
      if (response.status === 409) {
        const ready = await ensureCannyForSource(activeId, circleInput);
        if (!ready) return;
        response = await fetch(`${API}/images/${activeId}/circle/compare`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
        });
      }
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Circle comparison failed");
      if (requestId !== liveComparisonRequestRef.current) return;
      setCircleRuns((state) => (state[slot] ? { ...state, [slot]: { ...state[slot], ...data } } : state));
    } catch (caught) {
      if (requestId === liveComparisonRequestRef.current) setError(caught.message);
    }
  }

  useEffect(() => {
    if (!hydrated || activeTab !== "ellipse" || subTab !== "compare" || !original || !ellipseRuns[ellipseComparisonSlot]) return undefined;
    const timer = window.setTimeout(() => refreshEllipseComparison(), 250);
    return () => window.clearTimeout(timer);
  }, [hydrated, activeTab, subTab, original, ellipse, ellipseInput, ellipseComparisonSlot]);

  useEffect(() => {
    if (!hydrated || activeTab !== "circle" || subTab !== "compare" || !original || !circleRuns[circleComparisonSlot]) return undefined;
    const timer = window.setTimeout(() => refreshCircleComparison(), 250);
    return () => window.clearTimeout(timer);
  }, [hydrated, activeTab, subTab, original, circle[circleAlgorithm], circleInput, circleAlgorithm, circleComparisonSlot, cannyResult]);

  async function runSnake() {
    if (!original) return;
    setBusy("snake");
    setError("");
    try {
      const activeId = await ensureBackendSession();
      let contourToUse = snakePoints;
      if (contourToUse.length < 3) {
        contourToUse = makePresetContour("circle", imageDims.width, imageDims.height, snake.n_points);
        setSnakePoints(contourToUse);
      } else if (contourToUse.length !== snake.n_points) {
        contourToUse = resamplePolygon(contourToUse, snake.n_points);
        setSnakePoints(contourToUse);
      }
      const response = await fetch(`${API}/images/${activeId}/snake`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...snake, initial_contour: contourToUse, output_slot: snakeSlot }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Snake processing failed");
      setSnakeResult(data);
      setSnakeMetrics(data.metrics);
      const renderedOutput = await renderSnakeOverlayDataUrl(original, data, displayThickness, showVertices);
      setSnakeSlots((state) => ({ ...state, [snakeSlot]: renderedOutput || data.output }));
      setSnakeSlotSources((state) => ({ ...state, [snakeSlot]: "Snake contour" }));
      recordOutputHistory({
        processTab: "snake",
        flag: "Active Snake",
        image: renderedOutput || data.output,
        slot: snakeSlot,
        originalImage: original,
        imageDims,
        parameters: {
          snake: { ...snake },
          snakePoints: contourToUse,
          displayThickness,
          showVertices,
        },
        runData: data,
      });
    } catch (caught) {
      setError(caught.message);
    } finally {
      setBusy("");
    }
  }

  const hasImage = Boolean(original);
  const selectedInput =
    houghInput === "original" ? original : houghInput === "canny" ? cannyResult?.output || original : cannySlots[houghInput] || cannyResult?.output || original;
  const selectedEllipseInput =
    ellipseResult?.steps?.input_edges?.image ||
    (ellipseInput === "canny" ? cannyResult?.output : cannySlots[ellipseInput.replace("output", "")]) ||
    original;
  const selectedCannyRun = cannyPipelineSelection === "input" ? null : cannyRuns[cannyPipelineSelection];
  const selectedHoughRun = houghPipelineSelection === "input" ? null : houghRuns[houghPipelineSelection];
  const selectedEllipseRun = ellipsePipelineSelection === "input" ? null : ellipseRuns[ellipsePipelineSelection];
  const selectedCircleRun = circlePipelineSelection === "input" ? null : circleRuns[circlePipelineSelection];
  const cannyComparison = cannyRuns[cannyComparisonSlot];
  const houghComparison = houghRuns[houghComparisonSlot];
  const ellipseComparison = ellipseRuns[ellipseComparisonSlot];
  const circleComparison = circleRuns[circleComparisonSlot];
  const selectedCircleInput =
    (circleInput === "canny" ? cannyResult?.output : cannySlots[circleInput.replace("output", "")]) || original;
  const activeCircle = circle[circleAlgorithm];
  const linkedCircleOpenCV = (() => {
    const minRadius = 0.85 * activeCircle.min_diam / 2;
    const maxRadius = 1.15 * activeCircle.max_diam / 2;
    const evidence = circleAlgorithm === "cht" ? activeCircle.min_votes : activeCircle.min_inliers;
    const cannyParameters = cannyResult?.parameters || canny;
    return {
      dp: 1,
      minDist: Math.max(5, minRadius),
      param1: cannyParameters.high_threshold,
      param2: evidence,
      minRadius: Math.floor(minRadius),
      maxRadius: Math.ceil(maxRadius),
    };
  })();

  const customCircleSummary = (run) => {
    if (!run) return null;
    const parameters = run.custom_parameters || {};
    const compactNames =
      run.algorithm === "cht"
        ? ["min_diam", "max_diam", "min_votes", "min_coverage"]
        : ["min_diam", "max_diam", "tol", "min_inliers"];
    return (
      <CircleSummaryBlock
        className="slot-custom-summary"
        label="DETECTED"
        title={run.algorithm === "cht" ? "Circular Hough" : "RANSAC"}
        count={run.circles?.length ?? 0}
        parameters={Object.fromEntries(
          compactNames.filter((name) => parameters[name] !== undefined).map((name) => [name, parameters[name]])
        )}
      />
    );
  };

  return (
    <div className="app-shell">
      {/* Hidden global file input triggered by header upload button OR clicking input slot */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={upload}
      />

      <header className="topbar">
        <div className="brand">
          <span className="brand-dot" />
          <h1>
            Contour <em>lab</em>
          </h1>
        </div>

        <nav className="main-tabs" aria-label="Algorithm Tabs">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={activeTab === tab.id ? "main-tab active" : "main-tab"}
              onClick={() => selectMainTab(tab.id)}
            >
              <span className="tab-num">{tab.num}</span>
              <span>{tab.label}</span>
            </button>
          ))}
        </nav>

        <div className="header-actions">
          {original && (
            <div className="loaded-badge" title="Active source image">
              <img src={original} alt="Loaded thumbnail" />
              <span>{imageDims.width}×{imageDims.height}</span>
            </div>
          )}
          <button type="button" className="upload" onClick={openFilePicker}>
            {busy === "upload" ? "Uploading..." : hasImage ? "Change Image" : "+ Upload Image"}
          </button>
          <button
            className="theme-toggle"
            type="button"
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          >
            <span aria-hidden="true">{theme === "dark" ? "☼" : "☾"}</span>
          </button>
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button type="button" onClick={() => setError("")}>✕</button>
        </div>
      )}

      <div className="workspace">
        {/* LEFT SIDEBAR: COMPACT CONTROLS + HISTORY WINDOW */}
        <aside className="sidebar">
          {activeTab === "canny" && (
            <>
              <div className="sidebar-action">
                <button className="run" disabled={!hasImage || Boolean(busy)} onClick={runCanny}>
                  {busy === "canny" ? "Running Canny..." : `Run Canny → Slot ${cannySlot}`}
                </button>
                <PillGroup
                  label="Save To"
                  options={[
                    { value: "1", label: "Slot 1" },
                    { value: "2", label: "Slot 2" },
                  ]}
                  value={cannySlot}
                  onChange={setCannySlot}
                />
              </div>
              <div className="sidebar-controls">
                <h3>Canny Parameters</h3>
                <div className="control-grid two-col">
                  <Slider label="Kernel size" name="kernel_size" value={canny.kernel_size} min="3" max="15" step="2" onChange={update(setCanny)} />
                  <Slider label="Sigma" name="sigma" value={canny.sigma} min="0.1" max="10" step="0.1" onChange={update(setCanny)} />
                  <Slider label="Low thresh" name="low_threshold" value={canny.low_threshold} min="0" max="500" onChange={update(setCanny)} />
                  <Slider label="High thresh" name="high_threshold" value={canny.high_threshold} min="0" max="800" onChange={update(setCanny)} />
                </div>
              </div>
            </>
          )}

          {activeTab === "hough" && (
            <>
              <div className="sidebar-action">
                <button className="run" disabled={!hasImage || Boolean(busy)} onClick={runHough}>
                  {busy === "hough" ? "Running Hough..." : `Run Hough → Slot ${houghSlot}`}
                </button>
                <PillGroup
                  label="Input"
                  options={[
                    { value: "canny", label: "Auto Canny" },
                    { value: "1", label: "Slot 1" },
                    { value: "2", label: "Slot 2" },
                    { value: "original", label: "Original" },
                  ]}
                  value={houghInput}
                  onChange={setHoughInput}
                />
                <PillGroup
                  label="Save To"
                  options={[
                    { value: "1", label: "Slot 1" },
                    { value: "2", label: "Slot 2" },
                  ]}
                  value={houghSlot}
                  onChange={setHoughSlot}
                />
              </div>
              <div className="sidebar-controls">
                <h3>Hough Line Parameters</h3>
                <div className="control-grid two-col">
                  <Slider label="Radius res" name="radius_res" value={hough.radius_res} min="0.5" max="10" step="0.5" onChange={update(setHough)} />
                  <Slider label="Theta res" name="theta_res" value={hough.theta_res} min="0.5" max="20" step="0.5" onChange={update(setHough)} />
                  <Slider label="Max lines" name="max_lines" value={hough.max_lines} min="1" max="100" onChange={update(setHough)} />
                  <Slider label="Votes" name="vote_threshold" value={hough.vote_threshold} min="0" max="500" onChange={update(setHough)} />
                </div>
              </div>
            </>
          )}

          {activeTab === "ellipse" && (
            <>
              <div className="sidebar-action">
                <button
                  className="run"
                  disabled={!hasImage || Boolean(busy) || ellipse.a_min > ellipse.a_max}
                  onClick={runEllipse}
                >
                  {busy === "ellipse" ? "Running Ellipse..." : `Run Ellipse → Slot ${ellipseSlot}`}
                </button>
                {ellipse.a_min > ellipse.a_max && (
                  <p className="validation-message">a_min must be ≤ a_max.</p>
                )}
                <PillGroup
                  label="Input"
                  options={[
                    { value: "canny", label: "Auto Canny" },
                    { value: "output1", label: "Slot 1" },
                    { value: "output2", label: "Slot 2" },
                  ]}
                  value={ellipseInput}
                  onChange={setEllipseInput}
                />
                <PillGroup
                  label="Save To"
                  options={[
                    { value: "1", label: "Slot 1" },
                    { value: "2", label: "Slot 2" },
                  ]}
                  value={ellipseSlot}
                  onChange={setEllipseSlot}
                />
              </div>
              <div className="sidebar-controls">
                <div className="sidebar-heading-row">
                  <h3>Ellipse Parameters</h3>
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={showEllipseGuides}
                      onChange={(e) => setShowEllipseGuides(e.target.checked)}
                    />
                    Axis Guides
                  </label>
                </div>
                <div className="control-grid two-col">
                  <Slider label="a_min" name="a_min" value={ellipse.a_min} min="2" max="200" onChange={update(setEllipse)} suffix="px" />
                  <Slider label="a_max" name="a_max" value={ellipse.a_max} min="10" max="400" onChange={update(setEllipse)} suffix="px" />
                  <Slider label="Votes" name="vote_threshold" value={ellipse.vote_threshold} min="10" max="500" onChange={update(setEllipse)} />
                  <Slider label="Fit tol" name="fit_tol" value={ellipse.fit_tol} min="0.5" max="8" step="0.1" onChange={update(setEllipse)} suffix="px" />
                  <Slider label="Support" name="min_support" value={ellipse.min_support} min="0.1" max="1" step="0.05" onChange={update(setEllipse)} />
                  <Slider label="Max count" name="max_ellipses" value={ellipse.max_ellipses} min="1" max="8" onChange={update(setEllipse)} />
                </div>
                <Slider label="RANSAC iterations" name="ransac_iters" value={ellipse.ransac_iters} min="50" max="1500" step="50" onChange={update(setEllipse)} />
              </div>
            </>
          )}

          {activeTab === "circle" && (
            <>
              <div className="sidebar-action">
                <button
                  className="run"
                  disabled={
                    !hasImage ||
                    Boolean(busy) ||
                    activeCircle.min_diam > activeCircle.max_diam
                  }
                  onClick={runCircle}
                >
                  {busy === "circle" ? "Running Circle..." : `Run Circle → Slot ${circleSlot}`}
                </button>
                {activeCircle.min_diam > activeCircle.max_diam && (
                  <p className="validation-message">Min diameter must be ≤ max diameter.</p>
                )}
                <PillGroup
                  label="Method"
                  options={[
                    { value: "cht", label: "Circular Hough" },
                    { value: "ransac", label: "RANSAC 3-Pt" },
                  ]}
                  value={circleAlgorithm}
                  onChange={setCircleAlgorithm}
                />
                <PillGroup
                  label="Input"
                  options={[
                    { value: "canny", label: "Auto Canny" },
                    { value: "output1", label: "Slot 1" },
                    { value: "output2", label: "Slot 2" },
                  ]}
                  value={circleInput}
                  onChange={setCircleInput}
                />
                <PillGroup
                  label="Save To"
                  options={[
                    { value: "1", label: "Slot 1" },
                    { value: "2", label: "Slot 2" },
                  ]}
                  value={circleSlot}
                  onChange={setCircleSlot}
                />
              </div>
              <div className="sidebar-controls">
                <div className="sidebar-heading-row">
                  <h3>{circleAlgorithm === "cht" ? "CHT Parameters" : "RANSAC Parameters"}</h3>
                  <label className="inline-check">
                    <input type="checkbox" checked={activeCircle.refine} onChange={toggleCircleRefine} />
                    LS Refine
                  </label>
                </div>
                <div className="control-grid two-col">
                  <Slider label="Min ø" name="min_diam" value={activeCircle.min_diam} min="4" max="1000" onChange={updateCircle} suffix="px" />
                  <Slider label="Max ø" name="max_diam" value={activeCircle.max_diam} min="4" max="1000" onChange={updateCircle} suffix="px" />
                  {circleAlgorithm === "cht" ? (
                    <>
                      <Slider label="Min votes" name="min_votes" value={activeCircle.min_votes} min="5" max="1000" onChange={updateCircle} />
                      <Slider label="Coverage" name="min_coverage" value={activeCircle.min_coverage} min="0.1" max="1" step="0.05" onChange={updateCircle} />
                      <Slider label="Perim share" name="perim_frac" value={activeCircle.perim_frac} min="0" max="1" step="0.05" onChange={updateCircle} />
                      <Slider label="Smooth" name="smooth" value={activeCircle.smooth} min="1" max="9" step="2" onChange={updateCircle} />
                      <Slider label="NMS win" name="nms_win" value={activeCircle.nms_win} min="3" max="21" step="2" onChange={updateCircle} />
                      <Slider label="Align cos" name="align_cos" value={activeCircle.align_cos} min="0" max="0.95" step="0.05" onChange={updateCircle} />
                    </>
                  ) : (
                    <>
                      <Slider label="Tolerance" name="tol" value={activeCircle.tol} min="0.5" max="6" step="0.1" onChange={updateCircle} suffix="px" />
                      <Slider label="Iterations" name="max_iter" value={activeCircle.max_iter} min="100" max="10000" step="100" onChange={updateCircle} />
                      <Slider label="Min inliers" name="min_inliers" value={activeCircle.min_inliers} min="5" max="1000" onChange={updateCircle} />
                      <Slider label="Perim share" name="perim_frac" value={activeCircle.perim_frac} min="0" max="1" step="0.05" onChange={updateCircle} />
                      <Slider label="Coverage" name="min_coverage" value={activeCircle.min_coverage} min="0.1" max="1" step="0.05" onChange={updateCircle} />
                      <Slider label="Seed" name="seed" value={activeCircle.seed} min="0" max="99" onChange={updateCircle} />
                    </>
                  )}
                </div>
              </div>
            </>
          )}

          {activeTab === "snake" && (
            <>
              <div className="sidebar-action">
                <button className="run" disabled={!hasImage || Boolean(busy)} onClick={runSnake}>
                  {busy === "snake" ? "Running Snake..." : `Run Snake → Slot ${snakeSlot}`}
                </button>
                <div className="pill-row">
                  <span className="pill-label">Quick Init</span>
                  <div className="pill-group">
                    <button
                      type="button"
                      className="pill"
                      onClick={() => setSnakePoints(makePresetContour("circle", imageDims.width, imageDims.height, snake.n_points))}
                    >
                      Auto Circle
                    </button>
                    <button
                      type="button"
                      className="pill"
                      onClick={() => setSnakePoints(makePresetContour("box", imageDims.width, imageDims.height, snake.n_points))}
                    >
                      Auto Box
                    </button>
                    <button
                      type="button"
                      className="pill"
                      onClick={() => setSnakePoints((pts) => pts.slice(0, -1))}
                      disabled={!snakePoints.length}
                    >
                      Undo
                    </button>
                    <button
                      type="button"
                      className="pill"
                      onClick={() => setSnakePoints([])}
                      disabled={!snakePoints.length}
                    >
                      Clear
                    </button>
                  </div>
                </div>
                <PillGroup
                  label="Save To"
                  options={[
                    { value: "1", label: "Slot 1" },
                    { value: "2", label: "Slot 2" },
                  ]}
                  value={snakeSlot}
                  onChange={setSnakeSlot}
                />
              </div>
              <div className="sidebar-controls">
                <div className="sidebar-heading-row">
                  <h3>Snake Parameters ({snakePoints.length}/{snake.n_points})</h3>
                  <label className="inline-check">
                    <input
                      type="checkbox"
                      checked={showVertices}
                      onChange={(event) => setShowVertices(event.target.checked)}
                    />
                    Vertices
                  </label>
                </div>
                <div className="control-grid two-col">
                  <Slider
                    label="Samples"
                    name="n_points"
                    value={snake.n_points}
                    min="3"
                    max="100"
                    onChange={(e) => {
                      const nextN = Number(e.target.value);
                      setSnake((s) => ({ ...s, n_points: nextN }));
                      if (snakePoints.length >= 3) {
                        setSnakePoints((pts) => resamplePolygon(pts, nextN));
                      }
                    }}
                  />
                  <Slider label="Iterations" name="max_iterations" value={snake.max_iterations} min="1" max="500" onChange={update(setSnake)} />
                  <Slider label="Alpha" name="alpha" value={snake.alpha} min="0" max="5" step="0.1" onChange={update(setSnake)} />
                  <Slider label="Beta" name="beta" value={snake.beta} min="0" max="5" step="0.1" onChange={update(setSnake)} />
                  <Slider label="Gamma" name="gamma" value={snake.gamma} min="0" max="5" step="0.1" onChange={update(setSnake)} />
                  <Slider label="Window" name="window_size" value={snake.window_size} min="3" max="11" step="2" onChange={update(setSnake)} />
                  <Slider
                    label="Thickness"
                    name="displayThickness"
                    value={displayThickness}
                    min="1"
                    max="8"
                    onChange={(event) => setDisplayThickness(Number(event.target.value))}
                  />
                </div>
              </div>
            </>
          )}

          {/* Change 3: Small scrolling History window directly under Parameters */}
          <HistoryWindow
            items={historyItems}
            onViewProcess={handleViewProcess}
            onUseAsInput={handleUseAsInput}
            onClearHistory={() => setHistoryItems([])}
          />
        </aside>

        {/* RIGHT STAGE: TABBED VIEWPORT (NO PAGE SCROLL) */}
        <section className="stage-viewport">
          <div className="subtab-bar">
            <div className="subtabs">
              <button
                type="button"
                className={subTab === "io" ? "subtab active" : "subtab"}
                onClick={() => setSubTab("io")}
              >
                Input / Output Slots
              </button>
              {activeTab !== "snake" ? (
                <>
                  <button
                    type="button"
                    className={subTab === "stages" ? "subtab active" : "subtab"}
                    onClick={() => setSubTab("stages")}
                  >
                    Pipeline Stages
                  </button>
                  <button
                    type="button"
                    className={subTab === "compare" ? "subtab active" : "subtab"}
                    onClick={() => setSubTab("compare")}
                  >
                    OpenCV Comparison
                  </button>
                </>
              ) : (
                <>
                  <button
                    type="button"
                    className={subTab === "evolution" ? "subtab active" : "subtab"}
                    onClick={() => setSubTab("evolution")}
                  >
                    Evolution & Timing
                  </button>
                  <button
                    type="button"
                    className={subTab === "metrics" ? "subtab active" : "subtab"}
                    onClick={() => setSubTab("metrics")}
                  >
                    Chain Code & Metrics
                  </button>
                </>
              )}
            </div>

            {activeTab === "snake" && snakeMetrics && (
              <div className="quick-metrics-strip">
                <span>Iter: <b>{snakeResult?.iterations_run}</b></span>
                <span>Perim: <b>{Number(snakeMetrics.perimeter.toFixed(1))}</b></span>
                <span>Area: <b>{Number(snakeMetrics.shoelace_area.toFixed(1))}</b></span>
                <span>Pick's: <b>{snakeMetrics.pick_theorem_valid ? "Verified" : "Failed"}</b></span>
              </div>
            )}
          </div>

          <div className="viewport-content">
            {/* SUBTAB 1: I/O WORKSPACE */}
            {/* Change 2: Clicking Input slot opens file dialog on Canny/Hough/Ellipse/Circle; Snake has a dedicated button in its slot header */}
            {subTab === "io" && activeTab === "canny" && (
              <div className="viewer-grid">
                <div
                  className="viewer-card input-card clickable-input"
                  onClick={openFilePicker}
                  title="Click to select or change input image"
                >
                  <div className="card-header">
                    <span className="viewer-label">INPUT · ORIGINAL</span>
                    <span className="slot-label">Click slot to change image</span>
                  </div>
                  <div className="card-media">
                    {original ? <img src={original} alt="Original Canny input" /> : <div className="viewer-empty">Click here to select an image</div>}
                  </div>
                </div>
                <SlotCard slot="1" value={cannySlots["1"]} source={cannySlotSources["1"]} selected={cannySlot === "1"} onClick={() => setCannySlot("1")} />
                <SlotCard slot="2" value={cannySlots["2"]} source={cannySlotSources["2"]} selected={cannySlot === "2"} onClick={() => setCannySlot("2")} />
              </div>
            )}

            {subTab === "io" && activeTab === "hough" && (
              <div className="viewer-grid">
                <div
                  className="viewer-card input-card clickable-input"
                  onClick={openFilePicker}
                  title="Click to select or change input image"
                >
                  <div className="card-header">
                    <span className="viewer-label">INPUT · {houghInput.toUpperCase()}</span>
                    <span className="slot-label">Click slot to change image</span>
                  </div>
                  <div className="card-media">
                    {selectedInput ? <img src={selectedInput} alt="Selected Hough input" /> : <div className="viewer-empty">Click here to select an image</div>}
                  </div>
                </div>
                <SlotCard slot="1" value={houghSlots["1"]} source={houghSlotSources["1"]} selected={houghSlot === "1"} onClick={() => setHoughSlot("1")} />
                <SlotCard slot="2" value={houghSlots["2"]} source={houghSlotSources["2"]} selected={houghSlot === "2"} onClick={() => setHoughSlot("2")} />
              </div>
            )}

            {subTab === "io" && activeTab === "ellipse" && (
              <div className="viewer-grid">
                <div
                  className="viewer-card input-card clickable-input"
                  onClick={openFilePicker}
                  title="Click to select or change input image"
                >
                  <div className="card-header">
                    <span className="viewer-label">INPUT · {showEllipseGuides ? "AXIS GUIDES" : "CANNY EDGES"}</span>
                    <span className="slot-label">Click slot to change image</span>
                  </div>
                  <div className="card-media">
                    {showEllipseGuides ? (
                      <AxisReferencePreview source={original} aMin={ellipse.a_min} aMax={ellipse.a_max} />
                    ) : selectedEllipseInput ? (
                      <img src={selectedEllipseInput} alt="Selected Canny input for ellipse detection" />
                    ) : (
                      <div className="viewer-empty">Click here to select an image</div>
                    )}
                  </div>
                </div>
                <SlotCard slot="1" value={ellipseSlots["1"]} source={ellipseSlotSources["1"]} selected={ellipseSlot === "1"} onClick={() => setEllipseSlot("1")} />
                <SlotCard slot="2" value={ellipseSlots["2"]} source={ellipseSlotSources["2"]} selected={ellipseSlot === "2"} onClick={() => setEllipseSlot("2")} />
              </div>
            )}

            {subTab === "io" && activeTab === "circle" && (
              <div className="viewer-grid">
                <div
                  className="viewer-card input-card clickable-input"
                  onClick={openFilePicker}
                  title="Click to select or change input image"
                >
                  <div className="card-header">
                    <span className="viewer-label">INPUT · DIAMETER GUIDES</span>
                    <span className="slot-label">Click slot to change image</span>
                  </div>
                  <div className="card-media">
                    <CircleDiameterPreview source={selectedCircleInput} minDiam={activeCircle.min_diam} maxDiam={activeCircle.max_diam} />
                  </div>
                </div>
                <SlotCard
                  slot="1"
                  value={circleSlots["1"]}
                  source={circleSlotSources["1"]}
                  selected={circleSlot === "1"}
                  onClick={() => setCircleSlot("1")}
                  summary={customCircleSummary(circleRuns["1"])}
                />
                <SlotCard
                  slot="2"
                  value={circleSlots["2"]}
                  source={circleSlotSources["2"]}
                  selected={circleSlot === "2"}
                  onClick={() => setCircleSlot("2")}
                  summary={customCircleSummary(circleRuns["2"])}
                />
              </div>
            )}

            {subTab === "io" && activeTab === "snake" && (
              <div className="viewer-grid">
                <div className="viewer-card input-card">
                  <div className="card-header">
                    <span className="viewer-label">INPUT · CLICK CONTOUR ({snakePoints.length} PTS)</span>
                    <button
                      type="button"
                      className="slot-upload-btn"
                      onClick={openFilePicker}
                      title="Select or change input image"
                    >
                      {hasImage ? "Change Image" : "Select Image"}
                    </button>
                  </div>
                  <div className="card-media">
                    <SnakeContourInput
                      source={original}
                      points={snakePoints}
                      onAddPoint={(pt) => setSnakePoints((curr) => (curr.length >= snake.n_points ? [...curr.slice(1), pt] : [...curr, pt]))}
                      onImageLoaded={setImageDims}
                    />
                  </div>
                </div>
                <SlotCard slot="1" value={snakeSlots["1"]} source={snakeSlotSources["1"]} selected={snakeSlot === "1"} onClick={() => setSnakeSlot("1")} />
                <SlotCard slot="2" value={snakeSlots["2"]} source={snakeSlotSources["2"]} selected={snakeSlot === "2"} onClick={() => setSnakeSlot("2")} />
              </div>
            )}

            {/* SUBTAB 2: PIPELINE STAGES */}
            {subTab === "stages" && activeTab === "canny" && (
              <StagePipeline
                title="Canny stages"
                steps={selectedCannyRun?.steps}
                names={["grayscale", "gaussian_blur", "gradient_magnitude", "gradient_direction", "non_maximum_suppression", "edges"]}
                selection={cannyPipelineSelection}
                onSelectionChange={setCannyPipelineSelection}
                options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }]}
              />
            )}

            {subTab === "stages" && activeTab === "hough" && (
              <StagePipeline
                title="Hough stages"
                steps={selectedHoughRun?.steps}
                names={["input_edges", "accumulator", "line_overlay"]}
                selection={houghPipelineSelection}
                onSelectionChange={setHoughPipelineSelection}
                options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }]}
              />
            )}

            {subTab === "stages" && activeTab === "ellipse" && (
              <StagePipeline
                title="Ellipse stages"
                steps={selectedEllipseRun?.steps}
                names={["input_edges", "ellipse_overlay"]}
                selection={ellipsePipelineSelection}
                onSelectionChange={setEllipsePipelineSelection}
                options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }]}
              />
            )}

            {subTab === "stages" && activeTab === "circle" && (
              <StagePipeline
                title="Circle stages"
                steps={selectedCircleRun?.steps}
                names={(selectedCircleRun?.algorithm || circleAlgorithm) === "cht" ? ["input_edges", "accumulator", "circle_overlay"] : ["input_edges", "circle_overlay"]}
                selection={circlePipelineSelection}
                onSelectionChange={setCirclePipelineSelection}
                options={[{ value: "1", label: "Output 1" }, { value: "2", label: "Output 2" }]}
              />
            )}

            {/* SUBTAB 3: OPENCV COMPARISON */}
            {subTab === "compare" && activeTab === "canny" && (
              <ComparisonPanel
                title="Canny edges"
                custom={cannyComparison?.output}
                opencv={cannyComparison?.opencv_output}
                metric="Same thresholds · same blurred input"
                slot={cannyComparisonSlot}
                onSlotChange={setCannyComparisonSlot}
              />
            )}

            {subTab === "compare" && activeTab === "hough" && (
              <ComparisonPanel
                title="Hough line overlay"
                custom={houghComparison?.output}
                opencv={houghComparison?.opencv_output}
                metric="Same rho · theta · vote threshold"
                slot={houghComparisonSlot}
                onSlotChange={setHoughComparisonSlot}
              />
            )}

            {subTab === "compare" && activeTab === "ellipse" && (
              <ComparisonPanel
                title="Ellipse overlay"
                custom={ellipseComparison?.output}
                opencv={ellipseComparison?.opencv_output}
                metric="Same edge input · same axis filters"
                slot={ellipseComparisonSlot}
                onSlotChange={setEllipseComparisonSlot}
              />
            )}

            {subTab === "compare" && activeTab === "circle" && (
              <ComparisonPanel
                title={`${(circleComparison?.algorithm || circleAlgorithm) === "cht" ? "Circular Hough" : "RANSAC"} circle overlay`}
                custom={circleComparison?.output}
                opencv={circleComparison?.opencv_output}
                 metric="Same Canny blur · linked to selected from-scratch parameters"
                slot={circleComparisonSlot}
                onSlotChange={setCircleComparisonSlot}
                extraControls={
                  <LinkedCircleOpenCVPanel algorithm={circleAlgorithm} parameters={linkedCircleOpenCV} />
                }
                customSummary={
                  circleComparison && (
                    <CircleSummaryBlock
                      label="CUSTOM RESULT"
                      title={circleComparison.algorithm === "cht" ? "Circular Hough" : "RANSAC"}
                      count={circleComparison.circles?.length ?? 0}
                      parameters={circleComparison.custom_parameters}
                    />
                  )
                }
                opencvSummary={
                  circleComparison && (
                    <CircleSummaryBlock
                      label="OPENCV RESULT"
                      title="HoughCircles"
                      count={circleComparison.opencv_circles?.length ?? 0}
                      parameters={circleComparison.opencv_parameters}
                      className="opencv-inline-summary"
                    />
                  )
                }
              />
            )}

            {/* SNAKE SUBTAB 2: EVOLUTION & TIMING */}
            {subTab === "evolution" && activeTab === "snake" && (
              <SnakeEvolutionPlayer
                source={original}
                frames={snakeResult?.frames}
                showVertices={showVertices}
                timings={snakeResult?.stage_timings_ms}
                totalMs={snakeResult?.processing_time_ms}
              />
            )}

            {/* SNAKE SUBTAB 3: METRICS & CHAIN CODE */}
            {subTab === "metrics" && activeTab === "snake" && (
              snakeMetrics ? (
                <section className="snake-metrics-panel">
                  <div className="metrics-cards">
                    <div className="metric-card">
                      <span className="eyebrow">PERIMETER</span>
                      <strong>{Number(snakeMetrics.perimeter.toFixed(2))} <small>px</small></strong>
                    </div>
                    <div className="metric-card">
                      <span className="eyebrow">SHOELACE AREA</span>
                      <strong>{Number(snakeMetrics.shoelace_area.toFixed(2))} <small>px²</small></strong>
                    </div>
                    <div className="metric-card">
                      <span className="eyebrow">PIXEL FILL AREA</span>
                      <strong>{snakeMetrics.pixel_fill_area} <small>px²</small></strong>
                    </div>
                    <div className="metric-card">
                      <span className="eyebrow">BOUNDARY PIXELS</span>
                      <strong>{snakeMetrics.boundary_pixels} <small>px</small></strong>
                    </div>
                    <div className="metric-card">
                      <span className="eyebrow">PICK'S THEOREM</span>
                      <strong>{snakeMetrics.pick_theorem_valid ? "Verified ✓" : "Failed"}</strong>
                    </div>
                    <div className="metric-card">
                      <span className="eyebrow">ITERATIONS</span>
                      <strong>{snakeResult?.iterations_run ?? 0} <small>({snakeResult?.stop_reason})</small></strong>
                    </div>
                  </div>
                  <div className="chain-code-box">
                    <span className="eyebrow">FREEMAN 8-CONNECTED CHAIN CODE ({snakeMetrics.chain_code.length} STEPS)</span>
                    <code>{snakeMetrics.chain_code}</code>
                  </div>
                </section>
              ) : (
                <div className="comparison-empty">
                  <p>Run Snake first to compute Freeman chain code, perimeter, Shoelace area, and Pick's theorem metrics.</p>
                </div>
              )
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
