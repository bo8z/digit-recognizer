import { useState, useEffect, useRef, useCallback } from "react";
import { NeuralNetwork } from "./nn";
import type { NetworkData, ForwardResult } from "./nn";

export default function App() {
  const [nn, setNN] = useState<NeuralNetwork | null>(null);
  const [result, setResult] = useState<ForwardResult | null>(null);
  const [networkData, setNetworkData] = useState<NetworkData | null>(null);
  const [mode, setMode] = useState<"2layer" | "1layer">("2layer");
  const [liveInference, setLiveInference] = useState(true);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);

  useEffect(() => {
    const file = mode === "2layer" ? "/weights.json" : "/weights-1layer.json";
    fetch(file)
      .then((r) => r.json())
      .then((data: NetworkData) => {
        setNN(new NeuralNetwork(data));
        setNetworkData(data);
        setResult(null);
      })
      .catch(() => console.error(`No ${file} found.`));
  }, [mode]);

  const getPixels = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    // Step 1: Get raw pixels from the full canvas at native resolution
    const rawCtx = canvas.getContext("2d")!;
    const rawData = rawCtx.getImageData(0, 0, 280, 280);
    const raw = rawData.data;

    // Build a grayscale 280×280 image from alpha channel
    const img280: number[] = new Array(280 * 280);
    for (let i = 0; i < 280 * 280; i++) img280[i] = raw[i * 4 + 3] / 255;

    // Step 2: Find bounding box of the drawn digit
    let top = 280, bottom = 0, left = 280, right = 0;
    for (let y = 0; y < 280; y++) {
      for (let x = 0; x < 280; x++) {
        if (img280[y * 280 + x] > 0.01) {
          if (y < top) top = y;
          if (y > bottom) bottom = y;
          if (x < left) left = x;
          if (x > right) right = x;
        }
      }
    }
    if (top > bottom) return null; // nothing drawn

    // Step 3: Crop to bounding box and resize to fit 20×20 preserving aspect ratio
    const bw = right - left + 1;
    const bh = bottom - top + 1;
    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = bw;
    cropCanvas.height = bh;
    const cropCtx = cropCanvas.getContext("2d")!;
    cropCtx.drawImage(canvas, left, top, bw, bh, 0, 0, bw, bh);

    // Resize to fit in 20×20 box preserving aspect ratio
    const scale = 20 / Math.max(bw, bh);
    const sw = Math.round(bw * scale);
    const sh = Math.round(bh * scale);
    const fitCanvas = document.createElement("canvas");
    fitCanvas.width = sw;
    fitCanvas.height = sh;
    const fitCtx = fitCanvas.getContext("2d")!;
    fitCtx.drawImage(cropCanvas, 0, 0, sw, sh);

    // Step 4: Place in 28×28 canvas centered (initially pad to center of box)
    const padCanvas = document.createElement("canvas");
    padCanvas.width = 28;
    padCanvas.height = 28;
    const padCtx = padCanvas.getContext("2d")!;
    const ox = Math.round((28 - sw) / 2);
    const oy = Math.round((28 - sh) / 2);
    padCtx.drawImage(fitCanvas, ox, oy);

    // Read the 28×28 pixels
    const padData = padCtx.getImageData(0, 0, 28, 28);
    const px: number[] = new Array(784);
    for (let i = 0; i < 784; i++) px[i] = padData.data[i * 4 + 3] / 255;

    // Step 5: Compute center of mass and shift to center
    let cx = 0, cy = 0, total = 0;
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const v = px[y * 28 + x];
        cx += x * v;
        cy += y * v;
        total += v;
      }
    }
    if (total > 0) {
      cx /= total;
      cy /= total;
      const shiftX = Math.round(14 - cx);
      const shiftY = Math.round(14 - cy);

      // Apply shift
      const shifted: number[] = new Array(784).fill(0);
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          const sy = y - shiftY;
          const sx = x - shiftX;
          if (sy >= 0 && sy < 28 && sx >= 0 && sx < 28) {
            shifted[y * 28 + x] = px[sy * 28 + sx];
          }
        }
      }
      return shifted;
    }

    return px;
  }, []);

  const runInference = useCallback(() => {
    if (!nn) return;
    const pixels = getPixels();
    if (!pixels) return;
    setResult(nn.forward(pixels));
  }, [nn, getPixels]);

  const toggleLiveInference = useCallback(() => {
    setLiveInference((prev) => {
      if (!prev) runInference();
      return !prev;
    });
  }, [runInference]);

  const handlePointerDown = (e: React.PointerEvent) => {
    isDrawing.current = true;
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const ctx = canvas.getContext("2d")!;
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
    canvas.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isDrawing.current) return;
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const ctx = canvas.getContext("2d")!;
    ctx.strokeStyle = "white";
    ctx.lineWidth = 24;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    if (liveInference) runInference();
  };

  const handlePointerUp = () => {
    isDrawing.current = false;
    if (liveInference) runInference();
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setResult(null);
  };

  const outputActivations = result?.activations[result.activations.length - 1];
  const prediction = outputActivations
    ? outputActivations.indexOf(Math.max(...outputActivations))
    : null;

  return (
    <div
      style={{
        background: "#0d1117",
        minHeight: "100vh",
        color: "#e6edf3",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        padding: "40px 20px",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <h1 style={{ fontSize: "28px", fontWeight: 500, margin: "0 0 8px" }}>
        Neural Network Digit Recognizer
      </h1>
      <p style={{ color: "#8b949e", margin: "0 0 16px", fontSize: "14px" }}>
        {mode === "2layer" ? "784 → 16 → 16 → 10" : "784 → 10"} &nbsp;|&nbsp; Trained on MNIST from scratch
      </p>
      <div style={{ display: "flex", gap: "8px", marginBottom: "32px" }}>
        <button
          onClick={() => setMode("2layer")}
          style={{
            background: mode === "2layer" ? "#58a6ff" : "#21262d",
            color: mode === "2layer" ? "#0d1117" : "#e6edf3",
            border: "1px solid #30363d",
            borderRadius: "6px",
            padding: "6px 16px",
            cursor: "pointer",
            fontSize: "13px",
            fontWeight: mode === "2layer" ? 600 : 400,
          }}
        >
          2 Hidden Layers (96%)
        </button>
        <button
          onClick={() => setMode("1layer")}
          style={{
            background: mode === "1layer" ? "#58a6ff" : "#21262d",
            color: mode === "1layer" ? "#0d1117" : "#e6edf3",
            border: "1px solid #30363d",
            borderRadius: "6px",
            padding: "6px 16px",
            cursor: "pointer",
            fontSize: "13px",
            fontWeight: mode === "1layer" ? 600 : 400,
          }}
        >
          No Hidden Layers (92%)
        </button>
      </div>

      {!nn ? (
        <p style={{ color: "#f85149" }}>
          Loading weights... (if this persists, run: node train.mjs)
        </p>
      ) : (
        <div
          style={{
            display: "flex",
            gap: "48px",
            alignItems: "flex-start",
            flexWrap: "wrap",
            justifyContent: "center",
          }}
        >
          {/* Drawing area */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
            <p style={{ color: "#8b949e", fontSize: "13px", margin: 0 }}>
              Draw a digit (0-9)
            </p>
            <canvas
              ref={canvasRef}
              width={280}
              height={280}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerUp}
              style={{
                background: "#000",
                borderRadius: "8px",
                cursor: "crosshair",
                touchAction: "none",
                border: "1px solid #30363d",
              }}
            />
            <div style={{ display: "flex", gap: "8px" }}>
              <button
                onClick={clearCanvas}
                style={{
                  background: "#21262d",
                  color: "#e6edf3",
                  border: "1px solid #30363d",
                  borderRadius: "6px",
                  padding: "8px 24px",
                  cursor: "pointer",
                  fontSize: "14px",
                }}
              >
                Clear
              </button>
              <button
                onClick={toggleLiveInference}
                style={{
                  background: liveInference ? "#58a6ff" : "#21262d",
                  color: liveInference ? "#0d1117" : "#e6edf3",
                  border: "1px solid #30363d",
                  borderRadius: "6px",
                  padding: "8px 16px",
                  cursor: "pointer",
                  fontSize: "13px",
                  fontWeight: liveInference ? 600 : 400,
                }}
              >
                {liveInference ? "Live" : "Paused"}
              </button>
            </div>
          </div>

          {/* Network visualization */}
          <NetworkVis result={result} networkData={networkData} />

          {/* Output neurons */}
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
            <p style={{ color: "#8b949e", fontSize: "13px", margin: 0 }}>
              Output layer
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
              {Array.from({ length: 10 }, (_, i) => {
                const activation = outputActivations?.[i] ?? 0;
                const isPrediction = i === prediction && activation > 0.1;
                return (
                  <div
                    key={i}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "12px",
                    }}
                  >
                    <div
                      style={{
                        width: "32px",
                        height: "32px",
                        borderRadius: "50%",
                        background: `rgba(255, 255, 255, ${activation})`,
                        border: isPrediction
                          ? "2px solid #58a6ff"
                          : "1px solid #30363d",
                        transition: "background 0.1s",
                      }}
                    />
                    <span
                      style={{
                        fontSize: "18px",
                        fontWeight: isPrediction ? 700 : 400,
                        color: isPrediction ? "#58a6ff" : "#8b949e",
                        width: "16px",
                      }}
                    >
                      {i}
                    </span>
                    <div
                      style={{
                        width: "80px",
                        height: "8px",
                        background: "#21262d",
                        borderRadius: "4px",
                        overflow: "hidden",
                      }}
                    >
                      <div
                        style={{
                          width: `${activation * 100}%`,
                          height: "100%",
                          background: isPrediction ? "#58a6ff" : "#484f58",
                          borderRadius: "4px",
                          transition: "width 0.1s",
                        }}
                      />
                    </div>
                    <span
                      style={{
                        fontSize: "12px",
                        color: "#8b949e",
                        width: "40px",
                        textAlign: "right",
                      }}
                    >
                      {(activation * 100).toFixed(1)}%
                    </span>
                  </div>
                );
              })}
            </div>
            {prediction !== null && outputActivations && outputActivations[prediction] > 0.1 && (
              <p
                style={{
                  fontSize: "48px",
                  fontWeight: 700,
                  color: "#58a6ff",
                  margin: "12px 0 0",
                  lineHeight: 1,
                }}
              >
                {prediction}
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function NetworkVis({
  result,
  networkData,
}: {
  result: ForwardResult | null;
  networkData: NetworkData | null;
}) {
  if (!networkData) return null;

  const layers = networkData.sizes;
  const displayLayers = layers.slice(1); // skip 784 input neurons
  const numCols = displayLayers.length + 1; // +1 for input box
  const svgWidth = numCols * 80 + 40;
  const svgHeight = 400;
  const layerXPositions = Array.from({ length: numCols }, (_, i) => 60 + i * 80);

  const getNeuronY = (layerSize: number, index: number) => {
    const totalHeight = layerSize * 22;
    const startY = (svgHeight - totalHeight) / 2;
    return startY + index * 22 + 11;
  };

  const activations = result?.activations;

  // Get max weight magnitude for normalization
  let maxWeight = 0;
  for (let l = 1; l < layers.length - 1; l++) {
    for (const row of networkData.weights[l]) {
      for (const w of row) {
        maxWeight = Math.max(maxWeight, Math.abs(w));
      }
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
      <p style={{ color: "#8b949e", fontSize: "13px", margin: 0 }}>
        Network
      </p>
      <svg width={svgWidth} height={svgHeight}>
        {/* Connections between non-input layers */}
        {displayLayers.slice(1).map((_, li) => {
          const l = li + 1;
          const fromSize = displayLayers[li];
          const toSize = displayLayers[li + 1];
          const fromX = layerXPositions[li + 1];
          const toX = layerXPositions[li + 2];
          return Array.from({ length: toSize }, (_, j) =>
            Array.from({ length: fromSize }, (_, k) => {
              const w = networkData.weights[l]?.[j]?.[k] ?? 0;
              const norm = maxWeight > 0 ? Math.abs(w) / maxWeight : 0;
              const color = w >= 0 ? `rgba(88,166,255,${norm * 0.5})` : `rgba(248,81,73,${norm * 0.5})`;
              return (
                <line
                  key={`${l}-${j}-${k}`}
                  x1={fromX}
                  y1={getNeuronY(fromSize, k)}
                  x2={toX}
                  y2={getNeuronY(toSize, j)}
                  stroke={color}
                  strokeWidth={1}
                />
              );
            })
          );
        })}

        {/* Input layer label */}
        <text x={layerXPositions[0]} y={svgHeight - 10} textAnchor="middle" fill="#484f58" fontSize="11">
          784 inputs
        </text>

        {/* Input layer representation — small grid */}
        <rect
          x={layerXPositions[0] - 18}
          y={svgHeight / 2 - 18}
          width={36}
          height={36}
          rx={4}
          fill="none"
          stroke="#30363d"
        />
        {activations && activations[0] && (
          <>
            {Array.from({ length: 28 }, (_, row) =>
              Array.from({ length: 28 }, (_, col) => {
                const val = activations[0][row * 28 + col];
                if (val < 0.05) return null;
                return (
                  <rect
                    key={`p${row}-${col}`}
                    x={layerXPositions[0] - 17 + col * 1.25}
                    y={svgHeight / 2 - 17 + row * 1.25}
                    width={1.25}
                    height={1.25}
                    fill={`rgba(255,255,255,${val})`}
                  />
                );
              })
            )}
          </>
        )}

        {/* Connections from input box to first display layer */}
        {Array.from({ length: displayLayers[0] }, (_, j) => (
          <line
            key={`input-${j}`}
            x1={layerXPositions[0] + 18}
            y1={svgHeight / 2}
            x2={layerXPositions[1]}
            y2={getNeuronY(displayLayers[0], j)}
            stroke="rgba(88,166,255,0.08)"
            strokeWidth={1}
          />
        ))}

        {/* Neurons for all display layers */}
        {displayLayers.map((size, li) => {
          const x = layerXPositions[li + 1];
          const isOutput = li === displayLayers.length - 1;
          const layerLabel = isOutput ? "Output" : `Hidden ${li + 1}`;
          return (
            <g key={`layer-${li}`}>
              <text x={x} y={svgHeight - 10} textAnchor="middle" fill="#484f58" fontSize="11">
                {layerLabel}
              </text>
              {Array.from({ length: size }, (_, i) => {
                const activation = activations?.[li + 1]?.[i] ?? 0;
                return (
                  <circle
                    key={i}
                    cx={x}
                    cy={getNeuronY(size, i)}
                    r={8}
                    fill={`rgba(255,255,255,${activation})`}
                    stroke="#30363d"
                    strokeWidth={1}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
