import { useState, useEffect, useRef, useCallback } from "react";
import { NeuralNetwork } from "./nn";
import type { NetworkData, ForwardResult } from "./nn";

export default function App() {
  const [nn, setNN] = useState<NeuralNetwork | null>(null);
  const [result, setResult] = useState<ForwardResult | null>(null);
  const [networkData, setNetworkData] = useState<NetworkData | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);

  useEffect(() => {
    fetch("/weights.json")
      .then((r) => r.json())
      .then((data: NetworkData) => {
        setNN(new NeuralNetwork(data));
        setNetworkData(data);
      })
      .catch(() => console.error("No weights.json found. Run: node train.mjs"));
  }, []);

  const getPixels = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const offscreen = document.createElement("canvas");
    offscreen.width = 28;
    offscreen.height = 28;
    const offCtx = offscreen.getContext("2d")!;
    offCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = offCtx.getImageData(0, 0, 28, 28);
    const pixels: number[] = new Array(784);
    for (let i = 0; i < 784; i++) {
      pixels[i] = imageData.data[i * 4 + 3] / 255; // alpha channel (we draw white on transparent)
    }
    return pixels;
  }, []);

  const runInference = useCallback(() => {
    if (!nn) return;
    const pixels = getPixels();
    if (!pixels) return;
    setResult(nn.forward(pixels));
  }, [nn, getPixels]);

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
    runInference();
  };

  const handlePointerUp = () => {
    isDrawing.current = false;
    runInference();
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
      <p style={{ color: "#8b949e", margin: "0 0 32px", fontSize: "14px" }}>
        784 → 16 → 16 → 10 &nbsp;|&nbsp; Trained on MNIST from scratch
      </p>

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

  const layers = networkData.sizes; // [784, 16, 16, 10]
  const svgWidth = 360;
  const svgHeight = 400;
  const layerXPositions = [60, 140, 220, 300];
  // Only visualize hidden + output layers as circles (skip 784 input neurons)
  const displayLayers = [16, 16, 10];

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
        {/* Connections between hidden1→hidden2 and hidden2→output */}
        {[1, 2].map((l) => {
          const fromSize = displayLayers[l - 1];
          const toSize = displayLayers[l];
          const fromX = layerXPositions[l];
          const toX = layerXPositions[l + 1];
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

        {/* Connections from input box to hidden1 */}
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

        {/* Neurons for hidden and output layers */}
        {displayLayers.map((size, li) => {
          const x = layerXPositions[li + 1];
          const layerLabel = li < 2 ? `Hidden ${li + 1}` : "Output";
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
