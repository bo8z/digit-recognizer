import { useEffect, useMemo, useRef, useState } from "react";
import { ParameterNotation } from "./ParameterNotation";
import { TrainingNetworkScene } from "./TrainingNetworkScene";
import type { SceneParameter } from "./TrainingNetworkScene";

const ARCHITECTURE = [256, 16, 16, 10];
const LEARNING_RATE = 0.05;
const ROW_HEIGHT = 44;
const LIST_HEIGHT = 516;

interface TrainingParameter extends SceneParameter {
  label: string;
  group: string;
  gradient: number;
}

export function TrainingLab({
  onOpenRecognizer,
}: {
  onOpenRecognizer: () => void;
}) {
  const parameters = useMemo(buildPreviewParameters, []);
  const inputPixels = useMemo(createPreviewDigit, []);
  const [selectedId, setSelectedId] = useState("w-0-6-93");
  const [query, setQuery] = useState("");
  const [layerFilter, setLayerFilter] = useState("all");
  const [scrollTop, setScrollTop] = useState(0);
  const [running, setRunning] = useState(false);
  const [epoch, setEpoch] = useState(3);
  const [batch, setBatch] = useState(64);
  const [sample, setSample] = useState(13);
  const listRef = useRef<HTMLDivElement>(null);

  const selected =
    parameters.find((parameter) => parameter.id === selectedId) ?? parameters[0];
  const filteredParameters = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return parameters.filter((parameter) => {
      const matchesLayer =
        layerFilter === "all" ||
        (layerFilter === "bias"
          ? parameter.kind === "bias"
          : String(parameter.layer) === layerFilter && parameter.kind === "weight");
      const matchesQuery =
        !normalizedQuery ||
        parameter.label.toLowerCase().includes(normalizedQuery) ||
        parameter.id.toLowerCase().includes(normalizedQuery);
      return matchesLayer && matchesQuery;
    });
  }, [parameters, query, layerFilter]);

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => advanceSample(), 260);
    return () => window.clearInterval(timer);
  });

  useEffect(() => {
    const index = filteredParameters.findIndex((parameter) => parameter.id === selectedId);
    if (index < 0 || !listRef.current) return;
    const rowTop = index * ROW_HEIGHT;
    const rowBottom = rowTop + ROW_HEIGHT;
    const currentTop = listRef.current.scrollTop;
    const currentBottom = currentTop + LIST_HEIGHT;
    if (rowTop < currentTop || rowBottom > currentBottom) {
      listRef.current.scrollTop = Math.max(0, rowTop - LIST_HEIGHT / 2);
    }
  }, [selectedId, filteredParameters]);

  const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - 2);
  const visibleCount = Math.ceil(LIST_HEIGHT / ROW_HEIGHT) + 5;
  const visibleParameters = filteredParameters.slice(
    startIndex,
    startIndex + visibleCount,
  );
  const progress = ((batch - 1) * 32 + sample) / 7291;
  const optimizerStep = (epoch - 1) * 228 + batch;

  function advanceSample() {
    setSample((currentSample) => {
      if (currentSample < 32) return currentSample + 1;
      setBatch((currentBatch) => {
        if (currentBatch < 228) return currentBatch + 1;
        setEpoch((currentEpoch) => (currentEpoch < 20 ? currentEpoch + 1 : 1));
        return 1;
      });
      return 1;
    });
  }

  function advanceBatch() {
    setSample(1);
    setBatch((currentBatch) => {
      if (currentBatch < 228) return currentBatch + 1;
      setEpoch((currentEpoch) => (currentEpoch < 20 ? currentEpoch + 1 : 1));
      return 1;
    });
  }

  function selectFromScene(id: string) {
    setQuery("");
    setLayerFilter("all");
    setSelectedId(id);
  }

  return (
    <main
      style={{
        minHeight: "100vh",
        minWidth: "1180px",
        background: "#0d1117",
        color: "#e6edf3",
        fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
      }}
    >
      <header
        style={{
          height: "64px",
          padding: "0 24px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid #21262d",
          background: "#0d1117",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "22px" }}>
          <h1 style={{ margin: 0, fontSize: "17px", fontWeight: 600 }}>
            Neural Network Laboratory
          </h1>
          <nav
            aria-label="Workspace"
            style={{
              display: "flex",
              padding: "3px",
              border: "1px solid #30363d",
              borderRadius: "7px",
              background: "#161b22",
            }}
          >
            <button onClick={onOpenRecognizer} style={navButtonStyle(false)}>
              Recognizer
            </button>
            <button style={navButtonStyle(true)}>Training Lab</button>
          </nav>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span
            style={{
              padding: "5px 9px",
              border: "1px solid #9e6a03",
              borderRadius: "999px",
              color: "#d29922",
              background: "rgba(210,153,34,0.08)",
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "0.04em",
              textTransform: "uppercase",
            }}
          >
            UI preview
          </span>
          <span style={{ color: "#8b949e", fontSize: "12px" }}>
            USPS · 256 → 16 → 16 → 10 · 4,554 parameters
          </span>
        </div>
      </header>

      <section
        style={{
          padding: "18px 24px",
          display: "grid",
          gridTemplateColumns: "118px minmax(480px, 1fr) 310px",
          gap: "16px",
          borderBottom: "1px solid #21262d",
        }}
      >
        <SamplePreview pixels={inputPixels} sample={sample} />
        <TrainingProgress
          epoch={epoch}
          batch={batch}
          sample={sample}
          optimizerStep={optimizerStep}
          progress={progress}
          running={running}
          onToggle={() => setRunning((value) => !value)}
          onStepSample={advanceSample}
          onStepBatch={advanceBatch}
        />
        <LossHistoryPreview batch={batch} />
      </section>

      <section
        style={{
          padding: "16px 24px 24px",
          display: "grid",
          gridTemplateColumns: "290px minmax(430px, 1fr) 350px",
          gap: "16px",
          minHeight: "calc(100vh - 244px)",
        }}
      >
        <ParameterExplorer
          parameters={filteredParameters}
          visibleParameters={visibleParameters}
          selectedId={selectedId}
          query={query}
          layerFilter={layerFilter}
          startIndex={startIndex}
          scrollTop={scrollTop}
          listRef={listRef}
          onQueryChange={setQuery}
          onLayerFilterChange={setLayerFilter}
          onScroll={setScrollTop}
          onSelect={setSelectedId}
        />

        <section style={panelStyle}>
          <div
            style={{
              padding: "14px 16px",
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              borderBottom: "1px solid #21262d",
            }}
          >
            <div>
              <h2 style={panelTitleStyle}>Full 3D network</h2>
              <p style={panelSubtitleStyle}>
                Select a weight connection or neuron to inspect its parameter
              </p>
            </div>
            <span style={{ color: "#6e7681", fontSize: "11px" }}>
              4,512 visible weight connections
            </span>
          </div>
          <div style={{ minHeight: "560px", flex: 1 }}>
            <TrainingNetworkScene
              architecture={ARCHITECTURE}
              inputPixels={inputPixels}
              parameters={parameters}
              selectedId={selectedId}
              onSelect={selectFromScene}
            />
          </div>
        </section>

        <SelectedParameterPanel selected={selected} batch={batch} />
      </section>
    </main>
  );
}

function ParameterExplorer({
  parameters,
  visibleParameters,
  selectedId,
  query,
  layerFilter,
  startIndex,
  scrollTop,
  listRef,
  onQueryChange,
  onLayerFilterChange,
  onScroll,
  onSelect,
}: {
  parameters: TrainingParameter[];
  visibleParameters: TrainingParameter[];
  selectedId: string;
  query: string;
  layerFilter: string;
  startIndex: number;
  scrollTop: number;
  listRef: React.RefObject<HTMLDivElement | null>;
  onQueryChange: (value: string) => void;
  onLayerFilterChange: (value: string) => void;
  onScroll: (value: number) => void;
  onSelect: (value: string) => void;
}) {
  return (
    <aside style={panelStyle}>
      <div style={{ padding: "14px 14px 12px", borderBottom: "1px solid #21262d" }}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "space-between",
            marginBottom: "10px",
          }}
        >
          <h2 style={panelTitleStyle}>Parameters</h2>
          <span style={{ color: "#6e7681", fontSize: "11px" }}>
            {parameters.length.toLocaleString()} visible
          </span>
        </div>
        <input
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Find a w or b parameter"
          aria-label="Search parameters"
          style={{
            width: "100%",
            height: "34px",
            padding: "0 10px",
            marginBottom: "8px",
            border: "1px solid #30363d",
            borderRadius: "6px",
            outline: "none",
            background: "#0d1117",
            color: "#e6edf3",
            fontSize: "12px",
          }}
        />
        <select
          value={layerFilter}
          onChange={(event) => onLayerFilterChange(event.target.value)}
          aria-label="Filter parameter layer"
          style={{
            width: "100%",
            height: "34px",
            padding: "0 8px",
            border: "1px solid #30363d",
            borderRadius: "6px",
            background: "#0d1117",
            color: "#c9d1d9",
            fontSize: "12px",
          }}
        >
          <option value="all">All weights and biases</option>
          <option value="0">W⁽¹⁾ · Input → Hidden 1</option>
          <option value="1">W⁽²⁾ · Hidden 1 → Hidden 2</option>
          <option value="2">W⁽³⁾ · Hidden 2 → Output</option>
          <option value="bias">b⁽ℓ⁾ · Biases</option>
        </select>
      </div>

      <div
        style={{
          height: "30px",
          padding: "0 12px",
          display: "grid",
          gridTemplateColumns: "1fr 60px 60px",
          alignItems: "center",
          borderBottom: "1px solid #21262d",
          color: "#6e7681",
          fontSize: "10px",
          textTransform: "uppercase",
          letterSpacing: "0.06em",
        }}
      >
        <span>Connection</span>
        <span style={{ textAlign: "right" }}>Value</span>
        <span style={{ textAlign: "right" }}>Gradient</span>
      </div>

      <div
        ref={listRef}
        onScroll={(event) => onScroll(event.currentTarget.scrollTop)}
        style={{ height: `${LIST_HEIGHT}px`, overflowY: "auto", position: "relative" }}
      >
        <div
          style={{
            position: "relative",
            height: `${parameters.length * ROW_HEIGHT}px`,
          }}
        >
          {visibleParameters.map((parameter, visibleIndex) => {
            const index = startIndex + visibleIndex;
            const selected = parameter.id === selectedId;
            return (
              <button
                key={parameter.id}
                onClick={() => onSelect(parameter.id)}
                style={{
                  position: "absolute",
                  top: `${index * ROW_HEIGHT}px`,
                  left: 0,
                  width: "100%",
                  height: `${ROW_HEIGHT}px`,
                  padding: "0 12px",
                  display: "grid",
                  gridTemplateColumns: "1fr 60px 60px",
                  alignItems: "center",
                  border: "none",
                  borderBottom: "1px solid #1b2028",
                  background: selected ? "#1f6feb" : "transparent",
                  color: selected ? "#ffffff" : "#c9d1d9",
                  cursor: "pointer",
                  textAlign: "left",
                  fontFamily: "inherit",
                }}
              >
                <span
                  style={{
                    minWidth: 0,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    fontSize: "11px",
                  }}
                >
                  <ParameterNotation parameter={parameter} fontSize="14px" />
                </span>
                <span style={{ textAlign: "right", fontSize: "11px", fontVariantNumeric: "tabular-nums" }}>
                  {formatSigned(parameter.value)}
                </span>
                <span style={{ textAlign: "right", fontSize: "11px", fontVariantNumeric: "tabular-nums" }}>
                  {formatSigned(parameter.gradient)}
                </span>
              </button>
            );
          })}
        </div>
      </div>
      <p
        style={{
          padding: "10px 12px",
          borderTop: "1px solid #21262d",
          color: "#6e7681",
          fontSize: "10px",
        }}
      >
        Scroll the complete parameter set or select its edge or neuron in 3D.
        Current position: {Math.round(scrollTop / ROW_HEIGHT).toLocaleString()}.
      </p>
    </aside>
  );
}

function SelectedParameterPanel({
  selected,
  batch,
}: {
  selected: TrainingParameter;
  batch: number;
}) {
  const nextValue = selected.value - LEARNING_RATE * selected.gradient;

  return (
    <aside style={panelStyle}>
      <div style={{ padding: "14px 16px", borderBottom: "1px solid #21262d" }}>
        <p
          style={{
            color: "#58a6ff",
            fontSize: "10px",
            fontWeight: 600,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            marginBottom: "6px",
          }}
        >
          Selected parameter
        </p>
        <h2
          style={{
            margin: 0,
            color: "#f0f6fc",
            fontSize: "14px",
            fontWeight: 600,
            lineHeight: 1.45,
          }}
        >
          <ParameterNotation parameter={selected} fontSize="19px" />
        </h2>
        <p style={{ ...panelSubtitleStyle, marginTop: "4px" }}>{selected.group}</p>
      </div>

      <div
        style={{
          padding: "12px 16px",
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "8px",
          borderBottom: "1px solid #21262d",
        }}
      >
        <Metric label="Current value" value={formatSigned(selected.value, 5)} />
        <Metric label="Gradient" value={formatSigned(selected.gradient, 5)} />
        <Metric
          label="Update"
          value={formatSigned(-LEARNING_RATE * selected.gradient, 5)}
        />
        <Metric label="Next value" value={formatSigned(nextValue, 5)} />
      </div>

      <div style={{ padding: "14px 16px 8px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: "4px",
          }}
        >
          <h3 style={{ margin: 0, fontSize: "13px", fontWeight: 600 }}>
            Parameter loss landscape
          </h3>
          <span style={{ color: "#6e7681", fontSize: "10px" }}>
            Preview · batch {batch}
          </span>
        </div>
        <p style={panelSubtitleStyle}>
          The USPS engine will replace this preview with an exact frozen-snapshot scan.
        </p>
      </div>
      <LandscapePreview parameter={selected} />

      <div
        style={{
          margin: "4px 16px 16px",
          padding: "10px 12px",
          border: "1px solid #30363d",
          borderRadius: "7px",
          background: "#0d1117",
        }}
      >
        <p style={{ color: "#8b949e", fontSize: "10px", marginBottom: "5px" }}>
          Gradient-descent update
        </p>
        <code style={{ color: "#c9d1d9", fontSize: "11px" }}>
          {selected.value.toFixed(5)} − {LEARNING_RATE.toFixed(2)} × (
          {selected.gradient.toFixed(5)}) = {nextValue.toFixed(5)}
        </code>
      </div>
    </aside>
  );
}

function SamplePreview({ pixels, sample }: { pixels: number[]; sample: number }) {
  return (
    <article style={{ ...panelStyle, padding: "10px", alignItems: "center" }}>
      <div
        aria-label="Preview of a 16 by 16 USPS digit"
        style={{
          width: "94px",
          height: "94px",
          display: "grid",
          gridTemplateColumns: "repeat(16, 1fr)",
          overflow: "hidden",
          border: "1px solid #30363d",
          borderRadius: "6px",
          background: "#05070a",
        }}
      >
        {pixels.map((value, index) => (
          <span
            key={index}
            style={{ background: `rgba(255,255,255,${value})` }}
          />
        ))}
      </div>
      <p style={{ color: "#8b949e", fontSize: "10px", marginTop: "7px" }}>
        Sample {sample}/32 · Label 2
      </p>
    </article>
  );
}

function TrainingProgress({
  epoch,
  batch,
  sample,
  optimizerStep,
  progress,
  running,
  onToggle,
  onStepSample,
  onStepBatch,
}: {
  epoch: number;
  batch: number;
  sample: number;
  optimizerStep: number;
  progress: number;
  running: boolean;
  onToggle: () => void;
  onStepSample: () => void;
  onStepBatch: () => void;
}) {
  return (
    <article style={{ ...panelStyle, padding: "14px 16px" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "12px",
          marginBottom: "14px",
        }}
      >
        <div style={{ display: "flex", gap: "22px" }}>
          <ProgressValue label="Epoch" value={`${epoch} / 20`} />
          <ProgressValue label="Batch" value={`${batch} / 228`} />
          <ProgressValue label="Sample" value={`${sample} / 32`} />
          <ProgressValue label="Optimizer step" value={optimizerStep.toLocaleString()} />
        </div>
        <div style={{ display: "flex", gap: "7px" }}>
          <button onClick={onToggle} style={primaryButtonStyle}>
            {running ? "Pause" : "Run preview"}
          </button>
          <button onClick={onStepSample} style={secondaryButtonStyle}>
            Step sample
          </button>
          <button onClick={onStepBatch} style={secondaryButtonStyle}>
            Step batch
          </button>
        </div>
      </div>
      <div
        style={{
          height: "7px",
          overflow: "hidden",
          borderRadius: "999px",
          background: "#21262d",
        }}
      >
        <div
          style={{
            width: `${Math.min(progress * 100, 100)}%`,
            height: "100%",
            borderRadius: "999px",
            background: "#58a6ff",
            transition: "width 0.18s ease",
          }}
        />
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: "7px",
          color: "#6e7681",
          fontSize: "10px",
        }}
      >
        <span>USPS training sequence</span>
        <span>{Math.min((progress * 100), 100).toFixed(1)}%</span>
      </div>
    </article>
  );
}

function LossHistoryPreview({ batch }: { batch: number }) {
  const values = Array.from({ length: 52 }, (_, index) => {
    const decay = 0.78 * Math.exp(-index / 24);
    return 0.12 + decay + Math.sin(index * 1.72) * 0.045 * Math.exp(-index / 34);
  });
  const width = 360;
  const height = 92;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const path = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = 8 + (1 - (value - min) / (max - min)) * (height - 20);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <article style={{ ...panelStyle, padding: "12px 14px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
        }}
      >
        <h2 style={panelTitleStyle}>Training history</h2>
        <span style={{ color: "#6e7681", fontSize: "10px" }}>Preview data</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Preview of the batch loss history"
        style={{ width: "100%", height: "86px", display: "block" }}
      >
        {[24, 48, 72].map((y) => (
          <line key={y} x1="0" x2={width} y1={y} y2={y} stroke="#21262d" />
        ))}
        <path d={path} fill="none" stroke="#58a6ff" strokeWidth="2" />
        <circle
          cx={width}
          cy={8 + (1 - (values.at(-1)! - min) / (max - min)) * (height - 20)}
          r="3"
          fill="#f0f6fc"
        />
      </svg>
      <p style={{ color: "#6e7681", fontSize: "10px" }}>
        Batch {batch} · Loss 0.184 · Validation accuracy 93.7%
      </p>
    </article>
  );
}

function LandscapePreview({ parameter }: { parameter: TrainingParameter }) {
  const width = 378;
  const height = 262;
  const left = 44;
  const right = 14;
  const top = 12;
  const bottom = 34;
  const domainRadius = 1.25;
  const sampleCount = 96;
  const points = Array.from({ length: sampleCount }, (_, index) => {
    const x =
      parameter.value -
      domainRadius +
      (index / (sampleCount - 1)) * domainRadius * 2;
    const distance = x - parameter.value;
    const loss =
      0.24 +
      parameter.gradient * distance +
      0.19 * distance ** 2 +
      0.028 * distance ** 4;
    return { x, loss };
  });
  const minLoss = Math.min(...points.map((point) => point.loss));
  const maxLoss = Math.max(...points.map((point) => point.loss));
  const mapX = (x: number) =>
    left +
    ((x - (parameter.value - domainRadius)) / (domainRadius * 2)) *
      (width - left - right);
  const mapY = (loss: number) =>
    top +
    (1 - (loss - minLoss) / (maxLoss - minLoss)) * (height - top - bottom);
  const path = points
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"} ${mapX(point.x).toFixed(1)} ${mapY(point.loss).toFixed(1)}`,
    )
    .join(" ");
  const currentLoss = 0.24;
  const nextX = parameter.value - LEARNING_RATE * parameter.gradient;
  const nextDistance = nextX - parameter.value;
  const nextLoss =
    0.24 +
    parameter.gradient * nextDistance +
    0.19 * nextDistance ** 2 +
    0.028 * nextDistance ** 4;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Preview of a selected parameter loss landscape"
      style={{ width: "100%", height: "274px", display: "block" }}
    >
      {[0.25, 0.5, 0.75].map((fraction) => {
        const y = top + fraction * (height - top - bottom);
        return (
          <line
            key={fraction}
            x1={left}
            x2={width - right}
            y1={y}
            y2={y}
            stroke="#21262d"
          />
        );
      })}
      <line
        x1={left}
        x2={left}
        y1={top}
        y2={height - bottom}
        stroke="#484f58"
      />
      <line
        x1={left}
        x2={width - right}
        y1={height - bottom}
        y2={height - bottom}
        stroke="#484f58"
      />
      <path d={path} fill="none" stroke="#58a6ff" strokeWidth="2.2" />
      <line
        x1={mapX(parameter.value - 0.36)}
        y1={mapY(currentLoss - parameter.gradient * 0.36)}
        x2={mapX(parameter.value + 0.36)}
        y2={mapY(currentLoss + parameter.gradient * 0.36)}
        stroke="#d29922"
        strokeWidth="1.5"
        strokeDasharray="4 4"
      />
      <line
        x1={mapX(parameter.value)}
        y1={mapY(currentLoss)}
        x2={mapX(nextX)}
        y2={mapY(nextLoss)}
        stroke="#f0f6fc"
        strokeWidth="1.5"
        markerEnd="url(#landscape-arrow)"
      />
      <defs>
        <marker
          id="landscape-arrow"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="5"
          markerHeight="5"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#f0f6fc" />
        </marker>
      </defs>
      <circle
        cx={mapX(parameter.value)}
        cy={mapY(currentLoss)}
        r="5"
        fill="#0d1117"
        stroke="#f0f6fc"
        strokeWidth="2"
      />
      <circle cx={mapX(nextX)} cy={mapY(nextLoss)} r="3.5" fill="#d29922" />
      <text x="8" y="18" fill="#6e7681" fontSize="10">
        Loss
      </text>
      <text x={width - right} y={height - 8} fill="#6e7681" fontSize="10" textAnchor="end">
        Parameter value
      </text>
      <text
        x={mapX(parameter.value)}
        y={height - 16}
        fill="#8b949e"
        fontSize="9"
        textAnchor="middle"
      >
        {parameter.value.toFixed(3)}
      </text>
    </svg>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div
      style={{
        padding: "9px 10px",
        border: "1px solid #30363d",
        borderRadius: "6px",
        background: "#0d1117",
      }}
    >
      <p style={{ color: "#6e7681", fontSize: "9px", marginBottom: "4px" }}>
        {label}
      </p>
      <p
        style={{
          color: "#e6edf3",
          fontSize: "12px",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </p>
    </div>
  );
}

function ProgressValue({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p
        style={{
          color: "#6e7681",
          fontSize: "9px",
          marginBottom: "3px",
          textTransform: "uppercase",
          letterSpacing: "0.06em",
        }}
      >
        {label}
      </p>
      <p style={{ color: "#e6edf3", fontSize: "14px", fontVariantNumeric: "tabular-nums" }}>
        {value}
      </p>
    </div>
  );
}

function buildPreviewParameters(): TrainingParameter[] {
  const parameters: TrainingParameter[] = [];

  for (let layer = 0; layer < ARCHITECTURE.length - 1; layer++) {
    const sourceCount = ARCHITECTURE[layer];
    const destinationCount = ARCHITECTURE[layer + 1];
    for (let destination = 0; destination < destinationCount; destination++) {
      for (let source = 0; source < sourceCount; source++) {
        const seed = layer * 100000 + destination * 1000 + source;
        parameters.push({
          id: `w-${layer}-${destination}-${source}`,
          kind: "weight",
          layer,
          source,
          destination,
          value: signedHash(seed, layer === 0 ? 0.22 : 0.48),
          gradient: signedHash(seed + 77237, 0.065),
          group: layerName(layer),
          label: weightLabel(layer, source, destination),
        });
      }
    }
    for (let destination = 0; destination < destinationCount; destination++) {
      const seed = 900000 + layer * 1000 + destination;
      parameters.push({
        id: `b-${layer}-${destination}`,
        kind: "bias",
        layer,
        source: null,
        destination,
        value: signedHash(seed, 0.18),
        gradient: signedHash(seed + 23411, 0.055),
        group: `b⁽${superscriptDigit(layer + 1)}⁾ · layer ${layer + 1} biases`,
        label: biasLabel(layer, destination),
      });
    }
  }

  return parameters;
}

function weightLabel(layer: number, source: number, destination: number) {
  return `w_{${destination + 1},${source + 1}}^{(${layer + 1})}`;
}

function biasLabel(layer: number, destination: number) {
  return `b_{${destination + 1}}^{(${layer + 1})}`;
}

function layerName(layer: number) {
  return `W⁽${superscriptDigit(layer + 1)}⁾ · layer ${layer} → layer ${layer + 1}`;
}

function superscriptDigit(value: number) {
  return ["⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹"][value];
}

function signedHash(seed: number, scale: number) {
  const raw = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  const normalized = raw - Math.floor(raw);
  return (normalized * 2 - 1) * scale;
}

function createPreviewDigit() {
  const segments: Array<[[number, number], [number, number]]> = [
    [[3, 3], [12, 3]],
    [[12, 3], [12, 7]],
    [[12, 7], [3, 12]],
    [[3, 12], [3, 13]],
    [[3, 13], [13, 13]],
  ];

  return Array.from({ length: 256 }, (_, index) => {
    const x = index % 16;
    const y = Math.floor(index / 16);
    const distance = Math.min(
      ...segments.map(([start, end]) => distanceToSegment(x, y, start, end)),
    );
    if (distance > 1.45) return 0;
    return Math.max(0, Math.min(1, 1.08 - distance * 0.58));
  });
}

function distanceToSegment(
  x: number,
  y: number,
  start: [number, number],
  end: [number, number],
) {
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  const projection =
    lengthSquared === 0
      ? 0
      : Math.max(
          0,
          Math.min(1, ((x - start[0]) * dx + (y - start[1]) * dy) / lengthSquared),
        );
  const projectedX = start[0] + projection * dx;
  const projectedY = start[1] + projection * dy;
  return Math.hypot(x - projectedX, y - projectedY);
}

function formatSigned(value: number, digits = 3) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function navButtonStyle(active: boolean): React.CSSProperties {
  return {
    padding: "6px 11px",
    border: "none",
    borderRadius: "4px",
    background: active ? "#30363d" : "transparent",
    color: active ? "#f0f6fc" : "#8b949e",
    cursor: active ? "default" : "pointer",
    fontSize: "12px",
    fontFamily: "inherit",
  };
}

const panelStyle: React.CSSProperties = {
  minWidth: 0,
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
  border: "1px solid #30363d",
  borderRadius: "9px",
  background: "#161b22",
};

const panelTitleStyle: React.CSSProperties = {
  margin: 0,
  color: "#e6edf3",
  fontSize: "13px",
  fontWeight: 600,
};

const panelSubtitleStyle: React.CSSProperties = {
  margin: 0,
  color: "#6e7681",
  fontSize: "10px",
  lineHeight: 1.45,
};

const primaryButtonStyle: React.CSSProperties = {
  height: "32px",
  padding: "0 13px",
  border: "1px solid #58a6ff",
  borderRadius: "6px",
  background: "#1f6feb",
  color: "#ffffff",
  cursor: "pointer",
  fontFamily: "inherit",
  fontSize: "11px",
  fontWeight: 600,
};

const secondaryButtonStyle: React.CSSProperties = {
  height: "32px",
  padding: "0 11px",
  border: "1px solid #30363d",
  borderRadius: "6px",
  background: "#21262d",
  color: "#c9d1d9",
  cursor: "pointer",
  fontFamily: "inherit",
  fontSize: "11px",
};
