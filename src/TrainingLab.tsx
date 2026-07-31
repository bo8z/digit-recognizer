import { useEffect, useMemo, useRef, useState } from "react";
import { ParameterNotation } from "./ParameterNotation";
import { TrainingNetworkScene } from "./TrainingNetworkScene";
import type { SceneParameter } from "./TrainingNetworkScene";
import {
  ARCHITECTURE,
  LEARNING_RATE,
  buildParameterDescriptors,
  type LandscapeData,
} from "./training/types";
import {
  useUspsTrainer,
  type LossHistoryPoint,
} from "./training/useUspsTrainer";

const ROW_HEIGHT = 44;
const DEFAULT_LIST_HEIGHT = 352;
const PARAMETER_DESCRIPTORS = buildParameterDescriptors();

interface TrainingParameter extends SceneParameter {
  gradient: number;
}

export function TrainingLab({
  onOpenRecognizer,
}: {
  onOpenRecognizer: () => void;
}) {
  const {
    state,
    start,
    pause,
    stepSample,
    stepBatch,
    reset,
    selectParameter,
  } = useUspsTrainer();
  const [selectedId, setSelectedId] = useState("w-0-6-93");
  const [scrollTop, setScrollTop] = useState(0);
  const [listHeight, setListHeight] = useState(DEFAULT_LIST_HEIGHT);
  const listRef = useRef<HTMLDivElement>(null);
  const parameters = useMemo(
    () =>
      PARAMETER_DESCRIPTORS.map((descriptor) => ({
        id: descriptor.id,
        kind: descriptor.kind,
        layer: descriptor.layer,
        source: descriptor.source,
        destination: descriptor.destination,
        value: state.parameterValues[descriptor.index] ?? 0,
        gradient: state.parameterGradients[descriptor.index] ?? 0,
      })),
    [state.parameterGradients, state.parameterValues],
  );
  const inputPixels = useMemo(
    () => Array.from(state.pixels, (value) => value / 255),
    [state.pixels],
  );
  const activations = useMemo(
    () => state.activations.map((layer) => Array.from(layer)),
    [state.activations],
  );

  const selected =
    parameters.find((parameter) => parameter.id === selectedId) ?? parameters[0];

  useEffect(() => {
    selectParameter(selectedId);
  }, [selectParameter, selectedId]);

  useEffect(() => {
    const index = parameters.findIndex((parameter) => parameter.id === selectedId);
    if (index < 0 || !listRef.current) return;
    const rowTop = index * ROW_HEIGHT;
    const rowBottom = rowTop + ROW_HEIGHT;
    const currentTop = listRef.current.scrollTop;
    const currentBottom = currentTop + listHeight;
    if (rowTop < currentTop || rowBottom > currentBottom) {
      listRef.current.scrollTop = Math.max(0, rowTop - listHeight / 2);
    }
  }, [listHeight, selectedId, parameters]);

  useEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const updateHeight = () => setListHeight(list.clientHeight);
    const observer = new ResizeObserver(updateHeight);
    observer.observe(list);
    updateHeight();
    return () => observer.disconnect();
  }, []);

  const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - 2);
  const visibleCount = Math.ceil(listHeight / ROW_HEIGHT) + 5;
  const visibleParameters = parameters.slice(
    startIndex,
    startIndex + visibleCount,
  );
  const overallProgress =
    state.totalTrainingSamples > 0
      ? state.processedSamples / state.totalTrainingSamples
      : 0;

  function selectFromScene(id: string) {
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
        <span
          style={{
            color: state.phase === "error" ? "#f85149" : "#8b949e",
            fontSize: "11px",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {state.phase === "loading"
            ? "Loading USPS browser resources…"
            : state.phase === "error"
              ? state.error
              : `USPS · ${state.trainCount.toLocaleString()} train · ${state.testCount.toLocaleString()} test`}
        </span>
      </header>

      <section
        style={{
          padding: "18px 24px",
          display: "grid",
          gridTemplateColumns: "minmax(680px, 1fr) 310px",
          gap: "16px",
          borderBottom: "1px solid #21262d",
        }}
      >
        <TrainingProgress
          pixels={inputPixels}
          epoch={state.epoch}
          totalEpochs={state.totalEpochs}
          batch={state.batch}
          batchesPerEpoch={state.batchesPerEpoch}
          sample={state.sample}
          samplesInBatch={state.samplesInBatch}
          optimizerStep={state.optimizerStep}
          overallProgress={overallProgress}
          running={state.running}
          disabled={state.phase !== "ready"}
          complete={overallProgress >= 1}
          label={state.currentLabel}
          prediction={state.prediction}
          sampleLoss={state.sampleLoss}
          onToggle={state.running ? pause : start}
          onStepSample={stepSample}
          onStepBatch={stepBatch}
          onReset={reset}
        />
        <LossHistoryPreview
          history={state.lossHistory}
          optimizerStep={state.optimizerStep}
          batchLoss={state.batchLoss}
          testAccuracy={state.testAccuracy}
        />
      </section>

      <section
        style={{
          padding: "16px 24px 24px",
          display: "grid",
          gridTemplateColumns: "minmax(650px, 1fr) 390px",
          gap: "16px",
          height: "max(680px, calc(100vh - 244px))",
        }}
      >
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
            <h2 style={panelTitleStyle}>Full 3D network</h2>
            <span style={{ color: "#6e7681", fontSize: "11px" }}>
              4,512 visible weight connections
            </span>
          </div>
          <div style={{ minHeight: "560px", flex: 1 }}>
            <TrainingNetworkScene
              architecture={ARCHITECTURE}
              activations={activations}
              parameters={parameters}
              selectedId={selectedId}
              onSelect={selectFromScene}
            />
          </div>
        </section>

        <aside
          style={{
            minWidth: 0,
            minHeight: 0,
            display: "flex",
            flexDirection: "column",
            gap: "16px",
          }}
        >
          <SelectedParameterPanel
            selected={selected}
            landscape={
              state.landscape?.parameterId === selectedId
                ? state.landscape
                : null
            }
          />
          <ParameterExplorer
            parameters={parameters}
            visibleParameters={visibleParameters}
            selectedId={selectedId}
            startIndex={startIndex}
            listRef={listRef}
            onScroll={setScrollTop}
            onSelect={setSelectedId}
          />
        </aside>
      </section>
    </main>
  );
}

function ParameterExplorer({
  parameters,
  visibleParameters,
  selectedId,
  startIndex,
  listRef,
  onScroll,
  onSelect,
}: {
  parameters: TrainingParameter[];
  visibleParameters: TrainingParameter[];
  selectedId: string;
  startIndex: number;
  listRef: React.RefObject<HTMLDivElement | null>;
  onScroll: (value: number) => void;
  onSelect: (value: string) => void;
}) {
  return (
    <section style={{ ...panelStyle, flex: 1, minHeight: "260px" }}>
      <div style={{ padding: "14px", borderBottom: "1px solid #21262d" }}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "space-between",
          }}
        >
          <h2 style={panelTitleStyle}>Parameters</h2>
          <span style={{ color: "#6e7681", fontSize: "11px" }}>
            {parameters.length.toLocaleString()} visible
          </span>
        </div>
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
        <span>Parameter</span>
        <span style={{ textAlign: "right" }}>Value</span>
        <span style={{ textAlign: "right" }}>Gradient</span>
      </div>

      <div
        ref={listRef}
        onScroll={(event) => onScroll(event.currentTarget.scrollTop)}
        style={{
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
          position: "relative",
        }}
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
    </section>
  );
}

function SelectedParameterPanel({
  selected,
  landscape,
}: {
  selected: TrainingParameter;
  landscape: LandscapeData | null;
}) {
  return (
    <section style={{ ...panelStyle, flex: "none" }}>
      <div style={{ padding: "14px 16px 6px" }}>
        <h2 style={{ ...panelTitleStyle, marginBottom: "6px" }}>
          Selected parameter
        </h2>
        <div
          style={{
            color: "#f0f6fc",
            fontSize: "14px",
            fontWeight: 600,
            lineHeight: 1.45,
          }}
        >
          <ParameterNotation parameter={selected} fontSize="19px" />
          <span
            style={{
              marginLeft: "12px",
              color: "#8b949e",
              fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
              fontSize: "10px",
              fontWeight: 400,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            value {formatSigned(selected.value, 5)} · gradient{" "}
            {formatSigned(selected.gradient, 5)} · η {LEARNING_RATE.toFixed(2)}
            {landscape
              ? ` · next ${formatSigned(landscape.nextValue, 5)}`
              : ""}
          </span>
        </div>
      </div>
      <LandscapePreview parameter={selected} landscape={landscape} />
    </section>
  );
}

function SampleImage({ pixels }: { pixels: number[] }) {
  return (
    <div
      aria-label="Preview of the current 16 by 16 USPS training sample"
      style={{
        width: "104px",
        height: "104px",
        display: "grid",
        gridTemplateColumns: "repeat(16, 1fr)",
        overflow: "hidden",
        border: "1px solid #30363d",
        borderRadius: "7px",
        background: "#05070a",
      }}
    >
      {pixels.map((value, index) => (
        <span key={index} style={{ background: `rgba(255,255,255,${value})` }} />
      ))}
    </div>
  );
}

function TrainingProgress({
  pixels,
  epoch,
  totalEpochs,
  batch,
  batchesPerEpoch,
  sample,
  samplesInBatch,
  optimizerStep,
  overallProgress,
  running,
  disabled,
  complete,
  label,
  prediction,
  sampleLoss,
  onToggle,
  onStepSample,
  onStepBatch,
  onReset,
}: {
  pixels: number[];
  epoch: number;
  totalEpochs: number;
  batch: number;
  batchesPerEpoch: number;
  sample: number;
  samplesInBatch: number;
  optimizerStep: number;
  overallProgress: number;
  running: boolean;
  disabled: boolean;
  complete: boolean;
  label: number;
  prediction: number;
  sampleLoss: number;
  onToggle: () => void;
  onStepSample: () => void;
  onStepBatch: () => void;
  onReset: () => void;
}) {
  return (
    <article style={{ ...panelStyle, padding: "14px 16px" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "104px minmax(0, 1fr)",
          alignItems: "center",
          gap: "18px",
        }}
      >
        <SampleImage pixels={pixels} />
        <div style={{ minWidth: 0 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "12px",
              marginBottom: "13px",
            }}
          >
            <p style={{ color: "#8b949e", fontSize: "10px" }}>
              Optimizer step{" "}
              <strong
                style={{
                  color: "#e6edf3",
                  fontSize: "12px",
                  fontWeight: 500,
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {optimizerStep.toLocaleString()}
              </strong>
              <span style={{ marginLeft: "12px" }}>
                Target {label} · Prediction {prediction} · Sample loss{" "}
                {sampleLoss.toFixed(4)}
              </span>
            </p>
            <div style={{ display: "flex", gap: "7px" }}>
              <button
                disabled={disabled || complete}
                onClick={onToggle}
                style={{
                  ...primaryButtonStyle,
                  ...(disabled || complete ? disabledButtonStyle : {}),
                }}
              >
                {running ? "Pause" : complete ? "Complete" : "Train"}
              </button>
              <button
                disabled={disabled || running || complete}
                onClick={onStepSample}
                style={{
                  ...secondaryButtonStyle,
                  ...(disabled || running || complete
                    ? disabledButtonStyle
                    : {}),
                }}
              >
                Step sample
              </button>
              <button
                disabled={disabled || running || complete}
                onClick={onStepBatch}
                style={{
                  ...secondaryButtonStyle,
                  ...(disabled || running || complete
                    ? disabledButtonStyle
                    : {}),
                }}
              >
                Step batch
              </button>
              <button
                disabled={disabled || running}
                onClick={onReset}
                style={{
                  ...secondaryButtonStyle,
                  ...(disabled || running ? disabledButtonStyle : {}),
                }}
              >
                Reset
              </button>
            </div>
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
              gap: "12px",
              marginBottom: "12px",
            }}
          >
            <ProgressMeter
              label="Epoch"
              value={`${epoch} / ${totalEpochs}`}
              progress={epoch / totalEpochs}
            />
            <ProgressMeter
              label="Batch"
              value={`${batch} / ${batchesPerEpoch}`}
              progress={batch / batchesPerEpoch}
            />
            <ProgressMeter
              label="Sample"
              value={`${sample} / ${samplesInBatch}`}
              progress={sample / samplesInBatch}
            />
          </div>
          <ProgressMeter
            label="Overall"
            value={`${Math.min(overallProgress * 100, 100).toFixed(1)}%`}
            progress={overallProgress}
            emphasized
          />
        </div>
      </div>
    </article>
  );
}

function LossHistoryPreview({
  history,
  optimizerStep,
  batchLoss,
  testAccuracy,
}: {
  history: LossHistoryPoint[];
  optimizerStep: number;
  batchLoss: number | null;
  testAccuracy: number | null;
}) {
  const values = history.map((point) => point.loss);
  const trendValues: number[] = [];
  const smoothing = 0.08;
  values.forEach((value, index) => {
    trendValues.push(
      index === 0
        ? value
        : (trendValues[index - 1] ?? value) * (1 - smoothing) +
            value * smoothing,
    );
  });
  const width = 360;
  const height = 92;
  const plotTop = 8;
  const plotBottom = height - 12;
  const maxLoss = values.length > 0 ? Math.max(...values, 0.1) : 1;
  const mapX = (index: number) =>
    values.length <= 1 ? width : (index / (values.length - 1)) * width;
  const mapY = (value: number) =>
    plotTop + (1 - value / maxLoss) * (plotBottom - plotTop);
  const buildPath = (series: number[]) =>
    series
      .map(
        (value, index) =>
          `${index === 0 ? "M" : "L"} ${mapX(index).toFixed(1)} ${mapY(value).toFixed(1)}`,
      )
      .join(" ");
  const rawPath = buildPath(values);
  const trendPath = buildPath(trendValues);

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
        <span style={{ color: "#6e7681", fontSize: "9px" }}>
          Full run · raw batches + trend
        </span>
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
        {rawPath && (
          <path
            d={rawPath}
            fill="none"
            stroke="#58a6ff"
            strokeWidth="0.9"
            opacity="0.28"
          />
        )}
        {trendPath && (
          <path
            d={trendPath}
            fill="none"
            stroke="#58a6ff"
            strokeWidth="2.2"
          />
        )}
        {values.length > 0 && (
          <circle
            cx={width}
            cy={mapY(values.at(-1) ?? 0)}
            r="3"
            fill="#f0f6fc"
          />
        )}
        {values.length === 0 && (
          <text
            x={width / 2}
            y={height / 2}
            fill="#6e7681"
            fontSize="10"
            textAnchor="middle"
          >
            Waiting for the first optimizer step
          </text>
        )}
      </svg>
      <p style={{ color: "#6e7681", fontSize: "10px" }}>
        Step {optimizerStep.toLocaleString()} · Batch loss{" "}
        {batchLoss === null ? "—" : batchLoss.toFixed(5)} · Test accuracy{" "}
        {testAccuracy === null ? "—" : `${(testAccuracy * 100).toFixed(1)}%`}
      </p>
    </article>
  );
}

function LandscapePreview({
  parameter,
  landscape,
}: {
  parameter: TrainingParameter;
  landscape: LandscapeData | null;
}) {
  const width = 378;
  const height = 262;
  const left = 44;
  const right = 14;
  const top = 12;
  const bottom = 34;
  const points = landscape
    ? Array.from(landscape.xValues, (x, index) => ({
        x,
        loss: landscape.losses[index],
      }))
    : [];
  const domainStart = points[0]?.x ?? parameter.value - 0.5;
  const domainEnd = points.at(-1)?.x ?? parameter.value + 0.5;
  const domainSpan = Math.max(domainEnd - domainStart, 0.000001);
  const minLoss = Math.min(...points.map((point) => point.loss));
  const maxLoss = Math.max(...points.map((point) => point.loss));
  const lossSpan = Math.max(maxLoss - minLoss, 0.000001);
  const mapX = (x: number) =>
    left +
    ((x - domainStart) / domainSpan) * (width - left - right);
  const mapY = (loss: number) =>
    top +
    (1 - (loss - minLoss) / lossSpan) * (height - top - bottom);
  const path = points
    .map(
      (point, index) =>
        `${index === 0 ? "M" : "L"} ${mapX(point.x).toFixed(1)} ${mapY(point.loss).toFixed(1)}`,
    )
    .join(" ");
  const currentLoss =
    landscape?.losses[Math.floor(landscape.losses.length / 2)] ?? 0;
  const nextX =
    landscape?.nextValue ??
    parameter.value - LEARNING_RATE * parameter.gradient;
  const nextLoss = landscape?.nextLoss ?? currentLoss;
  const tangentRadius = domainSpan * 0.14;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Exact selected-parameter batch loss landscape"
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
      {landscape ? (
        <>
          <path d={path} fill="none" stroke="#58a6ff" strokeWidth="2.2" />
          <line
            x1={mapX(parameter.value - tangentRadius)}
            y1={mapY(currentLoss - parameter.gradient * tangentRadius)}
            x2={mapX(parameter.value + tangentRadius)}
            y2={mapY(currentLoss + parameter.gradient * tangentRadius)}
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
        </>
      ) : (
        <text
          x={(left + width - right) / 2}
          y={(top + height - bottom) / 2}
          fill="#6e7681"
          fontSize="10"
          textAnchor="middle"
        >
          Calculating exact batch-loss slice…
        </text>
      )}
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
      {landscape && (
        <>
          <circle
            cx={mapX(parameter.value)}
            cy={mapY(currentLoss)}
            r="5"
            fill="#0d1117"
            stroke="#f0f6fc"
            strokeWidth="2"
          />
          <circle
            cx={mapX(nextX)}
            cy={mapY(nextLoss)}
            r="3.5"
            fill="#d29922"
          />
        </>
      )}
      <text x="8" y="18" fill="#6e7681" fontSize="10">
        Mean loss
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
        {parameter.value.toFixed(4)}
      </text>
      {landscape && (
        <text
          x={left}
          y={height - 8}
          fill="#6e7681"
          fontSize="9"
        >
          Exact slice · {landscape.batchSize} USPS sample
          {landscape.batchSize === 1 ? "" : "s"}
        </text>
      )}
    </svg>
  );
}

function ProgressMeter({
  label,
  value,
  progress,
  emphasized = false,
}: {
  label: string;
  value: string;
  progress: number;
  emphasized?: boolean;
}) {
  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: "8px",
          marginBottom: "5px",
        }}
      >
        <span
          style={{
            color: emphasized ? "#c9d1d9" : "#8b949e",
            fontSize: "9px",
            fontWeight: emphasized ? 600 : 500,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
          }}
        >
          {label}
        </span>
        <span
          style={{
            color: "#e6edf3",
            fontSize: emphasized ? "11px" : "10px",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {value}
        </span>
      </div>
      <div
        style={{
          height: emphasized ? "7px" : "5px",
          overflow: "hidden",
          borderRadius: "999px",
          background: "#21262d",
        }}
      >
        <div
          style={{
            width: `${Math.min(Math.max(progress, 0), 1) * 100}%`,
            height: "100%",
            borderRadius: "999px",
            background: emphasized ? "#58a6ff" : "#388bfd",
            transition: "width 0.18s ease",
          }}
        />
      </div>
    </div>
  );
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

const disabledButtonStyle: React.CSSProperties = {
  cursor: "not-allowed",
  opacity: 0.45,
};
