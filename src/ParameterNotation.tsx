export function ParameterNotation({
  parameter,
  fontSize,
}: {
  parameter: {
    kind: "weight" | "bias";
    layer: number;
    source: number | null;
    destination: number;
  };
  fontSize: string;
}) {
  const layer = parameter.layer + 1;
  const destination = parameter.destination + 1;
  const source = (parameter.source ?? 0) + 1;

  return (
    <MathSymbol
      base={parameter.kind === "weight" ? "w" : "b"}
      subscript={
        parameter.kind === "weight" ? `${destination},${source}` : String(destination)
      }
      superscript={String(layer)}
      fontSize={fontSize}
    />
  );
}

export function ActivationNotation({
  index,
  fontSize,
}: {
  index: number;
  fontSize: string;
}) {
  return (
    <MathSymbol
      base="a"
      subscript={String(index + 1)}
      superscript="0"
      fontSize={fontSize}
    />
  );
}

function MathSymbol({
  base,
  subscript,
  superscript,
  fontSize,
}: {
  base: string;
  subscript: string;
  superscript: string;
  fontSize: string;
}) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        color: "inherit",
        fontFamily: '"STIX Two Math", "Cambria Math", "Times New Roman", serif',
        fontSize,
        lineHeight: 1,
      }}
    >
      <i>{base}</i>
      <span
        style={{
          display: "inline-grid",
          gridTemplateRows: "1fr 1fr",
          marginLeft: "2px",
          fontSize: "0.65em",
          fontStyle: "normal",
          lineHeight: 0.86,
          textAlign: "left",
        }}
      >
        <sup style={{ gridRow: 1, fontSize: "inherit" }}>({superscript})</sup>
        <sub style={{ gridRow: 2, fontSize: "inherit" }}>{subscript}</sub>
      </span>
    </span>
  );
}
