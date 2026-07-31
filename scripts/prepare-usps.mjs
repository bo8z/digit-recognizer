import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const SOURCE_ROOT =
  "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass";
const OUTPUT_DIR = new URL("../public/usps/", import.meta.url);
const WIDTH = 16;
const HEIGHT = 16;
const FEATURE_COUNT = WIDTH * HEIGHT;

const resources = [
  { source: "usps.bz2", output: "train.bin", expectedCount: 7291 },
  { source: "usps.t.bz2", output: "test.bin", expectedCount: 2007 },
];

mkdirSync(OUTPUT_DIR, { recursive: true });

for (const resource of resources) {
  const compressedPath = join(tmpdir(), `digit-recognizer-${resource.source}`);
  const sourceUrl = `${SOURCE_ROOT}/${resource.source}`;

  console.log(`Downloading ${sourceUrl}`);
  const response = await fetch(sourceUrl);
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status} ${response.statusText}`);
  }
  writeFileSync(compressedPath, Buffer.from(await response.arrayBuffer()));

  console.log(`Decompressing ${resource.source}`);
  const text = execFileSync("bzip2", ["-dc", compressedPath], {
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  });
  rmSync(compressedPath);

  const { labels, pixels } = parseLibSvm(text);
  if (labels.length !== resource.expectedCount) {
    throw new Error(
      `${resource.source}: expected ${resource.expectedCount} samples, found ${labels.length}`,
    );
  }

  const output = encodeDataset(labels, pixels);
  writeFileSync(new URL(resource.output, OUTPUT_DIR), output);
  console.log(
    `Wrote ${resource.output}: ${labels.length} samples, ${(output.byteLength / 1024 / 1024).toFixed(2)} MiB`,
  );
}

function parseLibSvm(text) {
  const lines = text.trim().split(/\r?\n/);
  const labels = new Uint8Array(lines.length);
  const pixels = new Uint8Array(lines.length * FEATURE_COUNT);

  lines.forEach((line, sampleIndex) => {
    const fields = line.trim().split(/\s+/);
    const rawLabel = Number(fields[0]);
    if (!Number.isInteger(rawLabel) || rawLabel < 1 || rawLabel > 10) {
      throw new Error(`Invalid label in sample ${sampleIndex + 1}: ${fields[0]}`);
    }
    labels[sampleIndex] = rawLabel - 1;

    const sampleOffset = sampleIndex * FEATURE_COUNT;
    for (let fieldIndex = 1; fieldIndex < fields.length; fieldIndex++) {
      const [rawFeature, rawValue] = fields[fieldIndex].split(":");
      const feature = Number(rawFeature) - 1;
      const value = Number(rawValue);
      if (
        !Number.isInteger(feature) ||
        feature < 0 ||
        feature >= FEATURE_COUNT ||
        !Number.isFinite(value)
      ) {
        throw new Error(
          `Invalid feature in sample ${sampleIndex + 1}: ${fields[fieldIndex]}`,
        );
      }
      pixels[sampleOffset + feature] = Math.round(
        Math.max(0, Math.min(1, (value + 1) / 2)) * 255,
      );
    }
  });

  return { labels, pixels };
}

function encodeDataset(labels, pixels) {
  const headerSize = 16;
  const output = Buffer.allocUnsafe(headerSize + labels.length + pixels.length);
  output.write("USPSBIN1", 0, "ascii");
  output.writeUInt32LE(labels.length, 8);
  output.writeUInt16LE(WIDTH, 12);
  output.writeUInt16LE(HEIGHT, 14);
  Buffer.from(labels).copy(output, headerSize);
  Buffer.from(pixels).copy(output, headerSize + labels.length);
  return output;
}
