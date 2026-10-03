import { useEffect, useState } from "react";
import { listSamples, runUniversal } from "../api.js";

const CORRUPTIONS = ["none", "salt_pepper", "gaussian_blur", "occlusion"];
const SEVERITIES = ["low", "medium", "high", "custom"];

// Default custom parameters, one set per corruption type.
const DEFAULT_PARAMS = {
  salt_pepper: { p: 0.08 },
  gaussian_blur: { kernel: 5, sigma: 1.5 },
  occlusion: { n_rects: 2, coverage: 0.2 },
};

export default function UniversalWorkspace() {
  // --- input selection ---
  const [samples, setSamples] = useState([]);
  const [samplesError, setSamplesError] = useState("");
  const [sampleId, setSampleId] = useState(null);
  const [file, setFile] = useState(null);
  const [filePreview, setFilePreview] = useState(null);

  // --- corruption settings ---
  const [corruption, setCorruption] = useState("gaussian_blur");
  const [severity, setSeverity] = useState("medium");
  const [custom, setCustom] = useState(DEFAULT_PARAMS);
  const [seed, setSeed] = useState(42);

  // --- request state ---
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  // Load the sample list once.
  useEffect(() => {
    listSamples()
      .then(setSamples)
      .catch((e) => setSamplesError(e.message));
  }, []);

  function pickFile(f) {
    setFile(f);
    setSampleId(null); // an upload replaces the sample choice
    setFilePreview(f ? URL.createObjectURL(f) : null);
  }

  function pickSample(id) {
    setSampleId(id);
    setFile(null);
    setFilePreview(null);
  }

  function setCustomField(name, value) {
    setCustom({ ...custom, [corruption]: { ...custom[corruption], [name]: Number(value) } });
  }

  async function run() {
    setError("");
    setResult(null);
    if (!file && !sampleId) {
      setError("Choose a sample or upload an image first.");
      return;
    }
    setLoading(true);
    try {
      const useCustom = corruption !== "none" && severity === "custom";
      const data = await runUniversal({
        file,
        sampleId,
        corruption,
        severity,
        params: useCustom ? custom[corruption] : null,
        seed,
      });
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const inSrc = result ? `data:image/png;base64,${result.input_png_b64}` : null;
  const outSrc = result ? `data:image/png;base64,${result.output_png_b64}` : null;
  const cp = custom[corruption]; // undefined when corruption is "none"

  return (
    <div>
      <h2 className="mb-3 text-xl font-semibold">Universal Restoration</h2>

      {/* 1. choose input */}
      <section className="mb-4">
        <h3 className="mb-1 font-medium">1. Input image</h3>
        {samplesError && <p className="text-sm text-red-600">Could not load samples: {samplesError}</p>}
        <div className="mb-2 flex flex-wrap gap-2">
          {samples.map((s) => (
            <img
              key={s.id}
              src={s.url}
              alt={s.id}
              onClick={() => pickSample(s.id)}
              className={`h-20 w-20 cursor-pointer rounded border-4 object-cover ${
                sampleId === s.id ? "border-blue-600" : "border-transparent"
              }`}
            />
          ))}
        </div>
        <label className="text-sm">
          Or upload:{" "}
          <input type="file" accept="image/*" onChange={(e) => pickFile(e.target.files[0] || null)} />
        </label>
        {filePreview && <img src={filePreview} alt="upload" className="mt-2 h-20 w-20 rounded object-cover" />}
      </section>

      {/* 2. corruption settings */}
      <section className="mb-4">
        <h3 className="mb-1 font-medium">2. Corruption</h3>
        <div className="flex flex-wrap items-center gap-4 text-sm">
          <label>
            Type{" "}
            <select className="border p-1" value={corruption} onChange={(e) => setCorruption(e.target.value)}>
              {CORRUPTIONS.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
          {corruption !== "none" && (
            <label>
              Severity{" "}
              <select className="border p-1" value={severity} onChange={(e) => setSeverity(e.target.value)}>
                {SEVERITIES.map((s) => (
                  <option key={s}>{s}</option>
                ))}
              </select>
            </label>
          )}
          <label>
            Seed{" "}
            <input
              type="number"
              className="w-20 border p-1"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </label>
        </div>

        {/* custom parameters, only when severity = custom */}
        {cp && severity === "custom" && (
          <div className="mt-2 flex flex-wrap gap-4 text-sm">
            {Object.keys(cp).map((name) => (
              <label key={name}>
                {name}{" "}
                <input
                  type="number"
                  step="any"
                  className="w-24 border p-1"
                  value={cp[name]}
                  onChange={(e) => setCustomField(name, e.target.value)}
                />
              </label>
            ))}
          </div>
        )}
      </section>

      {/* 3. run */}
      <button
        onClick={run}
        disabled={loading}
        className="rounded bg-blue-600 px-4 py-2 text-white disabled:opacity-50"
      >
        {loading ? "Running..." : "Run"}
      </button>
      {error && (
        <p className="mt-3 rounded border border-red-300 bg-red-50 p-2 text-sm text-red-700">Error: {error}</p>
      )}

      {/* results */}
      {result && (
        <section className="mt-6">
          <div className="flex flex-wrap gap-6">
            <figure>
              <img
                src={inSrc}
                alt="input"
                className="h-64 w-64 rounded border object-contain"
                style={{ imageRendering: "pixelated" }}
              />
              <figcaption className="text-center text-sm">Input (after corruption)</figcaption>
            </figure>
            <figure>
              <img
                src={outSrc}
                alt="restored"
                className="h-64 w-64 rounded border object-contain"
                style={{ imageRendering: "pixelated" }}
              />
              <figcaption className="text-center text-sm">Restored output</figcaption>
            </figure>
          </div>
          <div className="mt-3 text-sm">
            <p>
              Corruption applied: <b>{String(result.corruption_applied)}</b>
            </p>
            <p>
              Parameters: <code>{JSON.stringify(result.params)}</code>
            </p>
            {result.timing_ms && (
              <p>
                Time (ms): preprocess {result.timing_ms.preprocess}, inference {result.timing_ms.inference}, total{" "}
                {result.timing_ms.total}
              </p>
            )}
            <a
              href={outSrc}
              download="restored.png"
              className="mt-2 inline-block rounded border px-3 py-1 hover:bg-gray-100"
            >
              Download output
            </a>
          </div>
        </section>
      )}
    </div>
  );
}
