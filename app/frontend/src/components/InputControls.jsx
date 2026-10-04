import { useEffect, useState } from "react";
import { listSamples } from "../api.js";

const CORRUPTIONS = ["none", "salt_pepper", "gaussian_blur", "occlusion"];
const SEVERITIES = ["low", "medium", "high", "custom"];

// Default custom parameters, one set per corruption type.
const DEFAULT_PARAMS = {
  salt_pepper: { p: 0.08 },
  gaussian_blur: { kernel: 5, sigma: 1.5 },
  occlusion: { n_rects: 2, coverage: 0.2 },
};

// The input controls shared by the Universal and the Hard-Routed workspaces:
// sample picker, upload, corruption type / severity / custom parameters, seed and the Run button.
// It keeps its own state; when Run is clicked it calls onRun({file, sampleId, corruption, severity, params, seed}).
// The parent owns the request (loading flag, error text), because it knows which API it calls.
export default function InputControls({ loading, error, onRun }) {
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

  function run() {
    const useCustom = corruption !== "none" && severity === "custom";
    onRun({
      file,
      sampleId,
      corruption,
      severity,
      params: useCustom ? custom[corruption] : null,
      seed,
    });
  }

  const cp = custom[corruption]; // undefined when corruption is "none"

  return (
    <div>
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
    </div>
  );
}
