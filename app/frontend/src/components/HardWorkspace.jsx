import { runHard } from "../api.js";
import useRun from "../useRun.js";
import InputControls from "./InputControls.jsx";

// Class order is fixed everywhere: the backend returns probs in this order.
const CLASSES = ["clean", "salt_pepper", "gaussian_blur", "occlusion"];

// One horizontal bar per class; the predicted class is drawn in a stronger colour.
function ProbabilityBars({ probs, predicted }) {
  return (
    <div className="my-3">
      <h3 className="mb-1 font-medium">Classifier probabilities</h3>
      {CLASSES.map((name, i) => {
        const percent = (probs[i] * 100).toFixed(1);
        return (
          <div key={name} className="mb-1 flex items-center gap-2 text-sm">
            <span className={`w-32 ${name === predicted ? "font-semibold" : ""}`}>{name}</span>
            <div className="h-4 flex-1 rounded bg-gray-200">
              <div
                className={`h-4 rounded ${name === predicted ? "bg-blue-600" : "bg-blue-300"}`}
                style={{ width: `${percent}%` }}
              />
            </div>
            <span className="w-14 text-right">{percent}%</span>
          </div>
        );
      })}
    </div>
  );
}

export default function HardWorkspace() {
  const { loading, error, result, run } = useRun(runHard);

  const inSrc = result ? `data:image/png;base64,${result.input_png_b64}` : null;
  const outSrc = result ? `data:image/png;base64,${result.output_png_b64}` : null;

  return (
    <div>
      <h2 className="mb-3 text-xl font-semibold">Hard-Routed Restoration</h2>

      {/* input image, corruption settings, Run button (shared with the Universal workspace) */}
      <InputControls loading={loading} error={error} onRun={run} />

      {result && (
        <section className="mt-6">
          <ProbabilityBars probs={result.probs} predicted={result.predicted} />

          <p className="mb-3 text-sm">
            Predicted corruption: <b>{result.predicted}</b>. Selected expert:{" "}
            {result.identity_bypass ? (
              <b>identity bypass (the image is returned unchanged, no expert is run)</b>
            ) : (
              <b>{result.expert} specialist</b>
            )}
          </p>

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
              <figcaption className="text-center text-sm">
                {result.identity_bypass ? "Output (identity bypass)" : "Restored output"}
              </figcaption>
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
                Time (ms): preprocess {result.timing_ms.preprocess}, classifier {result.timing_ms.classifier}, expert{" "}
                {result.timing_ms.expert}, total {result.timing_ms.total}
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
