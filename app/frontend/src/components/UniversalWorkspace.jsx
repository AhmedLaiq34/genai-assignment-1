import { runUniversal } from "../api.js";
import useRun from "../useRun.js";
import InputControls from "./InputControls.jsx";

export default function UniversalWorkspace() {
  const { loading, error, result, run } = useRun(runUniversal);

  const inSrc = result ? `data:image/png;base64,${result.input_png_b64}` : null;
  const outSrc = result ? `data:image/png;base64,${result.output_png_b64}` : null;

  return (
    <div>
      <h2 className="mb-3 text-xl font-semibold">Universal Restoration</h2>

      {/* input image, corruption settings, Run button (shared with the Hard-Routed workspace) */}
      <InputControls loading={loading} error={error} onRun={run} />

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
