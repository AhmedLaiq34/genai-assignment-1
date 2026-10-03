// Placeholder workspaces: final layout is stubbed, nothing is wired to the backend yet.

const CLASSES = ["clean", "salt_pepper", "gaussian_blur", "occlusion"];

function NotImplemented({ title, children }) {
  return (
    <div>
      <h2 className="mb-2 text-xl font-semibold">{title}</h2>
      <p className="mb-4 rounded border border-yellow-300 bg-yellow-50 p-3 text-sm">
        Not implemented yet. The layout below is a placeholder.
      </p>
      {children}
    </div>
  );
}

// Horizontal bars with labels; empty until the backend returns real numbers.
function BarPlaceholder({ title, labels }) {
  return (
    <div className="my-4">
      <h3 className="mb-1 font-medium">{title}</h3>
      {labels.map((l) => (
        <div key={l} className="mb-1 flex items-center gap-2 text-sm">
          <span className="w-32">{l}</span>
          <div className="h-4 flex-1 rounded bg-gray-200" />
          <span className="w-12 text-right text-gray-400">--</span>
        </div>
      ))}
    </div>
  );
}

function ImageBox({ label }) {
  return (
    <div className="flex h-48 w-48 items-center justify-center rounded border-2 border-dashed text-sm text-gray-400">
      {label}
    </div>
  );
}

export function HardStub() {
  return (
    <NotImplemented title="Hard-Routed Restoration">
      <div className="flex gap-4">
        <ImageBox label="input" />
        <ImageBox label="restored output" />
      </div>
      <BarPlaceholder title="Classifier probabilities" labels={CLASSES} />
      <p className="text-sm text-gray-500">Predicted class / selected expert: --</p>
    </NotImplemented>
  );
}

export function SoftStub() {
  return (
    <NotImplemented title="Soft Mixture-of-Experts Restoration">
      <div className="flex gap-4">
        <ImageBox label="input" />
        <ImageBox label="restored output" />
      </div>
      <BarPlaceholder title="Gate weights (w0 = identity)" labels={["w0", "w1", "w2", "w3"]} />
      <p className="text-sm text-gray-500">Dominant branch: --</p>
    </NotImplemented>
  );
}

export function SketchStub() {
  return (
    <NotImplemented title="Face-to-Sketch Generator">
      <div className="mb-3 flex gap-2">
        {["Style 1", "Style 2", "Style 3"].map((s) => (
          <button key={s} disabled className="rounded border px-3 py-1 text-sm text-gray-400">
            {s}
          </button>
        ))}
        <button disabled className="rounded border px-3 py-1 text-sm text-gray-400">
          Use webcam
        </button>
      </div>
      <div className="flex gap-4">
        <ImageBox label="photo" />
        <ImageBox label="sketch" />
      </div>
    </NotImplemented>
  );
}
