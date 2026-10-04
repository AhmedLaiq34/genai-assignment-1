import { useState } from "react";

// Request state shared by the workspaces: loading / error / result for one API call.
// `apiCall` is a function from api.js (runUniversal, runHard, ...) that takes the input-controls values.
export default function useRun(apiCall) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function run(args) {
    setError("");
    setResult(null);
    if (!args.file && !args.sampleId) {
      setError("Choose a sample or upload an image first.");
      return;
    }
    setLoading(true);
    try {
      setResult(await apiCall(args));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return { loading, error, result, run };
}
