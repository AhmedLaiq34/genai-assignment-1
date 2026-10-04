// Tiny mock backend (no dependencies) so the UI can be tried without the real API.
// Run: node mock/server.mjs   (listens on :8000, the port the Vite proxy expects)
// Endpoints: GET /api/samples, POST /api/universal, POST /api/hard (corruption=none -> identity bypass).
import http from "node:http";

// 1x1 grey PNG, used as every image.
const PNG_B64 =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
const PNG = Buffer.from(PNG_B64, "base64");
const IDS = ["a001", "a002", "a003"];

const send = (res, code, type, body) => {
  res.writeHead(code, { "Content-Type": type });
  res.end(body);
};

http
  .createServer((req, res) => {
    const url = new URL(req.url, "http://x");
    if (req.method === "GET" && url.pathname === "/api/samples") {
      return send(res, 200, "application/json", JSON.stringify(IDS.map((id) => ({ id }))));
    }
    if (req.method === "GET" && url.pathname.startsWith("/api/samples/")) {
      return send(res, 200, "image/png", PNG);
    }
    if (req.method === "POST" && (url.pathname === "/api/universal" || url.pathname === "/api/hard")) {
      const chunks = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => {
        const body = Buffer.concat(chunks).toString("latin1");
        // crude read of the multipart text; enough to echo the fields back
        const field = (n) => (body.match(new RegExp(`name="${n}"\\r\\n\\r\\n([^\\r]*)`)) || [])[1];
        if (!field("sample_id") && !body.includes('name="file"')) {
          return send(res, 422, "application/json", JSON.stringify({ detail: "Provide file or sample_id" }));
        }
        const corruption = field("corruption");
        const params = field("params") ? JSON.parse(field("params")) : { severity: field("severity") };
        const common = {
          input_png_b64: PNG_B64,
          output_png_b64: PNG_B64,
          corruption_applied: corruption,
          params,
          seed: Number(field("seed") || 42),
        };
        if (url.pathname === "/api/universal") {
          return send(
            res,
            200,
            "application/json",
            JSON.stringify({ ...common, timing_ms: { preprocess: 1.2, inference: 3.4, total: 5.0 } })
          );
        }
        // /api/hard: the fake classifier "recognises" the selected corruption.
        // corruption=none -> predicted clean -> identity bypass; anything else -> that expert.
        const routes = {
          none: { id: 0, expert: "identity", probs: [0.93, 0.03, 0.02, 0.02] },
          salt_pepper: { id: 1, expert: "salt", probs: [0.04, 0.88, 0.05, 0.03] },
          gaussian_blur: { id: 2, expert: "blur", probs: [0.07, 0.1, 0.78, 0.05] },
          occlusion: { id: 3, expert: "occlusion", probs: [0.02, 0.03, 0.05, 0.9] },
        };
        const r = routes[corruption] || routes.none;
        const names = ["clean", "salt_pepper", "gaussian_blur", "occlusion"];
        const bypass = r.expert === "identity";
        send(
          res,
          200,
          "application/json",
          JSON.stringify({
            ...common,
            probs: r.probs,
            predicted: names[r.id],
            predicted_id: r.id,
            expert: r.expert,
            identity_bypass: bypass,
            timing_ms: { preprocess: 1.2, classifier: 2.1, expert: bypass ? 0.0 : 3.4, total: bypass ? 3.6 : 7.0 },
          })
        );
      });
      return;
    }
    send(res, 404, "application/json", JSON.stringify({ detail: "Not found" }));
  })
  .listen(8000, () => console.log("mock API on http://localhost:8000"));
