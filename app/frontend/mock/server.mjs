// Tiny mock backend (no dependencies) so the UI can be tried without the real API.
// Run: node mock/server.mjs   (listens on :8000, the port the Vite proxy expects)
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
    if (req.method === "POST" && url.pathname === "/api/universal") {
      const chunks = [];
      req.on("data", (c) => chunks.push(c));
      req.on("end", () => {
        const body = Buffer.concat(chunks).toString("latin1");
        // crude read of the multipart text; enough to echo the fields back
        const field = (n) => (body.match(new RegExp(`name="${n}"\\r\\n\\r\\n([^\\r]*)`)) || [])[1];
        if (!field("sample_id") && !body.includes('name="file"')) {
          return send(res, 422, "application/json", JSON.stringify({ detail: "Provide file or sample_id" }));
        }
        send(
          res,
          200,
          "application/json",
          JSON.stringify({
            input_png_b64: PNG_B64,
            output_png_b64: PNG_B64,
            corruption_applied: field("corruption"),
            params: field("params") ? JSON.parse(field("params")) : { severity: field("severity") },
            timing_ms: { preprocess: 1.2, inference: 3.4, total: 5.0 },
          })
        );
      });
      return;
    }
    send(res, 404, "application/json", JSON.stringify({ detail: "Not found" }));
  })
  .listen(8000, () => console.log("mock API on http://localhost:8000"));
