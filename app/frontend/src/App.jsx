import { useState } from "react";
import UniversalWorkspace from "./components/UniversalWorkspace.jsx";
import { HardStub, SoftStub, SketchStub } from "./components/Stubs.jsx";

// The four workspaces. Only the first one is functional for now.
const TABS = [
  { id: "universal", label: "Universal Restoration", view: <UniversalWorkspace /> },
  { id: "hard", label: "Hard-Routed Restoration", view: <HardStub /> },
  { id: "soft", label: "Soft Mixture-of-Experts Restoration", view: <SoftStub /> },
  { id: "sketch", label: "Face-to-Sketch Generator", view: <SketchStub /> },
];

export default function App() {
  const [active, setActive] = useState("universal");
  const tab = TABS.find((t) => t.id === active);
  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <nav className="flex flex-wrap gap-2 border-b bg-white p-3">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            className={`rounded px-3 py-2 text-sm ${
              active === t.id ? "bg-blue-600 text-white" : "bg-gray-100 hover:bg-gray-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>
      <main className="mx-auto max-w-5xl p-4">{tab.view}</main>
    </div>
  );
}
