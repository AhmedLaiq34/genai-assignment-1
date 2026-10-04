import { useState } from 'react';
import { Chip, Icon } from './components/UI.jsx';
import SystemPanel from './components/SystemPanel.jsx';
import RestorationWorkspace from './components/RestorationWorkspace.jsx';
import SketchWorkspace from './components/SketchWorkspace.jsx';
import { inferenceTime } from './components/Results.jsx';

const WORKSPACES = [
  { id: 'universal', name: 'Universal Restoration', subtitle: 'A single restoration model for noise, blur, and occlusion.', tag: 'Pipeline 01 · Unified weights' },
  { id: 'hard', name: 'Hard-Routed Restoration', subtitle: 'Classify the corruption, then dispatch to a dedicated restoration expert.', tag: 'Pipeline 02 · Discrete argmax dispatch' },
  { id: 'soft', name: 'Soft Mixture-of-Experts Restoration', subtitle: 'Continuous routing across four experts with learned reconstruction weights.', tag: 'Pipeline 03 · Weighted expert fusion' },
  { id: 'sketch', name: 'Face-to-Sketch Generator', subtitle: 'Transform a portrait with one of three learned sketch-style conditions.', tag: 'Pipeline 04 · Style-conditioned generation' },
];
export default function App() {
  const [active, setActive] = useState('universal');
  const [lastTime, setLastTime] = useState(null);
  const workspace = WORKSPACES.find(item => item.id === active);
  return <div className="flex min-h-screen flex-col"><a href="#workspace" className="sr-only focus:not-sr-only focus:p-4">Skip to workspace</a><header className="border-b border-line bg-surface"><div className="mx-auto flex max-w-studio flex-wrap items-center gap-x-6 gap-y-3 px-5 py-3 xl:flex-nowrap xl:px-8"><div className="flex shrink-0 items-center gap-3"><span className="flex h-9 w-9 items-center justify-center rounded-control bg-brand text-white"><Icon/></span><span className="text-lg font-semibold tracking-tight">Restoration Studio</span><span className="hidden rounded border border-line bg-wash px-2 text-xs italic text-muted 2xl:block">Research Edition</span></div><nav aria-label="Workspaces" className="order-3 grid w-full grid-cols-2 gap-1 md:grid-cols-4 xl:order-none xl:flex xl:flex-1">{WORKSPACES.map(item => <button type="button" key={item.id} aria-current={active === item.id ? 'page' : undefined} onClick={() => setActive(item.id)} className={`min-h-14 rounded border px-3 py-2 text-left text-sm leading-snug xl:flex-1 ${active === item.id ? 'border-brand/30 bg-wash text-brand' : 'border-transparent text-muted hover:bg-wash'}`}>{item.name}</button>)}</nav><div className="ml-auto"><SystemPanel lastTime={lastTime}/></div></div></header><main id="workspace" className="mx-auto w-full max-w-studio flex-1 px-4 py-6 md:px-7 md:py-9"><div className={`mb-gutter flex items-center gap-4 ${active === 'universal' ? 'card' : 'py-1'}`}><span className="hidden h-12 w-12 shrink-0 items-center justify-center rounded-control bg-brand text-white sm:flex"><Icon name={active === 'sketch' ? 'camera' : active === 'universal' ? 'spark' : 'route'} className="h-7 w-7"/></span><div><div className="mb-2 flex flex-wrap items-center gap-3"><h1 className={`font-semibold tracking-tight ${active === 'soft' ? 'text-2xl md:text-3xl' : 'text-xl'}`}>{workspace.name}</h1><Chip>{workspace.tag}</Chip></div><p className="text-sm leading-relaxed text-muted">{workspace.subtitle}</p></div></div>{active === 'sketch' ? <SketchWorkspace onResult={result => setLastTime(inferenceTime(result))}/> : <RestorationWorkspace key={active} workspace={active} onResult={result => setLastTime(inferenceTime(result))}/>}</main><footer className="flex flex-wrap justify-between gap-3 border-t border-line bg-surface px-7 py-5 text-xs italic text-muted"><span>Restoration Studio Research Edition · Computational Imaging Laboratory</span><span>Image Restoration & Neural Generation</span></footer></div>;
}
