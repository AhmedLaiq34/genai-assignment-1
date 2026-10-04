import { useEffect, useState } from 'react';
import { getHealth } from '../api.js';
import { Button } from './UI.jsx';

export default function SystemPanel({ lastTime }) {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState('');
  const [checking, setChecking] = useState(true);
  async function refresh() {
    setChecking(true);
    try { setHealth(await getHealth()); setError(''); }
    catch (e) { setHealth(null); setError(e.message); }
    finally { setChecking(false); }
  }
  useEffect(() => { refresh(); const timer = setInterval(refresh, 30000); return () => clearInterval(timer); }, []);
  const online = health && ['ok', 'online', 'healthy'].includes(health.status);
  return <details className="relative shrink-0"><summary className="cursor-pointer rounded-control border border-line px-3 py-2 text-sm text-muted"><span className={`mr-2 inline-block h-2 w-2 rounded-full ${online ? 'bg-success' : 'bg-danger'}`}/>{checking ? 'Checking…' : online ? 'Online' : 'Offline'} <span className="mx-2 text-line">|</span>System <span className="ml-2 rounded bg-wash px-2 font-mono text-xs">{lastTime == null ? '— ms' : `${lastTime.toFixed(1)} ms`}</span></summary><section aria-label="System panel" className="absolute right-0 z-20 mt-3 w-72 max-w-[85vw] rounded-card border border-line bg-surface p-4 shadow-button"><h2 className="mb-3 font-semibold">System</h2>{health?.mock && <p className="mb-3 text-xs text-warning">Mock API · illustrative results only</p>}{error && <p role="alert" className="mb-3 text-sm text-danger">{error}</p>}<ul className="space-y-2 text-xs">{Object.entries(health?.models || {}).map(([name, info]) => <li key={name} className="flex justify-between gap-3"><span className="break-all">{name}</span><span className={info.loaded ? 'text-success' : 'text-danger'}>{info.loaded ? 'Loaded' : 'Missing'}</span></li>)}</ul>{health && !Object.keys(health.models || {}).length && <p className="text-sm text-muted">No model status reported.</p>}<p className="my-4 text-xs text-muted">Last inference: {lastTime == null ? 'Not run yet' : `${lastTime.toFixed(1)} ms`}</p><Button secondary disabled={checking} onClick={refresh}>Refresh status</Button></section></details>;
}
