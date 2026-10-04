import { pngSource } from '../api.js';
import { Card, Chip, Download, Icon, ImagePanel, Notice, WeightBar } from './UI.jsx';

const CLASS_LABELS = ['Clean', 'Salt & pepper', 'Gaussian blur', 'Occlusion'];
const EXPERT_LABELS = ['Identity operator', 'Salt & pepper expert', 'Blur expert', 'Occlusion expert'];
const readable = value => String(value ?? '—').replaceAll('_', ' ');
export function inferenceTime(result) {
  const time = result?.timing_ms;
  if (typeof time === 'number') return time;
  if (!time) return null;
  return time.inference ?? (typeof time.classifier === 'number' && typeof time.expert === 'number' ? time.classifier + time.expert : time.total ?? null);
}

function Routing({ workspace, result }) {
  const soft = workspace === 'soft';
  const values = soft ? result?.weights : result?.probs;
  const ranking = values ? values.map((value, i) => ({ value, i })).sort((a, b) => b.value - a.value) : [];
  const order = soft && values ? ranking.map(item => item.i) : [0, 1, 2, 3];
  return <Card title={soft ? 'Gating Network Distribution' : 'Degradation Classifier (Hard Router)'} icon="route" aside={<Chip>{soft ? 'Soft dispatch mode' : '4 classes'}</Chip>}>
    <p className="mb-5 text-sm italic text-muted">{soft ? 'Continuous routing across four restoration branches.' : 'Classifier probabilities determine the selected specialist.'}</p>
    <div className="space-y-2">{order.map(i => <WeightBar key={i} label={(soft ? EXPERT_LABELS : CLASS_LABELS)[i]} value={values?.[i]} emphasized={ranking[0]?.i === i} secondary={soft && ranking[1]?.i === i} badge={ranking[0]?.i === i ? soft ? 'Dominant expert' : 'Top confidence' : soft && ranking[1]?.i === i ? 'Secondary contributor' : null}/>)}</div>
    {soft && values && <div className="mt-5"><p className="mb-3 text-sm italic text-muted">Segmented expert blend</p><div aria-hidden="true" className="flex h-3 overflow-hidden rounded-full bg-track">{order.map((i, rank) => <div key={i} className={['bg-brand-dark', 'bg-brand/60', 'bg-muted', 'bg-track'][rank]} style={{ width: `${values[i] * 100}%` }}/>)}</div><p className="mt-3 text-xs text-muted">{order.map(i => `${EXPERT_LABELS[i]} ${(values[i] * 100).toFixed(1)}%`).join(' · ')}</p></div>}
    {soft ? <div className="mt-5 rounded-control bg-wash p-4 text-sm"><div className="mb-2 flex justify-between gap-3"><span>Routing weight sum</span><strong className="font-mono">{values ? values.reduce((a, b) => a + b, 0).toFixed(3) : '—'}</strong></div><p className="text-muted">Dominant branch: <strong className="text-brand-dark">{readable(result?.dominant)}</strong></p></div> : <div className="mt-5 space-y-3"><p className="rounded-control bg-brand-dark px-3 py-2 text-sm text-white">Predicted corruption: <strong>{readable(result?.predicted)}</strong></p><p className="rounded-control bg-brand-light px-3 py-2 text-sm">Selected expert: <strong>{result?.identity_bypass ? 'Identity bypass' : readable(result?.expert)}</strong></p><p className="rounded-control bg-wash p-3 font-mono text-xs text-muted">Decision rule: argmax(probabilities) → specialist or identity bypass</p></div>}
  </Card>;
}

export default function Results({ workspace, result, preview, loading, error }) {
  const sketch = workspace === 'sketch';
  const output = pngSource(sketch ? result?.sketch_png_b64 : result?.output_png_b64);
  const input = pngSource(sketch ? result?.photo_png_b64 : result?.input_png_b64) || preview;
  const ms = inferenceTime(result);
  const status = loading ? 'Inference in progress' : error ? 'Action needed' : result ? sketch ? 'Sketch complete' : 'Restoration complete' : 'Ready when you are';
  const images = <div className="grid gap-4 sm:grid-cols-2"><ImagePanel title={sketch ? 'Original photo' : result ? 'Input (after corruption)' : 'Input preview'} src={input} loading={loading} caption={sketch ? 'Source photograph' : result ? `Applied: ${readable(result.corruption_applied)}` : 'Select a source to begin'}/><ImagePanel title={sketch ? 'Generated sketch' : 'Restored output'} src={output} output loading={loading} caption={result ? sketch ? result.style_label || `Style ${result.style_id}` : result.identity_bypass ? 'Identity bypass · unchanged input' : 'Model reconstruction' : 'No inference has been run'}/></div>;
  return <div className="stack" aria-busy={loading}>
    {workspace === 'universal' && <Card><div className="flex flex-wrap gap-4"><div className="flex items-center gap-3 rounded-control bg-wash px-4 py-3"><Icon name="clock" className="text-brand"/><div><p className="eyebrow">Inference time</p><strong className="font-mono">{ms == null ? '—' : ms.toFixed(1)} ms</strong></div></div><div className="rounded-control bg-wash px-4 py-3"><p className="eyebrow">Pipeline</p><strong className="text-sm">Unified autoencoder</strong></div></div><div className="mt-4" role="status"><Chip tone={result ? 'success' : 'brand'}>{status}</Chip></div></Card>}
    {['hard', 'soft'].includes(workspace) && <Routing workspace={workspace} result={result}/>}
    {error && <Notice error>{error}</Notice>}
    {workspace === 'hard' || sketch ? <Card title={sketch ? 'Dual Studio Render & Comparison' : 'Side-by-Side Restoration Verification'} icon="image" aside={<Chip>Side-by-side</Chip>}>{images}</Card> : images}
    {!sketch && <Card title="Applied Corruption Settings"><div className="rounded-control bg-wash p-4 text-sm"><div className="mb-3 flex flex-wrap justify-between gap-3"><span>Type: <strong>{readable(result?.corruption_applied)}</strong></span><span>Seed: <strong className="font-mono">{result?.seed ?? '—'}</strong></span></div><pre className="whitespace-pre-wrap break-words font-mono text-xs text-muted">{result ? JSON.stringify(result.params, null, 2) : 'Run restoration to see the settings applied by the server.'}</pre></div></Card>}
    <Card><div className="flex flex-wrap items-center justify-between gap-4"><div><p className="flex items-center gap-2 text-sm"><Icon name="clock" className="text-brand"/><strong>{ms == null ? '—' : ms.toFixed(1)} ms</strong> inference</p>{result?.timing_ms && typeof result.timing_ms === 'object' && <p className="mt-3 font-mono text-xs leading-relaxed text-muted">{Object.entries(result.timing_ms).map(([key, value]) => `${key}: ${value} ms`).join(' · ')}</p>}</div><Download src={output} workspace={workspace}/></div></Card>
    <Notice><strong>{status}.</strong> {loading ? 'Please wait while the server processes your image.' : error ? 'Review the message above, then try again.' : result ? result.mock ? 'Mock response: images and timings demonstrate the interface; no trained model was run.' : 'The images, routing values, and timings above were returned by the backend.' : 'Choose an input and run the pipeline to inspect its result.'}</Notice>
  </div>;
}
