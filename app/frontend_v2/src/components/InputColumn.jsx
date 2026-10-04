import { useEffect, useState } from 'react';
import { listSamples } from '../api.js';
import { validateFile } from '../hooks.js';
import { Button, Card, Chip, Icon, Notice, Tabs, Upload } from './UI.jsx';

export const CORRUPTIONS = [
  { value: 'none', label: 'None' }, { value: 'salt_pepper', label: 'Salt & pepper' },
  { value: 'gaussian_blur', label: 'Gaussian blur' }, { value: 'occlusion', label: 'Occlusion' },
];
const PARAMETERS = {
  salt_pepper: [{ key: 'p', label: 'Noise density', min: 0, max: 1, step: 0.01, initial: 0.15 }],
  gaussian_blur: [{ key: 'kernel', label: 'Kernel (odd, 3–15)', min: 3, max: 15, step: 2, initial: 5 }, { key: 'sigma', label: 'Sigma', min: 0.01, step: 0.01, initial: 1.5 }],
  occlusion: [{ key: 'n_rects', label: 'Rectangles', min: 1, max: 5, step: 1, initial: 2 }, { key: 'coverage', label: 'Coverage', min: 0.02, max: 0.6, step: 0.01, initial: 0.2 }],
};

export default function InputColumn({ workspace, input, setInput, loading, onRun, onPreview }) {
  const [samples, setSamples] = useState([]);
  const [sampleError, setSampleError] = useState('');
  const [fetching, setFetching] = useState(true);
  const [fileError, setFileError] = useState('');
  const [custom, setCustom] = useState(false);
  const [params, setParams] = useState({});
  async function load() {
    setFetching(true); setSampleError('');
    try { setSamples(await listSamples()); } catch (e) { setSampleError(e.message); }
    finally { setFetching(false); }
  }
  useEffect(() => { load(); }, []);
  function update(values) { setInput({ ...input, ...values }); }
  function pickFile(file) {
    const error = validateFile(file); setFileError(error);
    if (error) return;
    update({ file, sampleId: '', corruption: 'none', alreadyCorrupted: false });
    onPreview(null);
  }
  function changeCorruption(corruption) {
    setCustom(false); setParams({}); update({ corruption, params: null });
  }
  function toggleCustom(checked) {
    setCustom(checked);
    const values = Object.fromEntries((PARAMETERS[input.corruption] || []).map(p => [p.key, p.initial]));
    setParams(values); update({ params: checked ? values : null });
  }
  return <form onSubmit={e => { e.preventDefault(); onRun(input); }} className="stack">
    <Card title={workspace === 'universal' ? 'Sample Reference Images' : 'Input Raster Selection'} icon="image" aside={<Chip>{samples.length} samples</Chip>}>
      <p className="mb-4 text-sm text-muted">Select a clean sample or bring your own image.</p>
      {fetching ? <Notice>Loading samples…</Notice> : sampleError ? <Notice error>{sampleError} <button type="button" className="underline" onClick={load}>Retry samples</button></Notice> : !samples.length ? <Notice>No samples available. Upload an image below.</Notice> : <div className="mb-5 grid grid-cols-4 gap-2">{samples.map(sample => <button type="button" key={sample.id} aria-label={`Select sample ${sample.id}`} aria-pressed={input.sampleId === sample.id} disabled={loading} className={`relative overflow-hidden rounded-control border-2 ${input.sampleId === sample.id ? 'border-brand ring-2 ring-brand-light' : 'border-transparent'}`} onClick={() => { setFileError(''); update({ sampleId: sample.id, file: null, alreadyCorrupted: false }); onPreview(sample.url); }}><img src={sample.url} alt={sample.id.replaceAll('_', ' ')} className="aspect-square w-full object-cover"/><span className="absolute inset-x-0 bottom-0 truncate bg-ink/70 px-1 py-1 text-[10px] text-white">{sample.id.replaceAll('_', ' ')}</span></button>)}</div>}
      <Upload onFile={pickFile} disabled={loading} file={input.file}/>
      {fileError && <div className="mt-3"><Notice error>{fileError}</Notice></div>}
      {input.file && <label className="mt-4 flex items-start gap-2 text-sm"><input type="checkbox" checked={input.alreadyCorrupted} disabled={loading} onChange={e => update({ alreadyCorrupted: e.target.checked, corruption: 'none', params: null })}/><span>Already corrupted — restore without adding corruption</span></label>}
    </Card>
    <Card title="Synthetic Degradation Pipeline" icon="settings" aside={<button type="button" disabled={loading} className="text-sm text-brand" onClick={() => { setCustom(false); update({ corruption: 'none', severity: 'medium', seed: 42, params: null }); }}>Reset</button>}>
      <fieldset disabled={loading} className="min-w-0 space-y-5">
        <div><p className="eyebrow mb-2">Corruption type</p><Tabs label="Corruption type" options={CORRUPTIONS} value={input.corruption} disabled={loading || input.alreadyCorrupted} onChange={changeCorruption}/></div>
        {input.alreadyCorrupted && <p className="text-xs text-success">Additional corruption is disabled for this upload.</p>}
        {input.corruption !== 'none' && <><div><p className="eyebrow mb-2">Severity degree</p><Tabs label="Severity" options={['low', 'medium', 'high'].map(value => ({ value, label: value[0].toUpperCase() + value.slice(1) }))} value={input.severity} onChange={severity => { setCustom(false); update({ severity, params: null }); }}/></div><label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={custom} onChange={e => toggleCustom(e.target.checked)}/>Use custom parameters</label>{custom && <div className="grid grid-cols-2 gap-3 rounded-control bg-wash p-4">{PARAMETERS[input.corruption]?.map(p => <label key={p.key} className="text-sm text-muted">{p.label}<input aria-label={p.label} type="number" required min={p.min} max={p.max} step={p.step} value={params[p.key] ?? ''} onChange={e => { const next = { ...params, [p.key]: e.target.value === '' ? '' : Number(e.target.value) }; setParams(next); update({ params: next }); }}/></label>)}</div>}</>}
        <label className="block rounded-control bg-wash p-4"><span className="field-label">Stochastic seed</span><input aria-label="Stochastic seed" type="number" required min="0" max="4294967295" step="1" value={input.seed} onChange={e => update({ seed: e.target.value })}/><span className="mt-2 block text-xs text-muted">Use the same seed to repeat a corruption.</span></label>
        <Button type="submit" className="w-full py-4" disabled={loading || (!input.file && !input.sampleId)}><Icon name={workspace === 'hard' ? 'route' : 'spark'}/>{loading ? 'Processing…' : workspace === 'universal' ? 'Run Universal Restoration' : workspace === 'hard' ? 'Classify & Route Restoration' : 'Run Soft MoE Inference'}</Button>
      </fieldset>
    </Card>
    <Notice>{workspace === 'soft' ? 'Four experts contribute to the reconstruction. The gating network chooses their weights.' : workspace === 'hard' ? 'The highest classifier probability selects one specialist. Clean images use the identity bypass.' : 'One unified autoencoder restores the input. Images are processed at the model’s native resolution.'}</Notice>
  </form>;
}
