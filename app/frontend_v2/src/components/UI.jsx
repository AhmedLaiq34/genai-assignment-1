import { useId, useState } from 'react';

export function Icon({ name = 'spark', className = '' }) {
  const paths = {
    spark: 'm12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z',
    upload: 'M12 16V4m-4 4 4-4 4 4M4 15v5h16v-5',
    download: 'M12 3v13m-4-4 4 4 4-4M4 17v4h16v-4',
    image: 'M3 4h18v16H3V4Zm0 12 5-5 5 5 3-3 5 5M16 8h.01',
    settings: 'M4 6h16M4 12h16M4 18h16M8 3v6m8 0v6m-7 0v6',
    camera: 'M3 7h4l2-3h6l2 3h4v13H3V7Zm13 6a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z',
    route: 'M6 3v18m0-9h12M18 3v18M3 3h6v4H3V3Zm12 0h6v4h-6V3Zm0 14h6v4h-6v-4Z',
    clock: 'M12 8v5l3 2M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z',
  };
  return <svg aria-hidden="true" viewBox="0 0 24 24" className={`h-5 w-5 shrink-0 ${className}`} fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round"><path d={paths[name] || paths.spark}/></svg>;
}
export function Card({ title, icon, aside, children, className = '' }) {
  return <section className={`card ${className}`}>{title && <div className="mb-5 flex flex-wrap items-center justify-between gap-3"><h2 className="flex items-center gap-2 text-lg font-semibold">{icon && <Icon name={icon} className="text-brand"/>}{title}</h2>{aside}</div>}{children}</section>;
}
export function Button({ children, secondary, className = '', ...props }) {
  return <button type="button" className={`button ${secondary ? 'button-secondary' : 'button-primary'} ${className}`} {...props}>{children}</button>;
}
export function Chip({ children, tone = 'brand' }) {
  return <span className={`inline-flex items-center gap-2 rounded-control px-2 py-1 text-xs font-medium ${tone === 'success' ? 'bg-success-soft text-success' : 'bg-wash text-brand-dark'}`}>{children}</span>;
}
export function Badge({ children, active }) { return <span className={`rounded px-2 py-1 text-xs ${active ? 'bg-brand text-white' : 'bg-track text-muted'}`}>{children}</span>; }
export function Tabs({ label, options, value, onChange, disabled }) {
  return <div role="group" aria-label={label} className="flex flex-wrap gap-1 rounded-control bg-wash p-1">{options.map(option => <button key={option.value} type="button" disabled={disabled} aria-pressed={value === option.value} onClick={() => onChange(option.value)} className={`min-h-10 flex-1 rounded px-3 py-2 text-sm transition-colors ${value === option.value ? 'bg-brand font-semibold text-white shadow-card' : 'text-muted hover:bg-brand-light'}`}>{option.label}</button>)}</div>;
}
export function Notice({ error, children }) {
  return <div role={error ? 'alert' : 'status'} className={`rounded-control p-4 text-sm leading-relaxed ${error ? 'bg-danger-soft text-danger' : 'bg-wash text-muted'}`}>{children}</div>;
}
export function WeightBar({ label, value, emphasized, secondary, badge }) {
  const percent = value == null ? null : value * 100;
  return <div className={`rounded-control p-3 ${emphasized ? 'bg-wash' : ''}`}><div className="mb-2 flex flex-wrap items-center gap-2 text-sm"><span className={emphasized ? 'font-semibold text-brand-dark' : 'text-muted'}>{label}</span>{badge && <Badge active={emphasized}>{badge}</Badge>}<span className="ml-auto font-mono">{percent == null ? '—' : `${percent.toFixed(1)}%`}</span></div><div role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined} aria-valuetext={percent == null ? 'Waiting for inference' : undefined} className="h-2.5 overflow-hidden rounded-full bg-track"><div className={`h-full rounded-full transition-all ${emphasized ? 'bg-brand-dark' : secondary ? 'bg-brand/60' : 'bg-muted/40'}`} style={{ width: `${percent || 0}%` }}/></div></div>;
}
export function ImagePanel({ title, src, output, loading, caption }) {
  const [broken, setBroken] = useState(null);
  return <figure className="min-w-0 overflow-hidden rounded-card bg-wash shadow-card"><figcaption className="flex min-h-12 items-center gap-2 px-3 py-3 text-sm"><span className={`h-2 w-2 shrink-0 rounded-full ${output ? 'bg-brand' : 'bg-warning'}`}/><span className="font-medium">{title}</span></figcaption>{src && broken !== src ? <img src={src} alt={title} onError={() => setBroken(src)} className="aspect-square w-full bg-track object-contain"/> : <div className={`image-empty ${loading ? 'animate-pulse' : ''}`}><Icon name="image" className="h-10 w-10 text-brand/40"/><span>{loading ? 'Processing your image…' : broken === src && src ? 'Image could not be displayed.' : output ? 'Your result will appear here' : 'Choose a sample or upload an image'}</span></div>}<div className="min-h-10 bg-surface px-3 py-3 text-xs text-muted">{caption}</div></figure>;
}
export function Upload({ onFile, disabled, file }) {
  const id = useId();
  return <div onDragOver={e => e.preventDefault()} onDrop={e => { e.preventDefault(); if (!disabled && e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); }} className="relative rounded-card bg-wash p-6 text-center"><div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-surface text-brand"><Icon name="upload" className="h-7 w-7"/></div><label htmlFor={id} className="mb-2 block font-semibold">{file ? file.name : 'Drop an image here or browse'}</label><p id={`${id}-help`} className="mb-4 text-sm text-muted">PNG, JPG, WebP · up to 10 MB</p><input id={id} aria-describedby={`${id}-help`} aria-label="Upload image" type="file" accept="image/png,image/jpeg,image/webp" disabled={disabled} className="block w-full min-w-0 text-xs text-muted file:mr-3 file:rounded-control file:border-0 file:bg-surface file:px-3 file:py-2 file:font-semibold file:text-brand" onChange={e => { if (e.target.files[0]) onFile(e.target.files[0]); e.target.value = ''; }}/></div>;
}
export function Download({ src, workspace }) {
  if (!src) return <Button disabled><Icon name="download"/>Download {workspace === 'sketch' ? 'sketch' : 'result'}</Button>;
  return <a className="button button-primary" href={src} download={`${workspace}-result.png`}><Icon name="download"/>Download {workspace === 'sketch' ? 'sketch' : 'result'} (PNG)</a>;
}
