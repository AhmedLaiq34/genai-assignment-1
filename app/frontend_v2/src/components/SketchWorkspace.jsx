import { useEffect, useRef, useState } from 'react';
import { listSketchSamples } from '../api.js';
import { useInference, useObjectUrl, validateFile } from '../hooks.js';
import { Button, Card, Chip, Icon, Notice, Upload } from './UI.jsx';
import Results from './Results.jsx';

const STYLES = [
  { id: 1, title: 'Style 1', text: 'First learned sketch-style condition.' },
  { id: 2, title: 'Style 2', text: 'Second learned sketch-style condition.' },
  { id: 3, title: 'Style 3', text: 'Third learned sketch-style condition.' },
];
export default function SketchWorkspace({ onResult }) {
  const [file, setFile] = useState(null);
  const [samples, setSamples] = useState([]);
  const [sampleId, setSampleId] = useState('');
  const [style, setStyle] = useState(1);
  const [error, setError] = useState('');
  const [camera, setCamera] = useState(false);
  const [opening, setOpening] = useState(false);
  const stream = useRef(null);
  const video = useRef(null);
  const requestId = useRef(0);
  const objectUrl = useObjectUrl(file);
  const sample = samples.find(item => item.id === sampleId);
  const preview = file ? objectUrl : sample?.url || null;
  const inference = useInference('sketch', onResult);
  function stopCamera() {
    requestId.current++;
    stream.current?.getTracks().forEach(track => track.stop());
    stream.current = null; setCamera(false); setOpening(false);
  }
  useEffect(() => () => { requestId.current++; stream.current?.getTracks().forEach(track => track.stop()); }, []);
  useEffect(() => { listSketchSamples().then(setSamples).catch(() => setSamples([])); }, []);
  useEffect(() => { if (camera && video.current) video.current.srcObject = stream.current; }, [camera]);
  function pickFile(next) {
    const problem = validateFile(next); setError(problem);
    if (problem) return;
    stopCamera(); inference.clear(); setSampleId(''); setFile(next);
  }
  function pickSample(id) {
    setError(''); stopCamera(); inference.clear(); setFile(null); setSampleId(id);
  }
  async function openCamera() {
    setError(''); setOpening(true);
    const id = ++requestId.current;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Webcam is unavailable. Use localhost or HTTPS, or upload a photo.');
      const media = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      if (id !== requestId.current) { media.getTracks().forEach(track => track.stop()); return; }
      stream.current = media; setCamera(true);
    } catch (e) {
      if (id === requestId.current) setError(e.name === 'NotAllowedError' ? 'Camera permission denied. Allow camera access in your browser or upload a photo.' : e.name === 'NotFoundError' ? 'No webcam found. Connect a camera or upload a photo.' : e.message || 'The camera could not be started. Upload a photo instead.');
    } finally { if (id === requestId.current) setOpening(false); }
  }
  function capture() {
    const source = video.current;
    if (!source?.videoWidth) { setError('The camera is warming up. Try capture again.'); return; }
    const canvas = document.createElement('canvas');
    canvas.width = source.videoWidth; canvas.height = source.videoHeight;
    canvas.getContext('2d').drawImage(source, 0, 0);
    const id = requestId.current;
    canvas.toBlob(blob => {
      if (id !== requestId.current) return;
      if (blob) pickFile(new File([blob], 'webcam-photo.png', { type: 'image/png' }));
      else setError('Could not capture the photo. Please try again.');
    }, 'image/png');
  }
  return <div className="studio-grid"><div className="stack"><Card title="Capture Source" icon="camera" aside={<Chip>Step 1 of 2</Chip>}><div className="mb-5 flex flex-wrap gap-2"><Button secondary disabled={inference.loading || opening || camera} onClick={openCamera}><Icon name="camera"/>{opening ? 'Opening camera…' : 'Use Webcam'}</Button>{(camera || opening) && <Button secondary onClick={stopCamera}>Close camera</Button>}</div>{!camera && samples.length > 0 && <div className="mb-5"><p className="mb-2 text-sm text-muted">Select a sample photo or bring your own.</p><div className="grid grid-cols-3 gap-2 sm:grid-cols-6">{samples.map(item => <button type="button" key={item.id} aria-label={`Select sample ${item.id}`} aria-pressed={sampleId === item.id && !file} disabled={inference.loading} className={`overflow-hidden rounded-control border-2 ${sampleId === item.id && !file ? 'border-brand ring-2 ring-brand-light' : 'border-transparent'}`} onClick={() => pickSample(item.id)}><img src={item.url} alt={item.id.replaceAll('_', ' ')} className="aspect-square w-full object-cover"/></button>)}</div></div>}{camera ? <div><video ref={video} autoPlay playsInline muted aria-label="Webcam preview" className="mb-3 aspect-square w-full rounded-control bg-ink object-cover"/><Button onClick={capture}>Capture photo</Button></div> : <Upload onFile={pickFile} file={file} disabled={inference.loading}/>}{preview && !camera && <div className="mt-4 overflow-hidden rounded-control border border-line"><img src={preview} alt="Selected photo" className="max-h-96 w-full object-contain"/><div className="flex items-center justify-between gap-2 p-3 text-xs text-muted"><span className="truncate">{file ? file.name : sampleId.replaceAll('_', ' ')}</span><button type="button" disabled={inference.loading} className="text-danger" onClick={() => { inference.clear(); setFile(null); setSampleId(''); }}>Remove photo</button></div></div>}{error && <div className="mt-4"><Notice error>{error}</Notice></div>}</Card><Card title="Artistic Sketch Style" icon="spark" aside={<Chip>Step 2 of 2</Chip>}><fieldset disabled={inference.loading}><legend className="sr-only">Sketch style</legend><div className="space-y-3">{STYLES.map(item => <label key={item.id} className={`flex cursor-pointer items-start gap-3 rounded-control border p-4 ${style === item.id ? 'border-brand-light bg-wash' : 'border-transparent bg-track/60'}`}><input type="radio" name="sketch-style" value={item.id} checked={style === item.id} onChange={() => { inference.clear(); setStyle(item.id); }} className="mt-1"/><span><strong className="block text-sm">{item.title}{style === item.id && <span className="ml-2 text-xs italic text-brand">Active</span>}</strong><span className="mt-1 block text-sm text-muted">{item.text}</span></span></label>)}</div><Button className="mt-5 w-full py-4" disabled={(!file && !sampleId) || inference.loading || camera || opening} onClick={() => inference.run({ file, sampleId, style })}><Icon/>{inference.loading ? 'Generating…' : 'Generate Sketch'}</Button></fieldset></Card></div><Results workspace="sketch" {...inference} preview={preview}/></div>;
}
