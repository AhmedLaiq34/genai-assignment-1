import { useState } from 'react';
import { useInference, useObjectUrl } from '../hooks.js';
import InputColumn from './InputColumn.jsx';
import Results from './Results.jsx';

export default function RestorationWorkspace({ workspace, onResult }) {
  const [input, setInput] = useState({ file: null, sampleId: '', corruption: 'none', severity: 'medium', seed: 42, params: null, alreadyCorrupted: false });
  const [samplePreview, setSamplePreview] = useState(null);
  const filePreview = useObjectUrl(input.file);
  const inference = useInference(workspace, onResult);
  // Changing settings clears old results so the displayed metadata always matches the run.
  function updateInput(value) { inference.clear(); setInput(value); }
  return <div className="studio-grid"><InputColumn workspace={workspace} input={input} setInput={updateInput} loading={inference.loading} onRun={inference.run} onPreview={setSamplePreview}/><Results workspace={workspace} {...inference} preview={filePreview || samplePreview}/></div>;
}
