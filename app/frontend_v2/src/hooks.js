import { useEffect, useRef, useState } from 'react';
import { runModel } from './api.js';

export function useObjectUrl(file) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    if (!file) { setUrl(null); return; }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return url;
}

export function useInference(workspace, onResult) {
  const [state, setState] = useState({ loading: false, result: null, error: '' });
  const generation = useRef(0);
  useEffect(() => () => { generation.current++; }, []);
  function clear() {
    generation.current++;
    setState({ loading: false, result: null, error: '' });
  }
  async function run(input) {
    const current = ++generation.current;
    setState({ loading: true, result: null, error: '' });
    try {
      const result = await runModel(workspace, input);
      if (current !== generation.current) return;
      setState({ loading: false, result, error: '' });
      onResult(result);
    } catch (error) {
      if (current === generation.current) setState({ loading: false, result: null, error: error.message });
    }
  }
  return { ...state, run, clear };
}

export function validateFile(file) {
  if (!file) return 'Choose an image first.';
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) return 'Choose a PNG, JPG, or WebP image.';
  if (!file.size) return 'This file is empty. Choose another image.';
  if (file.size > 10 * 1024 * 1024) return 'The image is too large. The maximum size is 10 MB.';
  return '';
}
