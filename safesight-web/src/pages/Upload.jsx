import { useState, useRef, useCallback, useEffect } from 'react';
import ViolationsDialog from '../components/ViolationsDialog';
import './Upload.css';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

const PROCESS_STEPS = [
    'Opening video & preparing frames',
    'Running Faster R-CNN detection',
    'Compiling violation report',
    'Finalizing output',
];

export default function Upload() {
    const [file, setFile] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const [processing, setProcessing] = useState(false);
    const [currentStep, setCurrentStep] = useState(-1);
    const [progress, setProgress] = useState(0);
    const [done, setDone] = useState(false);
    const [showDialog, setShowDialog] = useState(false);
    const [jobId, setJobId] = useState(null);
    const [results, setResults] = useState(null);
    const [errorMsg, setErrorMsg] = useState('');
    const [statusMessage, setStatusMessage] = useState('');
    const inputRef = useRef(null);
    const pollRef = useRef(null);

    const handleFile = useCallback((f) => {
        if (f && (f.type.startsWith('video/') || /\.(mp4|avi|mov|mkv)$/i.test(f.name))) {
            setFile(f);
            setProcessing(false);
            setCurrentStep(-1);
            setProgress(0);
            setDone(false);
            setJobId(null);
            setResults(null);
            setErrorMsg('');
            setStatusMessage('');
        }
    }, []);

    const onDrop = (e) => {
        e.preventDefault();
        setDragOver(false);
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    };

    // ── Poll for job status ──────────────────────────────────────────────────
    const startPolling = useCallback((id) => {
        if (pollRef.current) clearInterval(pollRef.current);

        pollRef.current = setInterval(async () => {
            try {
                const res = await fetch(`${API_BASE}/status/${id}`);
                const data = await res.json();

                if (data.status === 'error') {
                    clearInterval(pollRef.current);
                    setProcessing(false);
                    setErrorMsg(data.message || 'Processing failed');
                    return;
                }

                setProgress(data.progress || 0);
                setCurrentStep(data.current_step ?? -1);
                setStatusMessage(data.message || '');

                if (data.status === 'done') {
                    clearInterval(pollRef.current);
                    setProcessing(false);
                    setDone(true);
                    setProgress(100);
                    setCurrentStep(PROCESS_STEPS.length);

                    // Fetch full results
                    const resResult = await fetch(`${API_BASE}/results/${id}`);
                    const resultData = await resResult.json();
                    setResults(resultData);
                }
            } catch (err) {
                console.error('Polling error:', err);
            }
        }, 1000);
    }, []);

    // Clean up interval on unmount
    useEffect(() => {
        return () => {
            if (pollRef.current) clearInterval(pollRef.current);
        };
    }, []);

    // ── Upload & start processing ────────────────────────────────────────────
    const startProcessing = async () => {
        if (!file) return;

        setProcessing(true);
        setDone(false);
        setProgress(0);
        setCurrentStep(0);
        setErrorMsg('');
        setResults(null);
        setStatusMessage('Uploading video...');

        try {
            const formData = new FormData();
            formData.append('video', file);

            const res = await fetch(`${API_BASE}/upload`, {
                method: 'POST',
                body: formData,
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || 'Upload failed');
            }

            const data = await res.json();
            setJobId(data.job_id);
            setStatusMessage('Processing started...');

            // Start polling for progress
            startPolling(data.job_id);
        } catch (err) {
            setProcessing(false);
            setErrorMsg(err.message || 'Failed to upload video');
        }
    };

    const resetUpload = () => {
        if (pollRef.current) clearInterval(pollRef.current);
        setFile(null);
        setProcessing(false);
        setCurrentStep(-1);
        setProgress(0);
        setDone(false);
        setJobId(null);
        setResults(null);
        setErrorMsg('');
        setStatusMessage('');
    };

    const downloadResult = () => {
        if (jobId) {
            window.open(`${API_BASE}/download/${jobId}`, '_blank');
        }
    };

    const formatSize = (bytes) => {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    };

    const violationCount = results?.violations?.length || 0;

    return (
        <div className="upload-page">
            <div className="upload-header">
                <div className="container">
                    <p className="section-label">Analysis</p>
                    <h1 className="section-title">Upload footage</h1>
                    <p className="section-subtitle">
                        Upload your CCTV video file to run the safety analysis pipeline.
                        Supported formats include MP4, AVI, and MOV.
                    </p>
                </div>
            </div>

            <div className="upload-body">
                <div className="container">
                    {/* Drop Zone */}
                    {!file && (
                        <div
                            className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
                            onClick={() => inputRef.current?.click()}
                            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                            onDragLeave={() => setDragOver(false)}
                            onDrop={onDrop}
                        >
                            <div className="drop-zone-icon">↑</div>
                            <h3>Drop your video file here</h3>
                            <p>or click to browse from your computer</p>
                            <div className="file-types">
                                <span className="file-type-badge">.mp4</span>
                                <span className="file-type-badge">.avi</span>
                                <span className="file-type-badge">.mov</span>
                                <span className="file-type-badge">.mkv</span>
                            </div>
                            <input
                                ref={inputRef}
                                type="file"
                                accept="video/*"
                                style={{ display: 'none' }}
                                onChange={(e) => e.target.files.length && handleFile(e.target.files[0])}
                            />
                        </div>
                    )}

                    {/* File Selected */}
                    {file && (
                        <>
                            <div className="file-info">
                                <div className="file-meta">
                                    <div className="file-icon">▶</div>
                                    <div className="file-details">
                                        <h4>{file.name}</h4>
                                        <span>{formatSize(file.size)}</span>
                                    </div>
                                </div>
                                {!processing && !done && (
                                    <button className="remove-file" onClick={resetUpload}>✕</button>
                                )}
                            </div>

                            {/* Error */}
                            {errorMsg && (
                                <div className="error-banner" style={{
                                    marginTop: 16,
                                    padding: '12px 20px',
                                    background: 'rgba(255, 50, 50, 0.1)',
                                    border: '1px solid rgba(255, 50, 50, 0.3)',
                                    borderRadius: 8,
                                    color: '#ff5252',
                                    fontSize: '0.9rem',
                                }}>
                                    ⚠ {errorMsg}
                                </div>
                            )}

                            {/* Start Button */}
                            {!processing && !done && currentStep === -1 && (
                                <div style={{ marginTop: 24 }}>
                                    <button className="btn btn-primary" onClick={startProcessing}>
                                        Run Analysis →
                                    </button>
                                </div>
                            )}

                            {/* Progress */}
                            {(processing || done) && (
                                <div className="progress-section">
                                    <div className="progress-header">
                                        <span>{done ? 'Complete' : statusMessage || 'Processing...'}</span>
                                        <span>{Math.round(progress)}%</span>
                                    </div>
                                    <div className="progress-bar-container">
                                        <div
                                            className="progress-bar-fill"
                                            style={{ width: `${progress}%` }}
                                        />
                                    </div>
                                    <div className="processing-steps">
                                        {PROCESS_STEPS.map((step, i) => {
                                            let status = '';
                                            if (i < currentStep) status = 'done';
                                            else if (i === currentStep && processing) status = 'active';
                                            else if (i === currentStep && done) status = 'done';

                                            return (
                                                <div className={`proc-step ${status}`} key={i}>
                                                    {status === 'active' ? (
                                                        <div className="spinner" />
                                                    ) : (
                                                        <div className="proc-step-icon">
                                                            {status === 'done' ? '✓' : (i + 1)}
                                                        </div>
                                                    )}
                                                    <span>{step}{status === 'active' ? '…' : ''}</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}

                            {/* Results */}
                            {done && (
                                <div className="results-section">
                                    <h3>Analysis complete</h3>
                                    <p>
                                        {violationCount} safety violation{violationCount !== 1 ? 's' : ''} detected
                                        {results?.elapsed ? ` in ${results.elapsed}s` : ''} in {file.name}
                                    </p>
                                    <button
                                        className="btn btn-primary"
                                        onClick={() => setShowDialog(true)}
                                    >
                                        View Results
                                    </button>
                                    <button
                                        className="btn btn-outline"
                                        onClick={downloadResult}
                                        style={{ marginLeft: 12 }}
                                    >
                                        Download Video
                                    </button>
                                    <button
                                        className="btn btn-outline"
                                        onClick={resetUpload}
                                        style={{ marginLeft: 12 }}
                                    >
                                        Upload Another
                                    </button>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>

            {/* Violations Dialog */}
            {showDialog && (
                <ViolationsDialog
                    filename={file?.name || 'video.mp4'}
                    violations={results?.violations || []}
                    elapsed={results?.elapsed || 0}
                    onClose={() => setShowDialog(false)}
                />
            )}
        </div>
    );
}
