import { useEffect } from 'react';
import './ViolationsDialog.css';

export default function ViolationsDialog({ filename, violations = [], elapsed = 0, onClose }) {
    const highCount = violations.filter((v) => v.severity === 'high').length;
    const medCount = violations.filter((v) => v.severity === 'medium').length;
    const lowCount = violations.filter((v) => v.severity === 'low').length;

    useEffect(() => {
        const onKey = (e) => e.key === 'Escape' && onClose();
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [onClose]);

    const exportToCSV = () => {
        if (violations.length === 0) return;

        const headers = ['Timestamp', 'Violation', 'Severity', 'Confidence (%)'];
        const rows = violations.map(v => [
            v.time,
            v.type,
            v.severity.charAt(0).toUpperCase() + v.severity.slice(1),
            v.confidence
        ]);

        const csvContent = [
            headers.join(','),
            ...rows.map(r => r.map(cell => `"${cell}"`).join(','))
        ].join('\n');

        // Prepend BOM (\ufeff) so Excel opens UTF-8 correctly
        const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', `safesight_report_${filename.split('.')[0]}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="dialog-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
            <div className="dialog" role="dialog" aria-modal="true">
                {/* Header */}
                <div className="dialog-header">
                    <div className="dialog-header-info">
                        <h2>Violation Report</h2>
                        <div className="dialog-meta">
                            <span>📄 {filename}</span>
                            <span>⏱ Processing time: {elapsed}s</span>
                            <span>🔍 {violations.length} violations found</span>
                        </div>
                    </div>
                    <button className="dialog-close" onClick={onClose} aria-label="Close">
                        ✕
                    </button>
                </div>

                {/* Summary Stats */}
                <div className="dialog-stats">
                    <span className="stat-chip high">
                        <span className="stat-dot" />
                        {highCount} High
                    </span>
                    <span className="stat-chip medium">
                        <span className="stat-dot" />
                        {medCount} Medium
                    </span>
                    <span className="stat-chip low">
                        <span className="stat-dot" />
                        {lowCount} Low
                    </span>
                </div>

                {/* Table */}
                <div className="dialog-body">
                    {violations.length === 0 ? (
                        <div style={{
                            textAlign: 'center',
                            padding: '48px 24px',
                            color: 'rgba(255,255,255,0.5)',
                        }}>
                            <p style={{ fontSize: '1.1rem' }}>✅ No violations detected</p>
                            <p style={{ fontSize: '0.85rem', marginTop: 8 }}>
                                All workers appear to be wearing helmets in this footage.
                            </p>
                        </div>
                    ) : (
                        <table className="violations-table">
                            <thead>
                                <tr>
                                    <th>Timestamp</th>
                                    <th>Violation</th>
                                    <th>Severity</th>
                                    <th>Confidence</th>
                                </tr>
                            </thead>
                            <tbody>
                                {violations.map((v, i) => (
                                    <tr key={i}>
                                        <td>
                                            <span style={{ fontFamily: 'var(--font-heading)', fontSize: '0.78rem' }}>
                                                {v.time}
                                            </span>
                                        </td>
                                        <td>{v.type}</td>
                                        <td>
                                            <span className={`severity-badge ${v.severity}`}>
                                                ● {v.severity.charAt(0).toUpperCase() + v.severity.slice(1)}
                                            </span>
                                        </td>
                                        <td>
                                            <span className="confidence">{v.confidence}%</span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    )}
                </div>

                {/* Footer */}
                <div className="dialog-footer">
                    <button className="btn btn-outline" onClick={onClose}>
                        Close
                    </button>
                    <button 
                        className="btn btn-primary" 
                        onClick={exportToCSV}
                        disabled={violations.length === 0}
                    >
                        Export Report
                    </button>
                </div>
            </div>
        </div>
    );
}
