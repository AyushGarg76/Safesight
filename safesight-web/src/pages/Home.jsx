import { Link } from 'react-router-dom';
import './Home.css';

const features = [
    {
        icon: '⊡',
        title: 'Faster R-CNN Detection',
        desc: 'A ResNet-50 FPN backbone trained specifically to identify persons, helmets, and bare heads without relying on YOLO.',
    },
    {
        icon: '◫',
        title: 'Spatial Reasoning',
        desc: 'Interprets bounding boxes using deterministic IoU-based logic to confidently infer whether a worker\'s head is protected.',
    },
    {
        icon: '◈',
        title: 'Adaptive Enhancement',
        desc: 'Automatic frame correction — CLAHE, gamma adjustment, sharpening — triggered by real-time scene analysis.',
    },
];

const steps = [
    {
        title: 'Upload & Job Creation',
        desc: 'Video is uploaded through the frontend, initiating a background processing daemon on the Flask API.',
    },
    {
        title: 'Frame Sampling',
        desc: 'To ensure efficiency, video frames are selectively sampled to reduce inference calls without losing context.',
    },
    {
        title: 'Batch Inference',
        desc: 'Sampled frames are processed by a custom Faster R-CNN model to detect persons, helmets, and bare heads.',
    },
    {
        title: 'Helmet Reasoning Engine',
        desc: 'Uses direct head detection and person-helmet IoU cross-checks to identify helmet compliance violations.',
    },
    {
        title: 'Annotation & Aggregation',
        desc: 'Violations are visually annotated on the video and grouped into concise, timestamped summary reports.',
    },
];

const modelDetails = [
    {
        title: 'ResNet-50 Backbone',
        desc: 'Extracts robust spatial features using 50 layers of residual blocks. Initialized with ImageNet weights to leverage generalized feature representations.',
    },
    {
        title: 'Feature Pyramid Network (FPN)',
        desc: 'Solves the multi-scale detection problem by merging features top-down, crucial for detecting small helmets at various camera depths.',
    },
    {
        title: 'Custom Detection Head',
        desc: 'The FastRCNNPredictor is fine-tuned from scratch to specifically classify three distinct objects: persons, helmets, and bare heads.',
    },
];

const techStack = [
    'Python',
    'PyTorch',
    'Faster R-CNN',
    'Flask',
    'React',
    'OpenCV',
];

export default function Home() {
    return (
        <>
            {/* ── Hero ── */}
            <section className="hero section" id="hero">
                <div className="container">
                    <div className="hero-content">
                        <div className="hero-label">
                            <span className="dot" />
                            Computer Vision Project — UCS532
                        </div>
                        <h1>
                            Intelligent safety
                            <br />
                            monitoring for
                            <br />
                            <span className="highlight">industrial environments.</span>
                        </h1>
                        <p className="hero-desc">
                            SafeSight transforms standard CCTV footage into actionable safety
                            intelligence — utilizing Faster R-CNN and spatial reasoning to detect
                            helmet compliance and flag violations automatically.
                        </p>
                        <div className="hero-actions">
                            <Link to="/upload" className="btn btn-primary">
                                Try It Out →
                            </Link>
                            <a href="#how-it-works" className="btn btn-outline">
                                How It Works
                            </a>
                        </div>
                    </div>
                </div>
            </section>

            {/* ── Features ── */}
            <section className="features section" id="features">
                <div className="container">
                    <p className="section-label">Capabilities</p>
                    <h2 className="section-title">What Safesight does</h2>
                    <p className="section-subtitle">
                        Three core modules working together to turn raw camera feeds into
                        safety insights.
                    </p>
                    <div className="features-grid">
                        {features.map((f, i) => (
                            <div className="card feature-card" key={i}>
                                <div className="feature-icon">{f.icon}</div>
                                <h3>{f.title}</h3>
                                <p>{f.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── How It Works ── */}
            <section className="section" id="how-it-works">
                <div className="container">
                    <p className="section-label">Pipeline</p>
                    <h2 className="section-title">How it works</h2>
                    <p className="section-subtitle">
                        A five-stage computer vision pipeline from raw footage to actionable
                        safety reports.
                    </p>
                    <div className="pipeline-steps">
                        {steps.map((s, i) => (
                            <div className="pipeline-step" key={i}>
                                <span className="step-number">0{i + 1}</span>
                                <div className="step-content">
                                    <h3>{s.title}</h3>
                                    <p>{s.desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── Model Architecture ── */}
            <section className="section" id="model-architecture">
                <div className="container">
                    <p className="section-label">Deep Learning</p>
                    <h2 className="section-title">Faster R-CNN ResNet-50 FPN</h2>
                    <p className="section-subtitle">
                        Under the hood, the system is powered by a state-of-the-art Faster R-CNN object detector tailored for workplace safety.
                    </p>
                    <div className="features-grid">
                        {modelDetails.map((m, i) => (
                            <div className="card feature-card" key={i}>
                                <h3>{m.title}</h3>
                                <p>{m.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── Tech Stack ── */}
            <section className="tech-stack section" id="tech-stack">
                <div className="container">
                    <p className="section-label">Built With</p>
                    <h2 className="section-title">Tech stack</h2>
                    <div className="tech-grid">
                        {techStack.map((t) => (
                            <span className="tech-badge" key={t}>
                                <span className="tech-dot" />
                                {t}
                            </span>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── Footer ── */}
            <footer className="footer">
                <div className="container">
                    <p>Safesight — UCS532 Computer Vision Project © 2026</p>
                </div>
            </footer>
        </>
    );
}
