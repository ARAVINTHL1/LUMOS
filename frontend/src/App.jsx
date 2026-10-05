import React, { useState } from 'react';
import {
  UploadCloud,
  Activity,
  Sparkles,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  RotateCcw,
  Layers
} from 'lucide-react';

export default function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeView, setActiveView] = useState('heatmap'); // 'heatmap' | 'original'

  // Pre-loaded test cases for quick clinical evaluation
  const testCases = [
    {
      id: 1,
      label: 'Severe Osteoporosis',
      tag: 'Osteoporosis',
      meta: 'Female, 68y • DXA BMD: 0.741 g/cm² (T-score: -3.11)',
      apUrl: '/static/images/patient_001/AP.png',
    },
    {
      id: 13,
      label: 'Osteopenia',
      tag: 'Osteopenia',
      meta: 'Female, 69y • DXA BMD: 0.983 g/cm² (T-score: -1.09)',
      apUrl: '/static/images/patient_013/AP.png',
    },
    {
      id: 22,
      label: 'Normal Bone Density',
      tag: 'Normal',
      meta: 'Male, 72y • DXA BMD: 1.283 g/cm² (T-score: +1.41)',
      apUrl: '/static/images/patient_022/AP.png',
    },
  ];

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  };

  const handleSelectTestCase = async (tc) => {
    setLoading(true);
    setImagePreview(tc.apUrl);
    setImageFile(null);

    try {
      const res = await fetch(`/api/predict/sample/${tc.id}`, { method: 'POST' });
      const data = await res.json();
      setResult({
        predicted_bmd_overall: data.predicted_bmd_overall,
        predicted_class: data.predicted_class,
        class_probabilities: data.class_probabilities,
        estimated_tscore: data.estimated_tscore,
        predicted_bmd_combinations: data.predicted_bmd_combinations,
        gradcam_url: data.gradcam_url,
        ap_preview: tc.apUrl,
        rois: data.rois?.AP || [],
        ground_truth: data.ground_truth,
        isSample: true,
      });
    } catch (err) {
      console.error('Error analyzing sample:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRunEvaluation = async () => {
    if (!imageFile && !imagePreview) {
      alert('Please upload an X-ray image or pick a test sample.');
      return;
    }

    setLoading(true);
    const formData = new FormData();
    if (imageFile) formData.append('ap_file', imageFile);

    try {
      const res = await fetch('/api/predict/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setResult({
        predicted_bmd_overall: data.predicted_bmd_overall,
        predicted_class: data.predicted_class,
        class_probabilities: data.class_probabilities,
        estimated_tscore: data.estimated_tscore,
        predicted_bmd_combinations: data.predicted_bmd_combinations,
        gradcam_url: data.gradcam_base64,
        ap_preview: data.ap_preview_base64 || imagePreview,
        rois_dict: data.rois_base64,
        isSample: false,
      });
    } catch (err) {
      console.error('Error evaluating uploaded image:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
  };

  // Helper for T-Score position (-4.0 to +2.0 normalized to 0-100%)
  const getMeterPercent = (tscore) => {
    const val = Math.max(-4.0, Math.min(2.0, tscore || 0));
    return ((val - -4.0) / (2.0 - -4.0)) * 100;
  };

  return (
    <div className="eval-container">
      {/* ── Minimalist Clean Header ────────────────────────────────────── */}
      <header className="eval-header">
        <div className="eval-header-inner">
          <div className="eval-brand">
            <div className="eval-logo-badge">
              <Activity size={22} strokeWidth={2.4} />
            </div>
            <div>
              <h1 className="eval-title">LUMOS</h1>
              <p className="eval-subtitle">
                Anatomy-Guided Lumbar BMD Estimation & Osteoporosis Screening
              </p>
            </div>
          </div>

          <div className="eval-tag-pill">
            <span className="dot-live"></span>
            Trained Model Active (MobileNetV2 + Transformer)
          </div>
        </div>
      </header>

      {/* ── Main Evaluation Studio ─────────────────────────────────────── */}
      <main className="eval-main">
        {!result ? (
          /* ============================================================== */
          /* STAGE 1: SINGLE IMAGE UPLOAD INTERFACE                         */
          /* ============================================================== */
          <div className="upload-workflow-card animate-fade-in">
            <div className="workflow-title-block" style={{ textAlign: 'center' }}>
              <h2>Upload Lumbar Spine Radiograph</h2>
              <p style={{ margin: '8px auto 0' }}>
                Upload an X-ray to estimate continuous bone mineral density (BMD),
                generate Grad-CAM++ explainability heatmaps, and classify osteoporosis severity.
              </p>
            </div>

            {/* Single Centered Upload Box */}
            <div className="single-upload-container">
              <div className={`upload-panel ${imagePreview ? 'has-file' : ''}`}>
                <div className="panel-header-label">
                  <span className="badge-required">X-RAY IMAGE</span>
                  <span>Lumbar Spine Radiograph</span>
                </div>

                {imagePreview ? (
                  <div className="preview-container" style={{ height: '260px' }}>
                    <img src={imagePreview} alt="Radiograph Preview" className="preview-img" />
                    <button
                      className="btn-change-file"
                      onClick={() => {
                        setImageFile(null);
                        setImagePreview(null);
                      }}
                    >
                      Change Image
                    </button>
                  </div>
                ) : (
                  <label className="drop-target" style={{ padding: '52px 24px' }}>
                    <input
                      type="file"
                      accept="image/*"
                      style={{ display: 'none' }}
                      onChange={handleFileChange}
                    />
                    <UploadCloud size={46} className="drop-icon" />
                    <div className="drop-title" style={{ fontSize: '15px' }}>
                      Click to upload X-ray or drag & drop
                    </div>
                    <div className="drop-sub" style={{ fontSize: '12.5px', marginTop: '6px' }}>
                      Supports PNG, JPG, JPEG, or DICOM exports
                    </div>
                  </label>
                )}
              </div>
            </div>

            {/* Quick Test Samples Bar */}
            <div className="quick-samples-bar" style={{ maxWidth: '620px', margin: '0 auto 28px' }}>
              <span className="samples-hint">Or test with a sample:</span>
              <div className="samples-buttons">
                {testCases.map((tc) => (
                  <button
                    key={tc.id}
                    className="btn-sample-pick"
                    onClick={() => handleSelectTestCase(tc)}
                  >
                    <span className={`status-dot-sm ${tc.tag}`}></span>
                    <strong>{tc.tag}</strong>
                  </button>
                ))}
              </div>
            </div>

            {/* Primary Action Button */}
            <div className="action-row">
              <button
                className="btn-analyze"
                onClick={handleRunEvaluation}
                disabled={loading || (!imageFile && !imagePreview)}
              >
                {loading ? (
                  <>
                    <div className="spinner-sm"></div>
                    Extracting Vertebrae & Running AI Evaluation...
                  </>
                ) : (
                  <>
                    <Sparkles size={18} />
                    Run AI BMD Estimation & Osteoporosis Screening
                  </>
                )}
              </button>
            </div>
          </div>
        ) : (
          /* ============================================================== */
          /* STAGE 2: DIAGNOSTIC & EVALUATION RESULTS                       */
          /* ============================================================== */
          <div className="results-container animate-fade-in">
            {/* Top Bar with Reset Button */}
            <div className="results-top-nav">
              <div>
                <h2 className="results-main-title">Radiological AI Diagnostic Report</h2>
                <p className="results-main-sub">
                  Anatomy-guided multi-target prediction using MobileNetV2 feature extractor and 4-layer inter-vertebral Transformer encoder.
                </p>
              </div>
              <button className="btn-reset" onClick={handleReset}>
                <RotateCcw size={15} />
                Evaluate Another Image
              </button>
            </div>

            {/* Diagnostic Hero Grid */}
            <div className="diagnostic-summary-grid">
              {/* Card 1: Osteoporosis Classification */}
              <div className="diag-card">
                <div className="diag-card-label">Diagnostic Classification</div>
                <div className="diag-status-wrap">
                  <div className={`status-hero-pill ${result.predicted_class}`}>
                    {result.predicted_class === 'Normal' && <ShieldCheck size={22} />}
                    {result.predicted_class === 'Osteopenia' && <AlertTriangle size={22} />}
                    {result.predicted_class === 'Osteoporosis' && <ShieldAlert size={22} />}
                    <span>{result.predicted_class.toUpperCase()}</span>
                  </div>
                </div>

                {/* Class Probabilities Bars */}
                <div className="probabilities-box">
                  {Object.entries(result.class_probabilities || {}).map(([cName, prob]) => (
                    <div key={cName} className="prob-item">
                      <span className="prob-name">{cName}</span>
                      <div className="prob-track">
                        <div
                          className={`prob-bar-fill ${cName}`}
                          style={{ width: `${prob * 100}%` }}
                        />
                      </div>
                      <span className="prob-pct">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Card 2: Estimated BMD Value */}
              <div className="diag-card">
                <div className="diag-card-label">Estimated Bone Mineral Density (L1–L4)</div>
                <div className="bmd-hero-value">
                  {result.predicted_bmd_overall.toFixed(3)}
                  <span className="bmd-unit">g/cm²</span>
                </div>

                {result.ground_truth?.ground_truth_bmd && (
                  <div className="dxa-gt-note">
                    <span>DXA Ground Truth:</span>
                    <strong>{result.ground_truth.ground_truth_bmd.toFixed(3)} g/cm²</strong>
                    <span className="delta-tag">
                      (Δ {(result.predicted_bmd_overall - result.ground_truth.ground_truth_bmd).toFixed(3)})
                    </span>
                  </div>
                )}

                <div className="bmd-interpretation">
                  {result.predicted_class === 'Normal'
                    ? 'Healthy trabecular bone structure and bone density within normal adult reference range.'
                    : result.predicted_class === 'Osteopenia'
                    ? 'Mild to moderate bone mineral loss detected. Lifestyle and dietary intervention recommended.'
                    : 'Significant bone mineral demineralization and architectural deterioration. High fracture vulnerability.'}
                </div>
              </div>

              {/* Card 3: WHO T-Score Meter */}
              <div className="diag-card">
                <div className="diag-card-label">Estimated WHO T-Score</div>
                <div className={`tscore-hero-value ${result.predicted_class}`}>
                  {result.estimated_tscore > 0 ? `+${result.estimated_tscore}` : result.estimated_tscore}
                </div>

                <div className="t-meter-wrap">
                  <div className="t-spectrum-bar">
                    <div
                      className="t-indicator-pin"
                      style={{ left: `${getMeterPercent(result.estimated_tscore)}%` }}
                    />
                  </div>
                  <div className="t-labels-row">
                    <span>&lt; -2.5 (Osteoporosis)</span>
                    <span>-1.0 (Osteopenia)</span>
                    <span>&gt; 0 (Normal)</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Explainable AI Visualizer (Grad-CAM vs Original) */}
            <div className="xai-section-card">
              <div className="xai-header">
                <div>
                  <div className="xai-section-title">
                    <Sparkles size={18} color="var(--brand-primary)" />
                    Explainable AI (Grad-CAM++ Attention Heatmap)
                  </div>
                  <div className="xai-section-sub">
                    Visualizes the neural network's activation intensity. Warmer red/yellow colors indicate vertebral bone regions that drove the model's BMD estimate.
                  </div>
                </div>

                {/* View switcher */}
                <div className="view-toggle-pills">
                  <button
                    className={`toggle-btn ${activeView === 'heatmap' ? 'active' : ''}`}
                    onClick={() => setActiveView('heatmap')}
                  >
                    Grad-CAM Overlay
                  </button>
                  <button
                    className={`toggle-btn ${activeView === 'original' ? 'active' : ''}`}
                    onClick={() => setActiveView('original')}
                  >
                    Original Radiograph
                  </button>
                </div>
              </div>

              <div className="xai-image-viewport">
                {activeView === 'heatmap' && result.gradcam_url ? (
                  <img
                    src={result.gradcam_url}
                    alt="Grad-CAM Explainability Heatmap"
                    className="xai-full-img"
                  />
                ) : (
                  <img
                    src={result.ap_preview}
                    alt="Original Radiograph"
                    className="xai-full-img"
                  />
                )}
              </div>
            </div>

            {/* Vertebral Sub-region Breakdown & Crops */}
            <div className="vertebrae-breakdown-card">
              <div className="breakdown-title">
                <Layers size={18} color="var(--brand-primary)" />
                Anatomical Vertebral Levels & Multi-target Regression
              </div>

              {/* Individual Vertebra Crops */}
              <div className="crops-row">
                {['L1', 'L2', 'L3', 'L4'].map((vName, idx) => (
                  <div key={vName} className="crop-box">
                    <div className="crop-img-wrap">
                      {result.isSample ? (
                        <img
                          src={`/static/rois/patient_${String(result.ground_truth?.patient_id || 1).padStart(3, '0')}/AP/${vName}.png`}
                          alt={`${vName} crop`}
                        />
                      ) : result.rois_dict?.[vName] ? (
                        <img src={result.rois_dict[vName]} alt={`${vName} crop`} />
                      ) : (
                        <div className="crop-placeholder">{vName}</div>
                      )}
                    </div>
                    <div className="crop-label">{vName} Vertebra</div>
                    <div className="crop-pos">Anatomical Token #{idx + 1}</div>
                  </div>
                ))}
              </div>

              {/* Sub-region Combinations Table */}
              {result.predicted_bmd_combinations && (
                <div className="combos-grid">
                  {Object.entries(result.predicted_bmd_combinations)
                    .filter(([name]) => name !== 'Overall')
                    .map(([name, val]) => (
                      <div key={name} className="combo-stat-chip">
                        <span className="combo-name">{name}</span>
                        <span className="combo-val">{val.toFixed(3)} g/cm²</span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
