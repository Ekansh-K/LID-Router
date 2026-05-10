import React, { useEffect, useRef } from 'react';
import gsap from 'gsap';
import {
  AudioLines,
  Wrench,
  Waves,
  GitMerge,
  Split,
  Cpu,
  CheckCircle2,
  ListTree,
  ArrowRightLeft,
  ShieldAlert,
  Play
} from 'lucide-react';

const mainSteps = [
  { id: 'input', label: 'Input', icon: AudioLines, color: 'var(--node-input)', anim: 'anim-pulse', desc: 'Raw 16kHz audio input' },
  { id: 'preprocess', label: 'Prep & VAD', icon: Wrench, color: 'var(--node-prep)', anim: 'anim-spin', desc: 'Voice Activity Detection & padding/trimming' },
  { id: 'lid', label: 'Dual LID', icon: Waves, color: 'var(--node-lid)', anim: 'anim-bounce', desc: 'Parallel language detection via Acoustic (MMS-LID) and Decoder (Whisper)' },
  { id: 'fusion', label: 'Fusion', icon: GitMerge, color: 'var(--node-fusion)', anim: 'anim-pulse', desc: 'Weighted probability fusion of dual signals' },
  { id: 'routing', label: 'Router', icon: Split, color: 'var(--node-route)', anim: 'anim-bounce', desc: 'Agentic routing decision based on fused uncertainty' }
];

export default function PipelineVisualizer({ data, expectedText, replayKey }) {
  const containerRef = useRef(null);
  const mainNodesRef = useRef([]);
  const mainLinesRef = useRef([]);
  const outputNodeRef = useRef(null);

  const branchRefs = { A: useRef(null), B: useRef(null), C: useRef(null) };
  const inPathRefs = { A: useRef(null), B: useRef(null), C: useRef(null) };
  const outPathRefs = { A: useRef(null), B: useRef(null), C: useRef(null) };

  const outputPanelRef = useRef(null);

  useEffect(() => {
    // ALWAYS reset animations to base state before running

    // 1. Reset main nodes safely
    mainNodesRef.current.forEach(n => {
      if (n) {
        n.classList.remove('active', 'completed');
        const box = n.querySelector('.node-icon-box');
        if (box) {
          box.style.backgroundColor = 'var(--bg-color)';
          box.style.borderColor = 'var(--border-color)';
        }
      }
    });

    // 2. Reset main lines (NO clearProps: 'all' to preserve React inline styles)
    gsap.set(mainLinesRef.current, { width: '0%', boxShadow: 'none' });

    // 3. Reset branch boxes
    gsap.set([branchRefs.A.current, branchRefs.B.current, branchRefs.C.current], { opacity: 0.3, filter: 'grayscale(100%)', scale: 0.95 });
    Object.keys(branchRefs).forEach(key => {
      if (branchRefs[key].current) branchRefs[key].current.classList.remove('branch-active');
    });

    // 4. Reset branch SVG paths
    const allPaths = [
      inPathRefs.A.current, inPathRefs.B.current, inPathRefs.C.current,
      outPathRefs.A.current, outPathRefs.B.current, outPathRefs.C.current
    ];
    gsap.set(allPaths, { stroke: "var(--border-color)", filter: "none", strokeDasharray: "none", strokeDashoffset: 0 });

    // 5. Reset output node and panel
    if (outputNodeRef.current) {
      outputNodeRef.current.classList.remove('active', 'completed');
      const box = outputNodeRef.current.querySelector('.node-icon-box');
      if (box) {
        box.style.backgroundColor = 'var(--bg-color)';
        box.style.borderColor = 'var(--border-color)';
      }
    }
    if (outputPanelRef.current) gsap.set(outputPanelRef.current, { opacity: 0, y: 20 });

    if (!data) {
      return;
    }

    const tl = gsap.timeline();

    // 1. Animate Main Linear Pipeline
    mainSteps.forEach((step, index) => {
      tl.add(() => {
        if (mainNodesRef.current[index]) {
          mainNodesRef.current[index].classList.add('active');
          mainNodesRef.current[index].querySelector('.node-icon-box').style.backgroundColor = step.color;
          mainNodesRef.current[index].querySelector('.node-icon-box').style.borderColor = step.color;

          if (index > 0 && mainNodesRef.current[index - 1]) {
            mainNodesRef.current[index - 1].classList.remove('active');
            mainNodesRef.current[index - 1].classList.add('completed');
          }
        }
      }, `+=${0.4}`);

      // Animate connecting line
      if (index < mainSteps.length - 1) {
        tl.to(mainLinesRef.current[index], {
          width: '100%',
          boxShadow: `0 0 10px ${step.color}`,
          duration: 0.3
        });
      }
    });

    // 2. Animate Branching Logic
    tl.add(() => {
      if (mainNodesRef.current[mainSteps.length - 1]) {
        mainNodesRef.current[mainSteps.length - 1].classList.remove('active');
        mainNodesRef.current[mainSteps.length - 1].classList.add('completed');
      }

      let modeStr = String(data.routing_mode).toUpperCase();
      let activeBranch = 'A';
      let branchColor = 'var(--node-decode)';

      if (modeStr.includes('MULTI') || modeStr === 'B') {
        activeBranch = 'B';
        branchColor = '#9c27b0';
      } else if (modeStr.includes('FALLBACK') || modeStr === 'C') {
        activeBranch = 'C';
        branchColor = '#ff9800';
      }

      // Highlight the active branch and its SVG paths
      Object.keys(branchRefs).forEach(key => {
        if (key === activeBranch) {
          gsap.to(branchRefs[key].current, { opacity: 1, filter: 'grayscale(0%)', scale: 1.05, duration: 0.5, ease: "back.out(1.5)" });
          branchRefs[key].current.classList.add('branch-active');

          // Simple color fade for the branch connections (as requested)
          gsap.to([inPathRefs[key].current, outPathRefs[key].current], {
            stroke: branchColor,
            filter: `drop-shadow(0 0 5px ${branchColor})`,
            duration: 0.5
          });
        } else {
          gsap.to(branchRefs[key].current, { opacity: 0.2, filter: 'grayscale(100%)', scale: 0.9, duration: 0.5 });
          branchRefs[key].current.classList.remove('branch-active');

          gsap.to([inPathRefs[key].current, outPathRefs[key].current], {
            stroke: "var(--border-color)",
            filter: "none",
            duration: 0.5
          });
        }
      });
    }, "+=0.2");

    // 3. Animate Output Node
    tl.add(() => {
      if (outputNodeRef.current) {
        outputNodeRef.current.classList.add('active');
        outputNodeRef.current.querySelector('.node-icon-box').style.backgroundColor = 'var(--node-output)';
        outputNodeRef.current.querySelector('.node-icon-box').style.borderColor = 'var(--node-output)';
      }
    }, "+=0.6");

    // 4. Show Results Panel
    if (outputPanelRef.current) {
      tl.to(outputPanelRef.current, {
        opacity: 1,
        y: 0,
        duration: 0.8,
        ease: 'back.out(1.5)'
      }, "+=0.2");
    }

  }, [data, replayKey]);

  return (
    <div ref={containerRef} className="w-full no-scrollbar" style={{ padding: '2rem', display: 'flex', flexDirection: 'column' }}>

      {/* 
        Using justify-content: flex-start and a minimum width to prevent 
        the left side (Input node) from getting cut off when scrolling 
      */}
      <div className="pipeline-graph" style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'flex-start', flexWrap: 'nowrap', minWidth: 'min-content', margin: '0 auto' }}>

        {/* Main Linear Steps */}
        <div style={{ display: 'flex', alignItems: 'flex-start' }}>
          {mainSteps.map((step, index) => {
            const Icon = step.icon;
            return (
              <React.Fragment key={step.id}>
                {/* Node */}
                <div
                  className="node-wrapper tooltip-wrapper"
                  ref={el => mainNodesRef.current[index] = el}
                  style={{ width: '90px' }}
                >
                  <div className="custom-tooltip">
                    {step.desc}
                    {data && step.id === 'lid' && data.lid_distribution ? <><br /><span style={{ opacity: 0.8, fontSize: '0.7rem' }}>Top: {Object.keys(data.lid_distribution).slice(0, 3).join(', ')}</span></> : null}
                  </div>
                  <div className="node-icon-box shadow-md">
                    <Icon size={28} className={`node-icon ${step.anim}`} />
                  </div>
                  <div className="node-title">{step.label}</div>
                  <div className="node-details">
                    {data && step.id === 'lid' && data.lid_distribution && Object.keys(data.lid_distribution)[0]}
                  </div>
                </div>

                {/* Centered Line connecting node-icon-box to the next */}
                {index < mainSteps.length - 1 && (
                  <div style={{ display: 'flex', alignItems: 'center', position: 'relative', width: '40px', height: '60px' /* exactly aligns with 60px icon box */ }}>
                    {/* Background grey line */}
                    <div style={{ width: '100%', height: '4px', background: 'var(--border-color)', borderRadius: '2px' }} />
                    {/* Mask container that animates width */}
                    <div
                      ref={el => mainLinesRef.current[index] = el}
                      style={{
                        position: 'absolute', left: 0, width: '0%',
                        height: '4px', overflow: 'hidden', borderRadius: '2px'
                      }}
                    >
                      {/* Fixed-width gradient line inside */}
                      <div style={{
                        width: '40px', height: '4px',
                        background: `linear-gradient(90deg, ${step.color}, ${mainSteps[index + 1].color})`
                      }} />
                    </div>
                  </div>
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Branching SVG Connector IN */}
        <div style={{ position: 'relative', width: '80px', height: '520px' }}>
          <svg width="100%" height="100%" style={{ position: 'absolute', top: 0, left: 0 }}>
            {/* Mode A path (top) */}
            <path ref={inPathRefs.A} d="M 0 30 C 40 30, 40 85, 80 85" fill="none" stroke="var(--border-color)" strokeWidth="4" />
            {/* Mode B path (middle) */}
            <path ref={inPathRefs.B} d="M 0 30 C 40 30, 40 260, 80 260" fill="none" stroke="var(--border-color)" strokeWidth="4" />
            {/* Mode C path (bottom) */}
            <path ref={inPathRefs.C} d="M 0 30 C 40 30, 40 435, 80 435" fill="none" stroke="var(--border-color)" strokeWidth="4" />
          </svg>
        </div>

        {/* Branching Logic Modes */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', width: '380px', height: '520px', justifyContent: 'center' }}>

          {/* Mode A */}
          <div className="branch-box tooltip-wrapper" ref={branchRefs.A} style={{ ...branchStyle, borderColor: 'var(--node-decode)' }}>
            <div className="custom-tooltip" style={{ top: '-30px' }}>Routes to Whisper for accurate decoding</div>
            <h4 style={{ color: 'var(--node-decode)', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <Cpu size={18} /> Mode A (Single)
            </h4>
            <div className="internal-nodes" style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: 'var(--bg-color)', padding: '0.8rem', borderRadius: '8px' }}>
              <div className="small-node">Whisper<br />Decode</div>
              <ArrowRightLeft size={16} color="var(--text-secondary)" />
              <div className="small-node">Done</div>
            </div>
          </div>

          {/* Mode B */}
          <div className="branch-box tooltip-wrapper" ref={branchRefs.B} style={{ ...branchStyle, borderColor: '#9c27b0' }}>
            <div className="custom-tooltip" style={{ top: '-30px' }}>Compares MMS-ASR and Whisper ASR </div>
            <h4 style={{ color: '#9c27b0', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <ListTree size={18} /> Mode B (Multi-Hypothesis)
            </h4>
            <div className="internal-nodes" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', background: 'var(--bg-color)', padding: '0.8rem', borderRadius: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div className="small-node">Whisper Decode</div>
                <div className="small-node">MMS-ASR Decode</div>
              </div>
              <div style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>↓</div>
              <div className="small-node" style={{ width: '100%', background: 'rgba(156, 39, 176, 0.2)' }}>Compare & Select Best</div>
            </div>
          </div>

          {/* Mode C */}
          <div className="branch-box tooltip-wrapper" ref={branchRefs.C} style={{ ...branchStyle, borderColor: '#ff9800' }}>
            <div className="custom-tooltip" style={{ top: '-30px' }}>Attempts MMS-ASR, falls back to Whisper</div>
            <h4 style={{ color: '#ff9800', display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <ShieldAlert size={18} /> Mode C (Fallback)
            </h4>
            <div className="internal-nodes" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg-color)', padding: '0.8rem', borderRadius: '8px' }}>
              <div className="small-node">MMS-ASR Decode</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>Fails? →</div>
              <div className="small-node" style={{ borderStyle: 'dashed' }}>Whisper</div>
            </div>
          </div>

        </div>

        {/* Branching SVG Connector OUT */}
        <div style={{ position: 'relative', width: '80px', height: '520px' }}>
          <svg width="100%" height="100%" style={{ position: 'absolute', top: 0, left: 0 }}>
            <path ref={outPathRefs.A} d="M 0 85 C 40 85, 40 30, 80 30" fill="none" stroke="var(--border-color)" strokeWidth="4" />
            <path ref={outPathRefs.B} d="M 0 260 C 40 260, 40 30, 80 30" fill="none" stroke="var(--border-color)" strokeWidth="4" />
            <path ref={outPathRefs.C} d="M 0 435 C 40 435, 40 30, 80 30" fill="none" stroke="var(--border-color)" strokeWidth="4" />
          </svg>
        </div>

        {/* Final Output Node */}
        <div className="node-wrapper tooltip-wrapper" ref={outputNodeRef} style={{ width: '90px' }}>
          <div className="custom-tooltip">
            {data ? `Final transcript via ${data.backend_used}` : 'Output Destination'}
          </div>
          <div className="node-icon-box shadow-md">
            <CheckCircle2 size={28} className="node-icon anim-pulse" />
          </div>
          <div className="node-title">Output</div>
          <div className="node-details">{data && data.backend_used}</div>
        </div>

      </div>

      {/* Output Panel */}
      {data && (
        <div className="output-panel w-full" ref={outputPanelRef} style={{ maxWidth: '1000px', margin: '3rem auto 0', position: 'relative' }}>

          <div className="stat-grid" style={{ marginTop: '1rem' }}>
            <div className="stat-box">
              <div className="stat-label">Language</div>
              <div className="stat-value">{data.detected_language_name || data.detected_language}</div>
            </div>
            <div className="stat-box">
              <div className="stat-label">Confidence</div>
              <div className="stat-value">{(data.confidence * 100).toFixed(1)}%</div>
            </div>
            <div className="stat-box">
              <div className="stat-label">Routing</div>
              <div className="stat-value">Mode {data.routing_mode}</div>
            </div>
            <div className="stat-box">
              <div className="stat-label">Backend</div>
              <div className="stat-value" style={{ textTransform: 'capitalize' }}>
                {data.backend_used === 'mms' ? 'MMS-ASR' : data.backend_used}
              </div>
            </div>
          </div>

          <div className="transcript-container">
            {expectedText && (
              <>
                <div className="transcript-box reference-box">
                  <div className="stat-label" style={{ color: 'var(--node-prep)', marginBottom: '0.5rem' }}>Expected Ground Truth (Reference)</div>
                  <div style={{ fontWeight: 'bold', fontSize: '1.2rem' }}>{expectedText.reference}</div>
                </div>

                {expectedText.stripped && expectedText.stripped !== expectedText.reference && (
                  <div className="transcript-box reference-box" style={{ opacity: 0.8, fontSize: '1rem' }}>
                    <div className="stat-label" style={{ marginBottom: '0.3rem' }}>Stripped Version</div>
                    <div>{expectedText.stripped}</div>
                  </div>
                )}
              </>
            )}

            <div className="transcript-box">
              <div className="stat-label" style={{ color: 'var(--node-decode)', marginBottom: '0.5rem' }}>Predicted Transcript</div>
              <div style={{ fontSize: '1.2rem' }}>{data.transcript}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const branchStyle = {
  padding: '1rem',
  background: 'var(--bg-panel)',
  border: '2px solid',
  borderRadius: '12px',
  opacity: 0.3,
  filter: 'grayscale(100%)',
  transform: 'scale(0.95)'
};
