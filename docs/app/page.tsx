const paperUrl = "https://arxiv.org/abs/2607.14935";
const codeUrl = "https://github.com/MCG-NJU/VideoChat3";
const modelUrl = "https://huggingface.co/collections/MCG-NJU/videochat3";
const huggingFaceLogoUrl = "https://huggingface.co/front/assets/huggingface_logo-noborder.svg";
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const asset = (path: string) => `${basePath}${path}`;

const Arrow = () => <span aria-hidden="true">↗</span>;
const ArxivMark = () => <span className="arxiv-mark" aria-hidden="true">arXiv</span>;
const GitHubIcon = () => <img className="brand-icon" src={asset("/brands/github.svg")} alt="" aria-hidden="true" />;
const HuggingFaceIcon = () => <img className="brand-icon brand-icon-huggingface" src={huggingFaceLogoUrl} alt="" aria-hidden="true" />;

export default function Home() {
  return (
    <main>
      <nav className="nav shell" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="VideoChat3 home">
          <img src={asset("/parrot.png")} alt="" />
          <span>VideoChat3</span>
        </a>
        <div className="nav-links">
          <a href="#architecture">Architecture</a>
          <a href="#data">Data</a>
          <a href="#results">Results</a>
          <a href="#demos">Demos</a>
        </div>
        <a className="button button-small button-dark" href={codeUrl} target="_blank" rel="noreferrer">
          <GitHubIcon /><span>View code</span><Arrow />
        </a>
      </nav>

      <section className="hero shell" id="top">
        <div className="hero-copy">
          <p className="eyebrow"><span>Research release</span> Generalist video MLLM</p>
          <h1>VideoChat3</h1>
          <p className="hero-lede">
            A generalist video MLLM built for fine-grained motion, long-form reasoning,
            temporal grounding, and live streaming.
          </p>
          <div className="hero-highlights" aria-label="VideoChat3 release highlights">
            <div><strong>Token efficient</strong><span>16× spatiotemporal compression</span></div>
            <div><strong>Fully open</strong><span>Models · code · data · training recipe</span></div>
          </div>
          <div className="hero-actions">
            <a className="button button-arxiv" href={paperUrl} target="_blank" rel="noreferrer"><ArxivMark /><span>Read the paper</span><Arrow /></a>
            <a className="button button-primary" href={codeUrl} target="_blank" rel="noreferrer"><GitHubIcon /><span>Explore the code</span><Arrow /></a>
            <a className="button button-secondary" href={modelUrl} target="_blank" rel="noreferrer"><HuggingFaceIcon /><span>Models &amp; data</span><Arrow /></a>
          </div>
        </div>

        <div className="hero-system" aria-label="VideoChat3 streaming state illustration">
          <div className="system-topline">
            <span>ADAPTIVE LIVE PERCEPTION</span><span>05:40 / 06:00</span>
          </div>
          <div className="resolution-labels" aria-hidden="true">
            <span>LOW RESOLUTION</span>
            <span className="resolution-boost">HIGH RESOLUTION</span>
            <span className="resolution-return">LOW RESOLUTION</span>
          </div>
          <div className="frame-window">
            <figure className="frame real-frame frame-low">
              <img src={asset("/hero/low-a.jpg")} alt="Low-resolution frame of routine cooking activity" />
              <figcaption><span>04:10</span><b>Silence</b></figcaption>
            </figure>
            <figure className="frame real-frame frame-low">
              <img src={asset("/hero/low-b.jpg")} alt="Low-resolution frame before new evidence appears" />
              <figcaption><span>05:00</span><b>Monitor</b></figcaption>
            </figure>
            <figure className="frame real-frame frame-high frame-standby">
              <img src={asset("/hero/standby.jpg")} alt="High-resolution frame selected after the model enters Standby" />
              <figcaption><span>05:30</span><b>Standby</b></figcaption>
            </figure>
            <figure className="frame real-frame frame-low frame-response">
              <img src={asset("/hero/response.jpg")} alt="Low-resolution frame used when the model responds" />
              <figcaption><span>05:40</span><b>Response</b></figcaption>
            </figure>
          </div>
          <div className="resolution-jump" aria-label="Standby switches to high resolution; Response returns to low resolution">
            <span>Low</span><i className="jump-up" /><strong>↑ High</strong><i className="jump-down" /><span>↓ Low</span>
          </div>
          <div className="states">
            <div><b>01</b><span>Silence</span><small>monitor efficiently</small></div>
            <div className="state-active"><b>02</b><span>Standby</span><small>look closer</small></div>
            <div><b>03</b><span>Response</span><small>speak with evidence</small></div>
          </div>
          <p className="system-caption">Standby alone switches the next window to high resolution; Response returns to low resolution.</p>
        </div>
      </section>

      <section className="proof-strip" aria-label="Key VideoChat3 results">
        <div className="shell proof-grid">
          <div><strong>4B</strong><span>parameters</span></div>
          <div><strong>16×</strong><span>spatiotemporal compression</span></div>
          <div><strong>3M</strong><span>curated instruction samples</span></div>
          <div><strong>20.4s</strong><span>latency at 2,048 frames</span></div>
        </div>
      </section>

      <section className="section shell thesis" id="research">
        <p className="kicker">THE PREMISE</p>
        <div className="section-heading split-heading">
          <h2>One model. Every tempo.</h2>
          <p>
            Most video models are optimized for one regime. VideoChat3 treats video as a
            continuous temporal signal—from a subtle hand movement, to an hour-long story,
            to the moment a live assistant should respond.
          </p>
        </div>
        <div className="tempo-grid">
          <article><span>01 / MOTION</span><h3>See what changes.</h3><p>Preserve fine-grained actions and short-lived transitions instead of relying on sparse still frames.</p></article>
          <article><span>02 / LONG FORM</span><h3>Remember what matters.</h3><p>Connect evidence across minutes or hours, retrieve events, and ground answers to precise moments.</p></article>
          <article><span>03 / STREAMING</span><h3>Know when to speak.</h3><p>Observe causally, stay silent without enough evidence, and respond when the answer becomes available.</p></article>
        </div>
        <figure className="paper-figure overview-figure">
          <img src={asset("/paper/overview.png")} alt="VideoChat3 benchmark overview and examples across motion, long-video QA, temporal grounding, and online response" />
          <figcaption><span>Figure 01</span> Generalist capability across offline and online video understanding.</figcaption>
        </figure>
      </section>

      <section className="section architecture" id="architecture">
        <div className="shell">
          <p className="kicker">ARCHITECTURE / EFFICIENCY</p>
          <div className="section-heading split-heading">
            <h2>Compress after <em>understanding.</em></h2>
            <p>
              VideoChat3 models local space and time inside the visual tokenizer—before visual
              tokens reach the language model. Redundancy is removed early, while motion cues stay intact.
            </p>
          </div>

          <div className="method-stack">
            <article className="method-card method-light">
              <div className="method-copy">
                <span className="method-number">01</span>
                <p className="mini-label">INFLATED 3D VISION TRANSFORMER</p>
                <h3>I3D‑ViT</h3>
                <p>Groups four neighboring frames, performs native-resolution spatiotemporal attention, then pools in time.</p>
                <ul>
                  <li>4× temporal compression</li>
                  <li>2×2 spatial pixel shuffle</li>
                  <li>16× combined compression</li>
                </ul>
              </div>
              <div className="method-visual method-i3d-visual"><img className="diagram-i3d" src={asset("/paper/i3d-vit.png")} alt="I3D-ViT architecture diagram" /></div>
            </article>

            <article className="method-card method-rust">
              <div className="method-copy">
                <span className="method-number">02</span>
                <p className="mini-label">ADAPTIVE FRAME RESOLUTION</p>
                <h3>Attention where it counts.</h3>
                <p>Streaming windows start at low resolution. A Standby state raises the next window’s pixel budget to inspect emerging evidence.</p>
                <div className="quota-row"><span>Silence · low resolution</span><i /><span>Standby · high resolution</span></div>
              </div>
              <div className="method-visual method-stream-visual"><img className="diagram-stream" src={asset("/paper/adaptive-streaming.png")} alt="Adaptive frame resolution and streaming state architecture" /></div>
            </article>
          </div>
        </div>
      </section>

      <section className="section shell data-section" id="data">
        <p className="kicker">DATA / EFFECTIVENESS</p>
        <div className="section-heading split-heading">
          <h2>Three datasets. One continuum.</h2>
          <p>
            The data recipe moves from reliable academic labels to long-range evidence and
            causal response timing. Every stage is designed to make supervision denser without
            losing its grounding in the video.
          </p>
        </div>
        <div className="data-grid">
          <article className="data-card">
            <div className="data-meta"><span>GENERAL</span><strong>2M</strong></div>
            <h3>Academic2M</h3>
            <p>Rewrites sparse labels into evidence-grounded answers, then filters for consistency with the original annotation.</p>
            <div className="data-line"><i /><i /><i /><i /></div>
          </article>
          <article className="data-card data-dark">
            <div className="data-meta"><span>LONG FORM</span><strong>116K</strong></div>
            <h3>LV116K</h3>
            <p>Builds timestamped evidence ledgers for temporal grounding, event timelines, and cross-segment question answering.</p>
            <div className="data-line"><i /><i /><i /><i /><i /><i /></div>
          </article>
          <article className="data-card data-green">
            <div className="data-meta"><span>ONLINE</span><strong>617K</strong></div>
            <h3>OL617K</h3>
            <p>Turns offline video QA into causal streams with explicit Silence, Standby, and Response supervision.</p>
            <div className="state-pills"><span>Silence</span><span>Standby</span><span>Response</span></div>
          </article>
        </div>
      </section>

      <section className="section results-section" id="results">
        <div className="shell">
          <p className="kicker">RESULTS</p>
          <div className="section-heading results-heading">
            <h2>Small model. Long reach.</h2>
            <p>At 4B parameters, VideoChat3 improves on 18 of 19 directly comparable offline metrics and 10 of 11 streaming metrics against Qwen3‑VL‑4B under the paper’s evaluation settings.</p>
          </div>
          <div className="result-panels">
            <article className="result-feature">
              <span className="result-label">ODVBENCH / OVERALL</span>
              <strong>72.3</strong>
              <p>+14.9 over Qwen3‑VL‑4B</p>
              <div className="benchmark-bar"><i style={{width:"72.3%"}} /></div>
            </article>
            <article className="result-feature result-feature-light">
              <span className="result-label">OVO‑TIMING / AVG. F1</span>
              <strong>35.5</strong>
              <p>+27.4 over Qwen3‑VL‑4B</p>
              <div className="benchmark-bar"><i style={{width:"71%"}} /></div>
            </article>
          </div>

          <div className="efficiency-card">
            <div className="efficiency-copy">
              <span className="mini-label">2,048 INPUT FRAMES · NVIDIA H200</span>
              <h3>Less work for the language model.</h3>
              <p>I3D‑ViT shifts computation into a linear vision stage and halves the visual tokens that enter the quadratic language stage.</p>
            </div>
            <div className="comparison" aria-label="Latency comparison at 2048 frames">
              <div className="comparison-row"><div><b>Qwen3‑VL</b><span>44.45 seconds</span></div><i className="bar-qwen" /></div>
              <div className="comparison-row"><div><b>VideoChat3</b><span>20.41 seconds</span></div><i className="bar-vc3" /></div>
              <p><strong>54% lower latency</strong> · 100,352 visual tokens · 80.8 GB</p>
            </div>
          </div>
          <p className="results-note">Results and efficiency figures are reported in the VideoChat3 paper. Benchmark protocols differ across suites.</p>
        </div>
      </section>

      <section className="section shell demos" id="demos">
        <p className="kicker">QUALITATIVE DEMOS</p>
        <div className="section-heading split-heading">
          <h2>See time become evidence.</h2>
          <p>From retrieving a detail deep inside a long video to waiting for the right moment in a live stream, these examples show the same model working across different temporal scales.</p>
        </div>
        <div className="demo-grid">
          <figure className="demo-wide"><img src={asset("/paper/proactive-demo.png")} alt="Proactive streaming response example" /><figcaption><span>01</span><b>Proactive response</b><p>Observe, wait, then respond.</p></figcaption></figure>
          <figure><img src={asset("/paper/long-video-demo.png")} alt="Long video question answering example" /><figcaption><span>02</span><b>Long-video QA</b><p>Retrieve details across extended context.</p></figcaption></figure>
          <figure><img src={asset("/paper/temporal-grounding-demo.png")} alt="Temporal video grounding example" /><figcaption><span>03</span><b>Temporal grounding</b><p>Map language to precise boundaries.</p></figcaption></figure>
        </div>
      </section>

      <section className="open-section">
        <div className="shell open-grid">
          <div>
            <p className="kicker">FULLY OPEN</p>
            <h2>Fully open. <em>Build on it.</em></h2>
          </div>
          <div className="open-copy">
            <p>VideoChat3 releases the pieces that turn a result into a research foundation: model weights, training code, training strategy, complete training datasets, and the data construction pipeline.</p>
            <div className="open-actions">
              <a className="button button-cream" href={codeUrl} target="_blank" rel="noreferrer"><GitHubIcon /><span>GitHub repository</span><Arrow /></a>
              <a className="button button-outline" href={modelUrl} target="_blank" rel="noreferrer"><HuggingFaceIcon /><span>Hugging Face collection</span><Arrow /></a>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer shell">
        <div className="footer-brand"><img src={asset("/parrot.png")} alt="" /><span>VideoChat3</span></div>
        <p>Nanjing University · Shanghai AI Laboratory · Nanyang Technological University · Peking University</p>
        <div className="footer-links"><a href={codeUrl}>Code</a><a href={modelUrl}>Models &amp; Data</a><a href="#top">Back to top ↑</a></div>
      </footer>
    </main>
  );
}
