from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openenv.core import create_app
from ..models import HOAAction, HOAObservation
from ..environment import HOAEnvironment

app = create_app(
    env=HOAEnvironment,
    action_cls=HOAAction,
    observation_cls=HOAObservation,
    env_name="heuristic-override-arena",
    max_concurrent_envs=16,
)

DEMO_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Heuristic Override Arena</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif;
            background: #000;
            color: #fff;
            min-height: 100vh;
            padding: 6rem 2rem;
            -webkit-font-smoothing: antialiased;
        }
        .container { max-width: 720px; margin: 0 auto; }
        
        h1 { 
            font-size: 3rem; 
            font-weight: 600; 
            letter-spacing: -0.03em;
            line-height: 1.1;
            margin-bottom: 1rem;
        }
        .tagline { 
            font-size: 1.25rem; 
            color: #86868b; 
            font-weight: 400;
            margin-bottom: 5rem;
        }
        
        section { margin-bottom: 4rem; }
        section h2 { 
            font-size: 0.875rem; 
            font-weight: 500; 
            color: #86868b; 
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .tasks {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        .task {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .task-name { 
            font-size: 1.125rem;
            font-weight: 500;
        }
        .task-trap { 
            color: #86868b;
            font-size: 0.9375rem;
        }
        
        .formula-box {
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
        }
        .formula {
            font-size: 1.125rem;
            font-weight: 400;
            font-family: 'SF Mono', 'Menlo', monospace;
            color: #f5f5f7;
            letter-spacing: -0.01em;
        }
        
        .endpoints {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        .endpoint {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
        }
        .endpoint-method {
            font-family: 'SF Mono', 'Menlo', monospace;
            font-size: 0.9375rem;
            font-weight: 500;
            margin-bottom: 0.25rem;
        }
        .endpoint-desc {
            font-size: 0.8125rem;
            color: #86868b;
        }
        
        .try-box {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 1.5rem;
            font-family: 'SF Mono', 'Menlo', monospace;
            font-size: 0.875rem;
            color: #a1a1a6;
            overflow-x: auto;
        }
        
        footer {
            margin-top: 6rem;
            padding-top: 2rem;
            border-top: 1px solid rgba(255,255,255,0.08);
            display: flex;
            justify-content: space-between;
            font-size: 0.8125rem;
            color: #86868b;
        }
        footer a { 
            color: #2997ff; 
            text-decoration: none;
        }
        footer a:hover { text-decoration: underline; }
        
        @media (max-width: 600px) {
            h1 { font-size: 2rem; }
            .endpoints { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Heuristic Override Arena</h1>
        <p class="tagline">Train agents to override satisficing shortcuts.</p>
        
        <section>
            <h2>Tasks</h2>
            <div class="tasks">
                <div class="task">
                    <span class="task-name">procurement</span>
                    <span class="task-trap">resists "cheaper vendor"</span>
                </div>
                <div class="task">
                    <span class="task-name">hr_decision</span>
                    <span class="task-trap">resists "more experience"</span>
                </div>
                <div class="task">
                    <span class="task-name">medical_triage</span>
                    <span class="task-trap">resists "sicker patient"</span>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Reward</h2>
            <div class="formula-box">
                <span class="formula">0.6·correct + 0.2·constraint + 0.2·heuristic − 0.3·trap</span>
            </div>
        </section>
        
        <section>
            <h2>Endpoints</h2>
            <div class="endpoints">
                <div class="endpoint">
                    <div class="endpoint-method">POST /reset</div>
                    <div class="endpoint-desc">Start episode</div>
                </div>
                <div class="endpoint">
                    <div class="endpoint-method">POST /step</div>
                    <div class="endpoint-desc">Submit action</div>
                </div>
                <div class="endpoint">
                    <div class="endpoint-method">GET /state</div>
                    <div class="endpoint-desc">Current state</div>
                </div>
                <div class="endpoint">
                    <div class="endpoint-method">GET /health</div>
                    <div class="endpoint-desc">Health check</div>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Try</h2>
            <div class="try-box">curl -X POST https://wysh3-heuristic-override-arena.hf.space/reset</div>
        </section>
        
        <footer>
            <span>Team Z · OpenEnv 2026</span>
            <a href="https://arxiv.org/abs/2603.29025">arXiv:2603.29025</a>
        </footer>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return DEMO_HTML
