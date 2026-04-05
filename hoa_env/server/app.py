from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from openenv.core import create_app
from ..models import HOAAction, HOAObservation
from ..environment import HOAEnvironment, HEURISTIC_CATEGORIES

app = create_app(
    env=HOAEnvironment,
    action_cls=HOAAction,
    observation_cls=HOAObservation,
    env_name="heuristic-override-arena",
    max_concurrent_envs=16,
)

# Keep a global env instance for stats
_stats_env = HOAEnvironment()

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
        
        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.08);
            color: #f5f5f7;
            font-size: 0.6875rem;
            font-weight: 500;
            padding: 0.375rem 0.875rem;
            border-radius: 6px;
            margin-bottom: 1.5rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border: 1px solid rgba(255,255,255,0.12);
        }
        
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
            margin-bottom: 3rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin-bottom: 4rem;
        }
        .stat-box {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 600;
            color: #f5f5f7;
        }
        .stat-label {
            font-size: 0.75rem;
            color: #86868b;
            margin-top: 0.25rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
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
            gap: 0;
        }
        .task {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        .task:last-child { border-bottom: none; }
        .task-name { 
            font-size: 1.125rem;
            font-weight: 500;
        }
        .task-info {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .task-count {
            color: #86868b;
            font-size: 0.875rem;
        }
        .difficulty-badge {
            font-size: 0.6875rem;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-weight: 500;
            text-transform: uppercase;
        }
        .diff-easy { background: #1d4d1d; color: #4ade80; }
        .diff-medium { background: #4d3d1d; color: #fbbf24; }
        .diff-hard { background: #4d1d1d; color: #f87171; }
        
        .bias-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.75rem;
        }
        .bias-tag {
            background: rgba(255,255,255,0.05);
            padding: 0.625rem 1rem;
            border-radius: 8px;
            font-size: 0.8125rem;
            color: #a1a1a6;
            text-align: center;
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
            grid-template-columns: repeat(3, 1fr);
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
            font-size: 0.8125rem;
            color: #a1a1a6;
            overflow-x: auto;
            white-space: pre-wrap;
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
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .endpoints { grid-template-columns: 1fr; }
            .bias-grid { grid-template-columns: repeat(2, 1fr); }
        }
    </style>
</head>
<body>
    <div class="container">
        <span class="badge">OpenEnv Hackathon 2026</span>
        <h1>Heuristic Override Arena</h1>
        <p class="tagline">Real-world compliance decisions: procurement, HR, medical. Policy beats shortcuts.</p>
        
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">100</div>
                <div class="stat-label">Scenarios</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">5</div>
                <div class="stat-label">Domains</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">15+</div>
                <div class="stat-label">Bias Types</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">3</div>
                <div class="stat-label">Difficulty</div>
            </div>
        </div>
        
        <section>
            <h2>Tasks</h2>
            <div class="tasks">
                <div class="task">
                    <span class="task-name">procurement</span>
                    <div class="task-info">
                        <span class="task-count">25 scenarios</span>
                        <span class="difficulty-badge diff-easy">Easy</span>
                    </div>
                </div>
                <div class="task">
                    <span class="task-name">hr_decision</span>
                    <div class="task-info">
                        <span class="task-count">25 scenarios</span>
                        <span class="difficulty-badge diff-medium">Medium</span>
                    </div>
                </div>
                <div class="task">
                    <span class="task-name">medical_triage</span>
                    <div class="task-info">
                        <span class="task-count">25 scenarios</span>
                        <span class="difficulty-badge diff-hard">Hard</span>
                    </div>
                </div>
                <div class="task">
                    <span class="task-name">cognitive_biases</span>
                    <div class="task-info">
                        <span class="task-count">15 scenarios</span>
                        <span class="difficulty-badge diff-medium">Medium</span>
                    </div>
                </div>
                <div class="task">
                    <span class="task-name">edge_cases</span>
                    <div class="task-info">
                        <span class="task-count">10 scenarios</span>
                        <span class="difficulty-badge diff-hard">Hard</span>
                    </div>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Cognitive Bias Types</h2>
            <div class="bias-grid">
                <div class="bias-tag">cost</div>
                <div class="bias-tag">authority_bias</div>
                <div class="bias-tag">sunk_cost</div>
                <div class="bias-tag">recency_bias</div>
                <div class="bias-tag">familiarity</div>
                <div class="bias-tag">social_pressure</div>
                <div class="bias-tag">status_quo</div>
                <div class="bias-tag">severity</div>
                <div class="bias-tag">experience</div>
                <div class="bias-tag">urgency</div>
                <div class="bias-tag">affinity_bias</div>
                <div class="bias-tag">prestige</div>
            </div>
        </section>
        
        <section>
            <h2>Reward Function</h2>
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
                <div class="endpoint">
                    <div class="endpoint-method">GET /info</div>
                    <div class="endpoint-desc">Environment info</div>
                </div>
                <div class="endpoint">
                    <div class="endpoint-method">GET /tasks</div>
                    <div class="endpoint-desc">List all tasks</div>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Quick Start</h2>
            <div class="try-box"># Start episode with curriculum
curl -X POST https://wysh3-heuristic-override-arena.hf.space/reset \\
  -H "Content-Type: application/json" \\
  -d '{"difficulty": "easy"}'

# Or specific task
curl -X POST .../reset -d '{"task": "cognitive_biases"}'</div>
        </section>
        
        <footer>
            <span>Team Z · Vishruth M R · Akshay Kumar</span>
            <a href="https://arxiv.org/abs/2603.29025">Based on arXiv:2603.29025</a>
        </footer>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return DEMO_HTML

@app.get("/info")
def info():
    """Return environment information and statistics."""
    return {
        "name": "heuristic-override-arena",
        "version": "2.0.0",
        "total_scenarios": 100,
        "domains": 5,
        "bias_types": 15,
        "tasks": _stats_env.get_available_tasks(),
        "difficulty_levels": ["easy", "medium", "hard"],
        "heuristic_categories": HEURISTIC_CATEGORIES,
        "based_on": "arXiv:2603.29025",
    }

@app.get("/tasks")
def tasks():
    """List all available tasks with their scenario counts."""
    env = HOAEnvironment()
    task_info = {}
    for task in env.get_available_tasks():
        scenarios = env._scenarios.get(task, [])
        task_info[task] = {
            "count": len(scenarios),
            "heuristic_types": list(set(
                s.get("ground_truth", {}).get("heuristic_type", "unknown")
                for s in scenarios
            )),
        }
    return task_info
