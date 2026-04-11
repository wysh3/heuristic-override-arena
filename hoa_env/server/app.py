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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body { 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #050505;
            background-image: 
                radial-gradient(ellipse at 50% -20%, rgba(255, 87, 34, 0.15), transparent 50%),
                radial-gradient(ellipse at 50% 0%, rgba(255, 69, 0, 0.1), transparent 70%);
            color: #ffffff;
            min-height: 100vh;
            padding: 5rem 2rem;
            -webkit-font-smoothing: antialiased;
            line-height: 1.6;
        }
        
        .container { max-width: 900px; margin: 0 auto; position: relative; z-index: 1; }
        
        /* Typography & Header */
        .header-section { text-align: center; margin-bottom: 5rem; }
        
        .badge {
            display: inline-block;
            background: rgba(255, 87, 34, 0.1);
            color: #FF5722;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 50px;
            margin-bottom: 2rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            border: 1px solid rgba(255, 87, 34, 0.2);
            box-shadow: 0 0 20px rgba(255, 87, 34, 0.1);
        }
        
        h1 { 
            font-size: 3.5rem; 
            font-weight: 700; 
            letter-spacing: -0.02em;
            line-height: 1.2;
            margin-bottom: 1.5rem;
        }
        
        .text-accent {
            background: linear-gradient(135deg, #FF7043 0%, #FFAB91 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .tagline { 
            font-size: 1.125rem; 
            color: #A1A1A6; 
            max-width: 600px;
            margin: 0 auto;
        }

        /* Reusable Card Styling */
        .glass-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        .glass-card:hover {
            border-color: rgba(255, 255, 255, 0.15);
            background: rgba(255, 255, 255, 0.04);
            transform: translateY(-3px);
            box-shadow: 0 10px 40px rgba(255, 255, 255, 0.02);
        }
        
        section { margin-bottom: 5rem; }
        section h2 { 
            font-size: 0.875rem; 
            font-weight: 600; 
            color: #FF5722; 
            margin-bottom: 1.5rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        section h2::before {
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #FF5722;
            border-radius: 50%;
            box-shadow: 0 0 10px #FF5722;
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1.5rem;
            margin-bottom: 5rem;
        }
        .stat-box {
            padding: 2rem 1.5rem;
            text-align: center;
        }
        .stat-value {
            font-size: 2.75rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.5rem;
            letter-spacing: -0.03em;
        }
        .stat-label {
            font-size: 0.75rem;
            color: #86868B;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }
        
        /* Tasks Grid */
        .tasks {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        .task {
            padding: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .task-name { 
            font-size: 1.125rem;
            font-weight: 500;
            color: #E8E8ED;
        }
        .task-info {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 0.5rem;
        }
        .task-count { color: #86868B; font-size: 0.875rem; }
        
        /* Badges */
        .difficulty-badge {
            font-size: 0.7rem;
            padding: 0.25rem 0.6rem;
            border-radius: 4px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .diff-easy { background: rgba(74, 222, 128, 0.1); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.2); }
        .diff-medium { background: rgba(251, 191, 36, 0.1); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.2); }
        .diff-hard { background: rgba(248, 113, 113, 0.1); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.2); }
        
        /* Biases Grid */
        .bias-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }
        .bias-tag {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            padding: 0.625rem 1.25rem;
            border-radius: 50px;
            font-size: 0.875rem;
            color: #A1A1A6;
            transition: all 0.3s cubic-bezier(0.165, 0.84, 0.44, 1);
        }
        .bias-tag:hover {
            color: #FFF;
            border-color: rgba(255, 255, 255, 0.15);
            background: rgba(255, 255, 255, 0.06);
            box-shadow: 0 4px 15px rgba(255, 255, 255, 0.03);
            transform: translateY(-1px);
        }
        
        /* Formula */
        .formula-box {
            padding: 2.5rem;
            text-align: center;
            background: linear-gradient(145deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
            border: 1px solid rgba(255,255,255,0.05);
        }
        .formula {
            font-size: 1.25rem;
            font-weight: 500;
            font-family: 'SF Mono', 'Menlo', monospace;
            color: #FF5722;
            letter-spacing: -0.01em;
        }
        
        /* Endpoints */
        .endpoints {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }
        .endpoint {
            padding: 1.5rem;
        }
        .endpoint-method {
            font-family: 'SF Mono', 'Menlo', monospace;
            font-size: 0.875rem;
            font-weight: 600;
            color: #E8E8ED;
            margin-bottom: 0.5rem;
        }
        .endpoint-desc { font-size: 0.875rem; color: #86868B; }
        
        /* Code Box */
        .try-box {
            padding: 2rem;
            font-family: 'SF Mono', 'Menlo', monospace;
            font-size: 0.875rem;
            color: #A1A1A6;
            overflow-x: auto;
            white-space: pre-wrap;
            line-height: 1.7;
            position: relative;
        }
        
        footer {
            margin-top: 6rem;
            padding-top: 3rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            justify-content: space-between;
            font-size: 0.875rem;
            color: #6E6E73;
        }
        footer a { color: #FF5722; text-decoration: none; transition: color 0.2s; }
        footer a:hover { color: #FF8A65; text-decoration: underline; }
        
        @media (max-width: 768px) {
            h1 { font-size: 2.5rem; }
            .stats-grid { grid-template-columns: repeat(2, 1fr); }
            .tasks { grid-template-columns: 1fr; }
            .endpoints { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 480px) {
            .endpoints { grid-template-columns: 1fr; }
            .stats-grid { gap: 1rem; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-section">
            <span class="badge">OpenEnv Hackathon 2026</span>
            <h1>Heuristic <span class="text-accent">Override</span> Arena</h1>
            <p class="tagline">Real-world compliance decisions: procurement, HR, medical. Policy beats shortcuts.</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-box glass-card">
                <div class="stat-value">100</div>
                <div class="stat-label">Scenarios</div>
            </div>
            <div class="stat-box glass-card">
                <div class="stat-value">5</div>
                <div class="stat-label">Domains</div>
            </div>
            <div class="stat-box glass-card">
                <div class="stat-value">15+</div>
                <div class="stat-label">Bias Types</div>
            </div>
            <div class="stat-box glass-card">
                <div class="stat-value">3</div>
                <div class="stat-label">Difficulty Levels</div>
            </div>
        </div>
        
        <section>
            <h2>Arena Tasks</h2>
            <div class="tasks">
                <div class="task glass-card">
                    <span class="task-name">Procurement</span>
                    <div class="task-info">
                        <span class="difficulty-badge diff-easy">Easy</span>
                        <span class="task-count">25 scenarios</span>
                    </div>
                </div>
                <div class="task glass-card">
                    <span class="task-name">HR Decision</span>
                    <div class="task-info">
                        <span class="difficulty-badge diff-medium">Medium</span>
                        <span class="task-count">25 scenarios</span>
                    </div>
                </div>
                <div class="task glass-card">
                    <span class="task-name">Medical Triage</span>
                    <div class="task-info">
                        <span class="difficulty-badge diff-hard">Hard</span>
                        <span class="task-count">25 scenarios</span>
                    </div>
                </div>
                <div class="task glass-card">
                    <span class="task-name">Cognitive Biases</span>
                    <div class="task-info">
                        <span class="difficulty-badge diff-medium">Medium</span>
                        <span class="task-count">15 scenarios</span>
                    </div>
                </div>
                <div class="task glass-card">
                    <span class="task-name">Edge Cases</span>
                    <div class="task-info">
                        <span class="difficulty-badge diff-hard">Hard</span>
                        <span class="task-count">10 scenarios</span>
                    </div>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Cognitive Bias Types</h2>
            <div class="bias-grid">
                <div class="bias-tag">Cost</div>
                <div class="bias-tag">Authority Bias</div>
                <div class="bias-tag">Sunk Cost</div>
                <div class="bias-tag">Recency Bias</div>
                <div class="bias-tag">Familiarity</div>
                <div class="bias-tag">Social Pressure</div>
                <div class="bias-tag">Status Quo</div>
                <div class="bias-tag">Severity</div>
                <div class="bias-tag">Experience</div>
                <div class="bias-tag">Urgency</div>
                <div class="bias-tag">Affinity Bias</div>
                <div class="bias-tag">Prestige</div>
            </div>
        </section>
        
        <section>
            <h2>Reward Function</h2>
            <div class="formula-box glass-card">
                <span class="formula">0.6·correct + 0.2·constraint + 0.2·heuristic − 0.3·trap</span>
            </div>
        </section>
        
        <section>
            <h2>API Endpoints</h2>
            <div class="endpoints">
                <div class="endpoint glass-card">
                    <div class="endpoint-method">POST /reset</div>
                    <div class="endpoint-desc">Start episode</div>
                </div>
                <div class="endpoint glass-card">
                    <div class="endpoint-method">POST /step</div>
                    <div class="endpoint-desc">Submit action</div>
                </div>
                <div class="endpoint glass-card">
                    <div class="endpoint-method">GET /state</div>
                    <div class="endpoint-desc">Current state</div>
                </div>
                <div class="endpoint glass-card">
                    <div class="endpoint-method">GET /health</div>
                    <div class="endpoint-desc">Health check</div>
                </div>
                <div class="endpoint glass-card">
                    <div class="endpoint-method">GET /info</div>
                    <div class="endpoint-desc">Environment info</div>
                </div>
                <div class="endpoint glass-card">
                    <div class="endpoint-method">GET /tasks</div>
                    <div class="endpoint-desc">List all tasks</div>
                </div>
            </div>
        </section>
        
        <section>
            <h2>Quick Start</h2>
            <div class="try-box glass-card"># Start episode with curriculum
curl -X POST https://wysh3-heuristic-override-arena.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Or specific task
curl -X POST .../reset -d '{"task": "cognitive_biases"}'</div>
        </section>
        
        <footer>
            <span>Team Z · Vishruth M R · Akshay Kumar</span>
            <a href="https://arxiv.org/abs/2603.29025" target="_blank">Based on arXiv:2603.29025</a>
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
