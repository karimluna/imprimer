"""
Three-panel interface: Input, Analysis and Optimization
"""

import os
import sys
import html
from dotenv import load_dotenv

load_dotenv()
# SSL workaround for certain environments
ssl_cert_file = os.environ.get("SSL_CERT_FILE")
if ssl_cert_file and not os.path.exists(ssl_cert_file):
    del os.environ["SSL_CERT_FILE"]

engine_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "engine")
)
if engine_path not in sys.path:
    sys.path.insert(0, engine_path)

import gradio as gr
from core.analyzer.stability import analyze as run_stability
from core.optimizer.graph import optimize as run_optimize
from core.registry.prompt_store import best_variant_for_task, init_db
from core.chains.prompt_chain import ModelBackend


# Standard task categories to keep the registry clean
TASK_CATEGORIES = [
    "summarize",
    "classify",
    "extract",
    "translate",
    "reasoning",
    "creative_writing",
    "code_generation",
    "rewrite",
    "roleplay",
    "qa"
]

BACKEND_ID = ModelBackend.OLLAMA # harcoded backend for dev and demo
BEST_PROMPT = []


CUSTOM_CSS = """
:root {
  --color-background-primary: #ffffff;
  --color-background-secondary: #f8f9fa;
  --color-border-primary: #e5e7eb;
  --color-border-secondary: #e5e7eb;
  --color-border-tertiary: #f3f4f6;
  --color-text-primary: #111827;
  --color-text-secondary: #374151;
  --color-text-tertiary: #6b7280;
  --border-radius-md: 8px;
}

.section-label { font-size: 11px; font-weight: 500; color: var(--color-text-tertiary); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; margin-top: 16px; }

.metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin-bottom: 1rem; }
.metric { background: var(--color-background-secondary); border-radius: var(--border-radius-md); padding: 10px 12px; border: 1px solid var(--color-border-tertiary); }
.metric .label { font-size: 11px; color: var(--color-text-tertiary); margin-bottom: 4px; }
.metric .value { font-size: 20px; font-weight: 500; color: var(--color-text-primary); }
.metric .delta { font-size: 11px; margin-top: 2px; }
.delta.pos { color: #1D9E75; }
.delta.neg { color: #D85A30; }

.timeline { display: flex; flex-direction: column; gap: 4px; margin-bottom: 1rem; }
.iter-row { display: grid; grid-template-columns: 80px 1fr 70px 70px; gap: 8px; align-items: center; padding: 8px 12px; border-radius: var(--border-radius-md); font-size: 12px; border: 1px solid var(--color-border-tertiary); background: var(--color-background-primary); }
.iter-row .iter-label { color: var(--color-text-secondary); font-weight: 500; }
.bar-wrap { background: var(--color-background-secondary); border-radius: 4px; height: 6px; position: relative; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; background: #1D9E75; transition: width 0.4s ease; }
.bar-fill.base { background: #888780; }
.score-val { text-align: right; font-weight: 500; font-size: 12px; }
.score-val.improved { color: #1D9E75; }
.iter-row.running { border-color: #FAC775; background: #FAEEDA22; }
.iter-row.done { border-color: var(--color-border-tertiary); }
.iter-row.pending { opacity: 0.5; }
.spin { display: inline-block; animation: spin 1s linear infinite; font-size: 12px; margin-left: 4px; }
@keyframes spin { to { transform: rotate(360deg); } }

.prompt-compare { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 1rem; }
.prompt-box { border: 1px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); padding: 12px; font-size: 13px; background: var(--color-background-primary); }
.prompt-box .box-label { font-size: 10px; font-weight: 600; color: var(--color-text-tertiary); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px; }
.prompt-box .text { color: var(--color-text-secondary); line-height: 1.5; white-space: pre-wrap; }
.prompt-box.best .box-label { color: #1D9E75; }
.prompt-box.best { border-color: #5DCAA5; background: #f0fdf455; }

.feedback-card { border-left: 4px solid #5DCAA5; padding: 12px; background: #E1F5EE55; border-radius: 0 var(--border-radius-md) var(--border-radius-md) 0; margin-bottom: 1rem; font-size: 13px; color: var(--color-text-secondary); line-height: 1.6; }
.feedback-card .fb-label { font-size: 10px; font-weight: 600; color: #0F6E56; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }

.run-list { display: flex; flex-direction: column; gap: 8px; }
details.run-item { border: 1px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); overflow: hidden; background: var(--color-background-primary); }
details.run-item > summary { display: flex; align-items: center; gap: 8px; padding: 10px 12px; cursor: pointer; font-size: 13px; font-weight: 500; list-style: none; }
details.run-item > summary::-webkit-details-marker { display: none; }
.run-header .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.run-header .dot.done { background: #1D9E75; }
.run-header .run-score { margin-left: auto; font-weight: 600; color: #1D9E75; }
.run-body { padding: 0px 12px 12px; border-top: 1px solid transparent; font-size: 12px; color: var(--color-text-secondary); line-height: 1.6; white-space: pre-wrap; }
details.run-item[open] > summary { border-bottom: 1px solid var(--color-border-tertiary); margin-bottom: 8px; }

.token-strip { display: flex; flex-wrap: wrap; gap: 4px; padding: 12px; background: var(--color-background-primary); border: 1px solid var(--color-border-tertiary); border-radius: var(--border-radius-md); margin-top: 8px; }
.tok { font-size: 12px; font-family: monospace; padding: 2px 6px; border-radius: 4px; border-bottom: 2px solid transparent; }

.status-bar { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-radius: var(--border-radius-md); font-size: 13px; font-weight: 500; margin-bottom: 1rem; border: 1px solid var(--color-border-tertiary); background: var(--color-background-secondary); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #EF9F27; animation: pulse 1s ease-in-out infinite; flex-shrink: 0; }
.status-dot.done { background: #1D9E75; animation: none; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
"""

def _render_token_confidence(token_confidence: list) -> str:
    """Renders token-level confidence using the new HTML styles."""
    if not token_confidence:
        return "<p>No token confidence data available.</p>"

    html_str = '<div class="token-strip">'
    for tc in token_confidence:
        certainty = tc.get("certainty", 0.5)
        r = int(216 if certainty < 0.5 else 29)
        g = int(90 if certainty < 0.5 else 158)
        b = int(48 if certainty < 0.5 else 117)
        color = f"#{r:02x}{g:02x}{b:02x}"
        bg = f"rgba({r},{g},{b},0.15)"
        
        token = html.escape(tc.get("token", ""))
        logprob = tc.get("logprob", 0)
        html_str += (
            f'<span class="tok" title="certainty={certainty:.3f} logprob={logprob:.3f}" '
            f'style="background:{bg};border-color:{color};">{token}</span>'
        )
    html_str += "</div>"
    return html_str

def build_status_bar(text, is_done=False):
    dot_class = "done" if is_done else ""
    return f"""
    <div class="status-bar">
        <div class="status-dot {dot_class}"></div>
        <span>{html.escape(text)}</span>
    </div>
    """

def build_metric_html(label, value, delta=None, is_target=False):
    delta_html = ""
    if delta is not None and not is_target:
        cls = "pos" if delta > 0 else "neg"
        sign = "+" if delta > 0 else ""
        delta_html = f'<div class="delta {cls}">{sign}{delta:.3f}</div>'
    
    val_str = f"{value:.3f}" if isinstance(value, float) else str(value)
    return f"""
    <div class="metric">
      <div class="label">{html.escape(label)}</div>
      <div class="value">{val_str}</div>
      {delta_html}
    </div>
    """


def run_optimization(
    prompt, input_text, task, model_id, hf_token,
    expected_output, n_variants, target_score, max_iterations, use_judge
):
    global BEST_PROMPT
    if not prompt or not task:
        yield "<div class='feedback-card'>Prompt, task, and expected output are required.</div>", "", "", "", ""
        return

    # Backend environment setup
    if BACKEND_ID == ModelBackend.HUGGINGFACE:
        if model_id: os.environ["HF_MODEL_ID"] = model_id
        if hf_token: os.environ["HF_TOKEN"] = hf_token
    elif BACKEND_ID == ModelBackend.OLLAMA:
        if model_id: os.environ["OLLAMA_MODEL"] = model_id

    # INITIAL UI STATE
    status_html = build_status_bar("Initializing optimization graph...", is_done=False)
    metrics_html = f"""<div class="metric-row">
        {build_metric_html("Baseline", "---")}
        {build_metric_html("Best so far", "---")}
        {build_metric_html("Target", target_score)}
        {build_metric_html("Cycles", f"0 / {max_iterations}")}
    </div>"""
    
    prompt_html = f"""<div class="prompt-compare">
        <div class="prompt-box"><div class="box-label">Original</div><div class="text">{html.escape(prompt)}</div></div>
        <div class="prompt-box best"><div class="box-label">Optimized</div><div class="text">⏳ Waiting for first cycle...</div></div>
    </div>"""
    
    timeline_html = "<div class='timeline'><div class='iter-row running'><span class='iter-label'>Cycle 1 <span class='spin'>↻</span></span><div class='bar-wrap'><div class='bar-fill base' style='width:0%'></div></div><span class='score-val'>—</span><span class='score-val'>—</span></div></div>"
    feedback_html = "<div class='feedback-card'>⏳ Waiting for AI judge reflection...</div>"

    yield status_html, metrics_html, timeline_html, prompt_html, feedback_html

    try:
        optimizer_output = run_optimize(
            task=task,
            base_prompt=prompt,
            input_example=input_text,
            expected_output=expected_output,
            n_variants=int(n_variants),
            backend=BACKEND_ID,
            use_judge=bool(use_judge),
            use_rpe=True,
            target_score=float(target_score), 
            max_iterations=int(max_iterations),
        )
        
        final_result = None
        
        if hasattr(optimizer_output, '__iter__') and not isinstance(optimizer_output, dict):
            for step_result in optimizer_output:
                final_result = step_result
                iteration = step_result.get('iterations_completed', step_result.get('current_iteration', 1))
                
                base_s = step_result.get('baseline_score', 0.0)
                best_s = step_result.get('best_score', 0.0)
                improv = step_result.get('improvement', 0.0)
                curr_p = step_result.get('best_prompt', '')
                fb_str = step_result.get('feedback', '')
                BEST_PROMPT.append(curr_p)

                # Status & Metrics
                status_html = build_status_bar(f"Cycle {iteration} of {max_iterations} — evaluating variants...", is_done=False)
                metrics_html = f"""<div class="metric-row">
                    {build_metric_html("Baseline", base_s)}
                    {build_metric_html("Best so far", best_s, delta=improv)}
                    {build_metric_html("Target", target_score)}
                    {build_metric_html("Cycles", f"{iteration} / {max_iterations}")}
                </div>"""

                # Timeline Construction
                tl = '<div class="timeline">'
                for i in range(1, int(max_iterations) + 1):
                    if i < iteration:
                        # Completed steps (We mock the history visualization strictly based on current progress)
                        tl += f"""<div class="iter-row done">
                            <span class="iter-label">Cycle {i}</span>
                            <div class="bar-wrap"><div class="bar-fill base" style="width:{min(100, (best_s/1.0)*100)}%"></div></div>
                            <span class="score-val">{best_s:.3f}</span><span class="score-val improved">+{improv:.3f}</span>
                        </div>"""
                    elif i == iteration:
                        tl += f"""<div class="iter-row running">
                            <span class="iter-label">Cycle {i} <span class="spin">↻</span></span>
                            <div class="bar-wrap"><div class="bar-fill" style="width:{min(100, (best_s/1.0)*100)}%"></div></div>
                            <span class="score-val improved">{best_s:.3f}</span><span class="score-val improved">+{improv:.3f}</span>
                        </div>"""
                    else:
                        tl += f"""<div class="iter-row pending">
                            <span class="iter-label">Cycle {i}</span>
                            <div class="bar-wrap"><div class="bar-fill" style="width:0%"></div></div>
                            <span class="score-val">—</span><span class="score-val">—</span>
                        </div>"""
                tl += '</div>'

                # Prompts & Feedback
                prompt_html = f"""<div class="prompt-compare">
                    <div class="prompt-box"><div class="box-label">Original</div><div class="text">{html.escape(prompt)}</div></div>
                    <div class="prompt-box best"><div class="box-label">Best so far (Cycle {iteration})</div><div class="text">{html.escape(curr_p)}</div></div>
                </div>"""
                
                feedback_html = f"""<div class="feedback-card">
                    <div class="fb-label">Cycle {iteration-1 if fb_str else iteration} Reflection</div>
                    {html.escape(fb_str) if fb_str else "Generating new variations and scoring..."}
                </div>"""

                yield status_html, metrics_html, tl, prompt_html, feedback_html
        else:
            final_result = optimizer_output
            BEST_PROMPT.append(final_result.get('best_prompt', ''))
            
    except Exception as e:
        err = f"<div class='feedback-card' style='border-color:red;'><div class='fb-label' style='color:red;'>Error</div>{html.escape(str(e))}</div>"
        yield build_status_bar("Optimization failed", True), "", "", "", err
        return

    result = final_result or {}
    status_msg = "Target score reached — optimization complete" if result.get("target_reached") else "Iteration cap reached — optimization finished"
    final_iteration = result.get('iterations_completed', result.get('current_iteration', max_iterations))
    
    base_s = result.get('baseline_score', 0.0)
    best_s = result.get('best_score', 0.0)
    improv = result.get('improvement', 0.0)

    status_html = build_status_bar(status_msg, is_done=True)
    metrics_html = f"""<div class="metric-row">
        {build_metric_html("Baseline", base_s)}
        {build_metric_html("Final Best", best_s, delta=improv)}
        {build_metric_html("Target", target_score)}
        {build_metric_html("Cycles", f"{final_iteration} / {max_iterations}")}
    </div>"""

    tl = '<div class="timeline">'
    for i in range(1, int(final_iteration) + 1):
        tl += f"""<div class="iter-row done">
            <span class="iter-label">Cycle {i}</span>
            <div class="bar-wrap"><div class="bar-fill" style="width:{min(100, (best_s/1.0)*100)}%"></div></div>
            <span class="score-val">{best_s:.3f}</span><span class="score-val improved">+{improv:.3f}</span>
        </div>"""
    tl += '</div>'

    prompt_html = f"""<div class="prompt-compare">
        <div class="prompt-box"><div class="box-label">Original</div><div class="text">{html.escape(prompt)}</div></div>
        <div class="prompt-box best"><div class="box-label">Final Optimized Prompt</div><div class="text">{html.escape(result.get('best_prompt', ''))}</div></div>
    </div>"""

    fb = result.get('feedback', '')
    feedback_html = f"""<div class="feedback-card">
        <div class="fb-label">Final Output Reflection</div>
        {html.escape(fb) if fb else "Target reached or maximum iterations exhausted."}
    </div>"""

    yield status_html, metrics_html, tl, prompt_html, feedback_html


def run_analysis(prompt, input_text, task, model_id, hf_token, n_runs, temperature):

    if not prompt or not task:
        return "<div class='feedback-card'>Prompt and task are required.</div>", "", "", "", None, None

    if model_id: os.environ["HF_MODEL_ID"] = model_id
    if hf_token: os.environ["HF_TOKEN"] = hf_token

    try:
        result = run_stability(
            prompt=prompt,
            input_text=input_text,
            task=task,
            backend=BACKEND_ID,
            n_runs=int(n_runs),
            temperature=float(temperature),
        )
    except Exception as e:
        err = f"<div class='feedback-card' style='border-color:red;'><div class='fb-label' style='color:red;'>Error</div>{html.escape(str(e))}</div>"
        return build_status_bar("Analysis Failed", True), "", err, "", None, None

    score = result.stability_score
    status_msg = f"Analysis complete — {n_runs} runs, stability score {score:.3f}"
    status_html = build_status_bar(status_msg, is_done=True)

    # We append the new recommendation here inside a feedback-card
    metrics_html = f"""<div class="metric-row">
        {build_metric_html("Stability Score", score)}
        {build_metric_html("Avg Reachability", result.avg_reachability)}
        {build_metric_html("Avg Similarity", result.avg_similarity)}
        {build_metric_html("Variance", result.variance)}
    </div>
    <div class="feedback-card" style="margin-top: 1rem;">
        <div class="fb-label">Analysis Recommendation</div>
        {html.escape(getattr(result, 'recommendation', 'No recommendation provided.'))}
    </div>
    """

    outputs_html = '<div class="run-list">'
    for i, out in enumerate(result.outputs):
        # First item open by default
        open_attr = "open" if i == 0 else ""
        outputs_html += f"""
        <details class="run-item" {open_attr}>
            <summary class="run-header">
                <div class="dot done"></div>
                <span>Run {i+1}</span>
            </summary>
            <div class="run-body">{html.escape(out)}</div>
        </details>
        """
    outputs_html += '</div>'

    token_html = _render_token_confidence([
        {"token": tc.token, "certainty": tc.certainty, "logprob": tc.logprob}
        for tc in result.token_confidence
    ])

    return status_html, metrics_html, outputs_html, token_html, result, None


def query_best(task, limit):
    if not task:
        return "Task is required."
    try:
        result = best_variant_for_task(task, limit=int(limit))
        if not result.get("task"):
            return f"No evaluations found for task '{task}'."
        
        return f"""**Task:** {result['task']}
**Evaluations sampled:** {result['evaluations']}
**Avg score:** {result['avg_score']:.4f}

**Best prompt:** {result['best_template']}
"""
    except Exception as e:
        return str(e)


with gr.Blocks(title="Imprimer - LLM Prompt Control") as demo:

    gr.Markdown("""
# Imprimer - LLM Prompt Control Platform

> *Prompts don't instruct a unified mind - they activate configurations within it.*
> Imprimer makes those activations **measurable**, **comparable**, and **improvable**.
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Setup")
            prompt_input = gr.Textbox(
                label="Prompt template",
                placeholder="Summarize this in one sentence: {input}",
                lines=3,
            )
            input_text = gr.Textbox(
                label="Input text",
                placeholder="The text your prompt will process...",
                lines=3,
            )
            
            task_input = gr.Dropdown(
                label="Task type",
                choices=TASK_CATEGORIES,
                value="summarize",
                allow_custom_value=True,
            )
            
            with gr.Row():
                try:
                    if BACKEND_ID == ModelBackend.HUGGINGFACE:
                        model_id = gr.Dropdown(
                            label="Hugging Face Model ID",
                            choices=[
                                "HuggingFaceTB/SmolLM2-1.7B-Instruct",
                                "Qwen/Qwen2.5-1.5B-Instruct",
                                "microsoft/phi-2",
                                "google/gemma-2b-it",
                                "meta-llama/Llama-3.2-1B-Instruct"
                            ],
                            value="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                            allow_custom_value=True,
                        )
                    else:
                        model_id = gr.Dropdown(
                            label="Ollama Model",
                            choices=[
                                "llama3.2:latest",                
                                "qwen2.5:0.5b",                          
                                "qwen2.5:1.5b",                        
                            ],
                            value="qwen2.5:1.5b",
                            allow_custom_value=True,
                        )
                except Exception as e:
                    raise f"No backend supported {e}"
                
                hf_token = gr.Textbox(
                    label="HF Token (Optional)",
                    placeholder="hf_...",
                    type="password",
                )

    gr.Markdown("---")

    with gr.Tabs():

        # Tab 1: Analysis 
        with gr.TabItem("🔬 Stability Analysis"):
            with gr.Row():
                n_runs = gr.Slider(minimum=2, maximum=5, value=3, step=1, label="Number of runs (N samples)")
                temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperature")

            analyze_btn = gr.Button("🔬 Analyze Prompt", variant="primary")
            
            # Using gr.HTML blocks mapped to the HTML UI layout
            stab_status_out = gr.HTML()
            stab_metrics_out = gr.HTML()
            
            gr.HTML('<div class="section-label">Outputs — Expand to read</div>')
            stab_outputs_out = gr.HTML()
            
            gr.HTML('<div class="section-label">Token confidence — First output</div>')
            stab_token_out = gr.HTML()
            
            _analysis_state = gr.State()

            analyze_btn.click(
                fn=run_analysis,
                inputs=[
                    prompt_input, input_text, task_input,
                    model_id, hf_token, n_runs, temperature
                ],
                outputs=[
                    stab_status_out, stab_metrics_out, stab_outputs_out,
                    stab_token_out, _analysis_state, gr.Textbox(visible=False)
                ],
            )

        # Tab 2: Optimization
        with gr.TabItem("⚡ Optimization"):
            gr.Markdown("""
Run Reflective Prompt Optimization inside a LangGraph control loop. The LLM generates its own variant prompts based on the current best and verbal feedback from prior rounds.
""")
            with gr.Row():
                expected_output = gr.Textbox(
                    label="Reference Output for Similarity Scoring",
                    placeholder="e.g., 'Positive' (Best for classification/extraction. Leave blank for creative tasks)",
                    lines=2,
                )

            with gr.Row():
                n_variants = gr.Slider(minimum=2, maximum=5, value=3, step=1, label="Variants per iteration")
                target_score= gr.Slider(minimum=0.5, maximum=0.97, value=0.70, step=0.01, label="Target score")
                max_iter = gr.Slider(minimum=2, maximum=10, value=3, step=1, label="Max graph iterations")

            use_judge = gr.Checkbox(label="Enable LLM-as-judge scoring (slower, more accurate)", value=False)
            optimize_btn = gr.Button("⚡ Optimize Prompt", variant="primary")

            # Mapped exactly to the HTML layout structure
            opt_status_out = gr.HTML()
            
            gr.HTML('<div class="section-label">Score progress</div>')
            opt_metrics_out = gr.HTML()
            
            gr.HTML('<div class="section-label">Iteration timeline</div>')
            opt_timeline_out = gr.HTML()

            gr.HTML('<div class="section-label">Prompt comparison</div>')
            opt_prompt_out = gr.HTML()

            gr.HTML('<div class="section-label">AI reflection</div>')
            opt_feedback_out = gr.HTML()

            optimize_btn.click(
                fn=run_optimization,
                inputs=[
                    prompt_input, input_text, task_input, model_id, hf_token,
                    expected_output, n_variants, target_score, max_iter, use_judge
                ],
                outputs=[
                    opt_status_out, opt_metrics_out, opt_timeline_out, opt_prompt_out, opt_feedback_out
                ],
            )

        # Tab 3: Registry 
        with gr.TabItem("📚 Registry"):
            gr.Markdown("""
Query the registry for the best known prompt for a given task,
based on the **average historical score** across all evaluations.
""")
            with gr.Row():
                registry_task = gr.Dropdown(
                    label="Task to search",
                    choices=TASK_CATEGORIES,
                    value="summarize",
                    allow_custom_value=True,
                )
                registry_limit = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Evaluations to sample")

            registry_btn = gr.Button("📚 Query Registry", variant="secondary")
            registry_out = gr.Markdown()

            registry_btn.click(
                fn=query_best,
                inputs=[registry_task, registry_limit],
                outputs=[registry_out],
            )

    gr.Markdown("""
---
**Imprimer** · [GitHub](https://github.com/BalorLC3/Imprimer) · Karim luna
""")

if __name__ == "__main__":
    init_db()  
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    )