package handler

import (
	"net/http"

	gen "github.com/BalorLC3/Imprimer/gateway/gen"
	"github.com/BalorLC3/Imprimer/gateway/internal/client"
	"github.com/BalorLC3/Imprimer/gateway/internal/httpx"
)

// Handles POST /optimize
type OptimizeHandler struct {
	engine *client.PythonClient
}

func NewOptimizeHandler(engine *client.PythonClient) *OptimizeHandler {
	return &OptimizeHandler{engine: engine}
}

// same pattern as to encode later
type optimizeRequest struct {
	Task               string  `json:"task"`
	BasePrompt         string  `json:"base_prompt"`
	InputExample       string  `json:"input_example"`
	ExpectedOutput     string  `json:"expected_output"`
	NTrials            int32   `json:"n_trials"`
	Backend            string  `json:"backend"` // e.g. "ollama", "openai", "groq", etc.
	UseJudge           bool    `json:"use_judge"`
	TargetReachability float32 `json:"target_reachability"`
	MaxIterations      int32   `json:"max_iterations"`
}

type optimizeResponse struct {
	BestPrompt           string  `json:"best_prompt"`
	BestScore            float32 `json:"best_score"`
	BestReachability     float32 `json:"best_reachability"`
	BaselineScore        float32 `json:"baseline_score"`
	BaselineReachability float32 `json:"baseline_reachability"`
	Improvement          float32 `json:"improvement"`
	TrialsRun            int32   `json:"trials_run"`
	IterationsCompleted  int32   `json:"iterations_completed"`
	TargetReached        bool    `json:"target_reached"`
}

func (h *OptimizeHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req optimizeRequest
	if !httpx.DecodeJSON(w, r, &req) {
		return
	}

	if req.Task == "" ||
		req.BasePrompt == "" ||
		req.InputExample == "" ||
		req.ExpectedOutput == "" ||
		req.NTrials <= 0 {
		httpx.WriteError(w, http.StatusBadRequest, "all fields must be filled and n_trials must be > 0")
		return
	}

	// Default backend to ollama if not specified
	if req.Backend == "" {
		req.Backend = "ollama"
	}

	grpcResp, err := h.engine.Optimize(r.Context(), &gen.OptimizeRequest{
		Task:               req.Task,
		BasePrompt:         req.BasePrompt,
		InputExample:       req.InputExample,
		ExpectedOutput:     req.ExpectedOutput,
		NTrials:            req.NTrials,
		Backend:            req.Backend,
		UseJudge:           req.UseJudge,
		TargetReachability: req.TargetReachability,
		MaxIterations:      req.MaxIterations,
	})
	if err != nil {
		http.Error(w, "engine error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// prometheus metrics for optimization improvement
	RecordOptimizationImprovement(req.Task, float64(grpcResp.Improvement))

	httpx.WriteJSON(w, http.StatusOK, optimizeResponse{
		BestPrompt:           grpcResp.BestPrompt,
		BestScore:            grpcResp.BestScore,
		BestReachability:     grpcResp.BestReachability,
		BaselineScore:        grpcResp.BaselineScore,
		BaselineReachability: grpcResp.BaselineReachability,
		Improvement:          grpcResp.Improvement,
		TrialsRun:            grpcResp.TrialsRun,
		IterationsCompleted:  grpcResp.IterationsCompleted,
		TargetReached:        grpcResp.TargetReached,
	})
}
