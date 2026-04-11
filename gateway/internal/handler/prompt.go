package handler

import (
	"encoding/json"
	"math"
	"net/http"

	gen "github.com/BalorLC3/Imprimer/gateway/gen"
	"github.com/BalorLC3/Imprimer/gateway/internal/client"
	"github.com/BalorLC3/Imprimer/gateway/internal/httpx"
	"github.com/BalorLC3/Imprimer/gateway/internal/middleware"
)

// promptRequest is what called sends in the HTTP body
type promptRequest struct {
	Task     string `json:"task"`
	Input    string `json:"input"`
	VariantA string `json:"variant_a"`
	VariantB string `json:"variant_b"`
	Backend  string `json:"backend"`
	UseJudge bool   `json:"use_judge"` // Flag is only used for judge-enabled evaluations
}

// promptResponse is what Imprimer returns, winner and evidence
type promptResponse struct {
	TraceID  string  `json:"trace_id"`
	Winner   string  `json:"winner"`
	OutputA  string  `json:"output_a"`
	OutputB  string  `json:"output_b"`
	LatencyA float32 `json:"latency_a_ms"`
	LatencyB float32 `json:"latency_b_ms"`
	ScoreA   float32 `json:"score_a"`
	ScoreB   float32 `json:"score_b"`
}

// PromptHandler handles POST /prompt
type PromptHandler struct {
	engine *client.PythonClient
}

func NewPromptHandler(engine *client.PythonClient) *PromptHandler {
	return &PromptHandler{engine: engine}
}

func (h *PromptHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req promptRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	// Validate all four fields are required
	if req.Task == "" || req.Input == "" || req.VariantA == "" || req.VariantB == "" {
		http.Error(w, "task, input, variant_a, and variant_b are all required", http.StatusBadRequest)
		return
	}
	// Pull trace ID the Audit middleware already placed in the context
	traceID := middleware.TraceIDFrom(r.Context())

	if req.Backend == "" {
		req.Backend = "ollama"
	}

	grpcReq := &gen.EvaluateRequest{
		TraceId:  traceID,
		Task:     req.Task,
		Input:    req.Input,
		VariantA: req.VariantA,
		VariantB: req.VariantB,
		Backend:  req.Backend,
		UseJudge: req.UseJudge,
	}

	grpcResp, err := h.engine.Call(r.Context(), grpcReq)
	if err != nil {
		http.Error(w, "engine error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Record evaluation metrics for this task.
	winnerScore := math.Max(float64(grpcResp.ScoreA), float64(grpcResp.ScoreB))
	RecordEvaluationMetrics(req.Task, winnerScore, winnerScore, req.UseJudge)

	resp := promptResponse{
		TraceID:  grpcResp.TraceId,
		Winner:   grpcResp.Winner,
		OutputA:  grpcResp.OutputA,
		OutputB:  grpcResp.OutputB,
		LatencyA: grpcResp.LatencyA,
		LatencyB: grpcResp.LatencyB,
		ScoreA:   grpcResp.ScoreA,
		ScoreB:   grpcResp.ScoreB,
	}
	httpx.WriteJSON(w, http.StatusOK, resp)
}
