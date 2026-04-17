package handler

import (
	"net/http"

	gen "github.com/BalorLC3/Imprimer/gateway/gen"
	"github.com/BalorLC3/Imprimer/gateway/internal/client"
	"github.com/BalorLC3/Imprimer/gateway/internal/httpx"
	"github.com/BalorLC3/Imprimer/gateway/internal/middleware"
)

type analyzeRequest struct {
	Prompt      string  `json:"prompt"`
	Input       string  `json:"input"`
	Task        string  `json:"task"`
	Backend     string  `json:"backend"`
	NRuns       int32   `json:"n_runs"`
	Temperature float32 `json:"temperature"`
}

type tokenConfidence struct {
	Token     string  `json:"token"`
	Logprob   float32 `json:"logprob"`
	Certainty float32 `json:"certainty"`
}

type analyzeResponse struct {
	TraceID         string            `json:"trace_id"`
	Outputs         []string          `json:"outputs"`
	AvgReachability float32           `json:"avg_reachability"`
	Variance        float32           `json:"variance"`
	AvgSimilarity   float32           `json:"avg_similarity"`
	StabilityScore  float32           `json:"stability_score"`
	TokenConfidence []tokenConfidence `json:"token_confidence"`
}

type AnalyzeHandler struct {
	engine *client.PythonClient
}

func NewAnalyzeHandler(engine *client.PythonClient) *AnalyzeHandler {
	return &AnalyzeHandler{engine: engine}
}

func (h *AnalyzeHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req analyzeRequest
	if !httpx.DecodeJSON(w, r, &req) {
		return
	}

	if req.Prompt == "" || req.Task == "" {
		httpx.WriteError(w, http.StatusBadRequest, "prompt and task are required")
		return
	}

	if req.NRuns <= 0 {
		req.NRuns = 5
	}
	if req.Temperature <= 0 {
		req.Temperature = 0.7
	}
	if req.Backend == "" {
		req.Backend = "ollama"
	}

	traceID := middleware.TraceIDFrom(r.Context())

	grpcResp, err := h.engine.Analyze(r.Context(), &gen.StabilityRequest{
		TraceId:     traceID,
		Prompt:      req.Prompt,
		Input:       req.Input,
		Task:        req.Task,
		Backend:     req.Backend,
		NRuns:       req.NRuns,
		Temperature: req.Temperature,
	})
	if err != nil {
		http.Error(w, "engine error: "+err.Error(), http.StatusInternalServerError)
		return
	}

	tcs := make([]tokenConfidence, len(grpcResp.TokenConfidence))
	for i, tc := range grpcResp.TokenConfidence {
		tcs[i] = tokenConfidence{
			Token:     tc.Token,
			Logprob:   tc.Logprob,
			Certainty: tc.Certainty,
		}
	}

	httpx.WriteJSON(w, http.StatusOK, analyzeResponse{
		TraceID:         grpcResp.TraceId,
		Outputs:         grpcResp.Outputs,
		AvgReachability: grpcResp.AvgReachability,
		Variance:        grpcResp.Variance,
		AvgSimilarity:   grpcResp.AvgSimilarity,
		StabilityScore:  grpcResp.StabilityScore,
		TokenConfidence: tcs,
	})
}
