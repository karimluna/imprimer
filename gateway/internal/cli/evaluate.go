package cli

import (
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"
)

type evaluatePayload struct {
	Task     string `json:"task"`
	Input    string `json:"input"`
	VariantA string `json:"variant_a"`
	VariantB string `json:"variant_b"`
	Backend  string `json:"backend"`
}

type evaluateResult struct {
	TraceID  string  `json:"trace_id"`
	Winner   string  `json:"winner"`
	OutputA  string  `json:"output_a"`
	OutputB  string  `json:"output_b"`
	LatencyA float32 `json:"latency_a_ms"`
	LatencyB float32 `json:"latency_b_ms"`
	ScoreA   float32 `json:"score_a"`
	ScoreB   float32 `json:"score_b"`
}

var evaluateCmd = &cobra.Command{
	Use:   "evaluate",
	Short: "Run two prompt variants and compare their reachability scores",
	Example: `  imprimer evaluate \
    --task summarize \
    --input "Minsky argued intelligence emerges from many small agents" \
    --a "Summarize this in one sentence: {input}" \
    --b "You are an expert writer. Give a precise one sentence summary of: {input}"`,
	RunE: func(cmd *cobra.Command, args []string) error {
		task, _ := cmd.Flags().GetString("task")
		input, _ := cmd.Flags().GetString("input")
		variantA, _ := cmd.Flags().GetString("a")
		variantB, _ := cmd.Flags().GetString("b")
		backend, _ := cmd.Flags().GetString("backend")

		if task == "" || input == "" || variantA == "" || variantB == "" {
			return fmt.Errorf("--task, --input, --a, and --b are all required")
		}

		c := NewImprimerClient(gatewayURL, apiKey)

		var result evaluateResult
		if err := c.post("/prompt", evaluatePayload{
			Task:     task,
			Input:    input,
			VariantA: variantA,
			VariantB: variantB,
			Backend:  backend,
		}, &result); err != nil {
			return err
		}

		if outputJSON {
			raw, _ := json.MarshalIndent(result, "", "  ")
			fmt.Println(string(raw))
			return nil
		}

		// Formatted output
		fmt.Printf("\n  Trace ID  %s\n", result.TraceID)
		fmt.Printf("  Winner    variant %s\n\n", result.Winner)
		fmt.Printf("  %-12s  score=%.3f  latency=%.0fms\n", "Variant A", result.ScoreA, result.LatencyA)
		fmt.Printf("  %s\n\n", result.OutputA)
		fmt.Printf("  %-12s  score=%.3f  latency=%.0fms\n", "Variant B", result.ScoreB, result.LatencyB)
		fmt.Printf("  %s\n\n", result.OutputB)

		return nil
	},
}

func init() {
	evaluateCmd.Flags().String("task", "", "Task type (summarize, classify, extract)")
	evaluateCmd.Flags().String("input", "", "Input text to process")
	evaluateCmd.Flags().String("a", "", "First prompt template (use {input} as placeholder)")
	evaluateCmd.Flags().String("b", "", "Second prompt template (use {input} as placeholder)")
	evaluateCmd.Flags().String("backend", "ollama", "Model backend: ollama or openai")

	RootCmd.AddCommand(evaluateCmd)
}
