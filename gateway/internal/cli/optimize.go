package cli

import (
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"
)

type optimizePayload struct {
	Task           string `json:"task"`
	BasePrompt     string `json:"base_prompt"`
	InputExample   string `json:"input_example"`
	ExpectedOutput string `json:"expected_output"`
	NTrials        int32  `json:"n_trials"`
	Backend        string `json:"backend"`
}

type optimizeResult struct {
	BestPrompt           string  `json:"best_prompt"`
	BestScore            float32 `json:"best_score"`
	BestReachability     float32 `json:"best_reachability"`
	BaselineScore        float32 `json:"baseline_score"`
	BaselineReachability float32 `json:"baseline_reachability"`
	Improvement          float32 `json:"improvement"`
	TrialsRun            int32   `json:"trials_run"`
}

var optimizeCmd = &cobra.Command{
	Use:   "optimize",
	Short: "Run Bayesian optimization (Optuna TPE) to find the best prompt",
	Long: `Searches over prompt mutations using Optuna's Tree-structured
Parzen Estimator to maximize reachability + similarity to expected output.

Each trial costs one LLM inference call. With n=20 trials, the optimizer
typically converges within 8-10 trials after the initial random exploration.`,
	Example: `  imprimer optimize \
    --task summarize \
    --prompt "Summarize this in one sentence: {input}" \
    --input "Minsky argued intelligence emerges from many small agents" \
    --expected "Minsky argued intelligence is an emergent property of simple agents." \
    --trials 20`,
	RunE: func(cmd *cobra.Command, args []string) error {
		task, _ := cmd.Flags().GetString("task")
		prompt, _ := cmd.Flags().GetString("prompt")
		input, _ := cmd.Flags().GetString("input")
		expected, _ := cmd.Flags().GetString("expected")
		trials, _ := cmd.Flags().GetInt32("trials")
		backend, _ := cmd.Flags().GetString("backend")

		if task == "" || prompt == "" || input == "" || expected == "" {
			return fmt.Errorf("--task, --prompt, --input, and --expected are all required")
		}
		if trials <= 0 {
			return fmt.Errorf("--trials must be greater than 0")
		}

		fmt.Printf("\n  Running %d optimization trials for task '%s'...\n", trials, task)
		fmt.Printf("  Base prompt: %s\n\n", prompt)

		c := NewImprimerClient(gatewayURL, apiKey)

		var result optimizeResult
		if err := c.post("/optimize", optimizePayload{
			Task:           task,
			BasePrompt:     prompt,
			InputExample:   input,
			ExpectedOutput: expected,
			NTrials:        trials,
			Backend:        backend,
		}, &result); err != nil {
			return err
		}

		if outputJSON {
			raw, _ := json.MarshalIndent(result, "", "  ")
			fmt.Println(string(raw))
			return nil
		}

		// Formatted output
		sign := "+"
		if result.Improvement < 0 {
			sign = ""
		}
		fmt.Printf("  Trials run          %d\n", result.TrialsRun)
		fmt.Printf("  Baseline score      %.4f  (reachability %.4f)\n",
			result.BaselineScore, result.BaselineReachability)
		fmt.Printf("  Best score          %.4f  (reachability %.4f)\n",
			result.BestScore, result.BestReachability)
		fmt.Printf("  Improvement         %s%.4f\n\n", sign, result.Improvement)
		fmt.Printf("  Best prompt:\n  %s\n\n", result.BestPrompt)

		return nil
	},
}

func init() {
	optimizeCmd.Flags().String("task", "", "Task type (summarize, classify, extract)")
	optimizeCmd.Flags().String("prompt", "", "Base prompt template to optimize")
	optimizeCmd.Flags().String("input", "", "Example input for evaluation")
	optimizeCmd.Flags().String("expected", "", "Expected output for similarity scoring")
	optimizeCmd.Flags().Int32("trials", 20, "Number of optimization trials")
	optimizeCmd.Flags().String("backend", "ollama", "Model backend: ollama or openai")

	RootCmd.AddCommand(optimizeCmd)
}
