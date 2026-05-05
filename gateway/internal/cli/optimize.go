package cli

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/karimluna/imprimer/gateway/internal/ui"
	"github.com/spf13/cobra"
)

type optimizePayload struct {
	Task               string  `json:"task"`
	BasePrompt         string  `json:"base_prompt"`
	InputExample       string  `json:"input_example"`
	ExpectedOutput     string  `json:"expected_output"`
	NVariants          int32   `json:"n_variants"`
	Backend            string  `json:"backend"`
	TargetReachability float32 `json:"target_reachability"`
	MaxIterations      int32   `json:"max_iterations"`
}

type optimizeResult struct {
	BestPrompt           string  `json:"best_prompt"`
	BestScore            float32 `json:"best_score"`
	BestReachability     float32 `json:"best_reachability"`
	BaselineScore        float32 `json:"baseline_score"`
	BaselineReachability float32 `json:"baseline_reachability"`
	Improvement          float32 `json:"improvement"`
	IterationsCompleted  int32   `json:"iterations_completed"`
	TargetReached        bool    `json:"target_reached"`
	Feedback             string  `json:"feedback"`
	GRPOGroupMean        float32 `json:"grpo_group_mean"`
}

var optimizeCmd = &cobra.Command{
	Use:   "optimize",
	Short: "Run Reflective Prompt Evolution to find the best prompt",
	Long: `Runs LLM-generated variant search inside a LangGraph control loop.
Each iteration generates N variants, scores them with SSC and reachability,
and feeds verbal reflection back into the next cycle.`,
	Example: `  imprimer optimize \
    --task summarize \
    --prompt "Summarize this in one sentence: {input}" \
    --input "Minsky argued intelligence emerges from many small agents" \
    --variants 4 \
    --max-iterations 3`,
	RunE: func(cmd *cobra.Command, args []string) error {
		task, _ := cmd.Flags().GetString("task")
		prompt, _ := cmd.Flags().GetString("prompt")
		input, _ := cmd.Flags().GetString("input")
		expected, _ := cmd.Flags().GetString("expected")
		variants, _ := cmd.Flags().GetInt32("variants")
		backend, _ := cmd.Flags().GetString("backend")
		targetReach, _ := cmd.Flags().GetFloat32("target-reachability")
		maxIterations, _ := cmd.Flags().GetInt32("max-iterations")

		if task == "" || prompt == "" {
			return fmt.Errorf("--task and --prompt are required")
		}
		if variants <= 0 {
			return fmt.Errorf("--variants must be greater than 0")
		}

		c := NewImprimerClient(gatewayURL, apiKey)

		// start spinner
		if !outputJSON {
			fmt.Print("\n🧠 Imprimer is thinking... (running graph iterations)")
		}

		done := make(chan bool)

		// Run the dots in the background
		if !outputJSON {
			go func() {
				for {
					select {
					case <-done:
						return
					default:
						// Import "time" at the top of your file to use time.Sleep
						fmt.Print(".")
						time.Sleep(1 * time.Second)
					}
				}
			}()
		}

		var result optimizeResult

		// Blocking call
		err := c.post("/optimize", optimizePayload{
			Task:               task,
			BasePrompt:         prompt,
			InputExample:       input,
			ExpectedOutput:     expected,
			NVariants:          variants,
			Backend:            backend,
			TargetReachability: targetReach,
			MaxIterations:      maxIterations,
		}, &result)

		// stop spinner
		if !outputJSON {
			done <- true
			fmt.Println(" Done!") // Moves to the next line before printing results
		}

		if err != nil {
			return err
		}

		if outputJSON {
			raw, _ := json.MarshalIndent(result, "", "  ")
			fmt.Println(string(raw))
			return nil
		}

		fmt.Println(ui.Banner())

		// Run summary
		status := "iteration cap reached"
		if result.TargetReached {
			status = "target reached ✓"
		}

		sign := "+"
		if result.Improvement < 0 {
			sign = ""
		}

		summary := strings.Join([]string{
			ui.Metric("Task", task),
			ui.Metric("Backend", backend),
			ui.Metric("Variants / iter", variants),
			ui.Metric("Iterations", fmt.Sprintf("%d / %d", result.IterationsCompleted, maxIterations)),
			ui.Metric("Status", status),
		}, "\n")

		fmt.Println(ui.Panel("Optimization Run", summary))

		// Score comparison table
		baseBar := ui.ScoreBar(result.BaselineScore, 20)
		bestBar := ui.ScoreBar(result.BestScore, 20)

		rows := [][2]string{
			{
				fmt.Sprintf("Score        %.4f", result.BaselineScore),
				fmt.Sprintf("Score        %.4f", result.BestScore),
			},
			{
				fmt.Sprintf("Reachability %.4f", result.BaselineReachability),
				fmt.Sprintf("Reachability %.4f", result.BestReachability),
			},
			{baseBar, bestBar},
			{
				"Improvement  —",
				fmt.Sprintf("Improvement  %s%.4f", sign, result.Improvement),
			},
		}

		if result.GRPOGroupMean > 0 {
			grpoBar := ui.ScoreBar(result.GRPOGroupMean, 20)
			rows = append(rows,
				[2]string{
					"GRPO mean    —",
					fmt.Sprintf("GRPO mean    %.4f", result.GRPOGroupMean),
				},
				[2]string{"—", grpoBar},
			)
		}

		fmt.Println(ui.Panel("Score Comparison", ui.Table(
			[2]string{"Baseline", "Best Found"},
			rows,
		)))

		// Prompt comparison
		fmt.Println(ui.Panel("Prompts", ui.Table(
			[2]string{"Original", "Optimized"},
			[][2]string{
				{
					ui.Prompt(truncate(prompt, 120)),
					ui.Prompt(truncate(result.BestPrompt, 120)),
				},
			},
		)))

		// AI reflection
		if result.Feedback != "" {
			fmt.Println(ui.Panel("AI Reflection", result.Feedback))
		}

		return nil
	},
}

func init() {
	optimizeCmd.Flags().String("task", "", "Task type (summarize, classify, extract)")
	optimizeCmd.Flags().String("prompt", "", "Base prompt template to optimize")
	optimizeCmd.Flags().String("input", "", "Example input (optional)")
	optimizeCmd.Flags().String("expected", "", "Expected output for similarity scoring (optional)")
	optimizeCmd.Flags().Int32("variants", 4, "Variants per iteration")
	optimizeCmd.Flags().String("backend", "ollama", "ollama or openai")
	optimizeCmd.Flags().Float32("target-reachability", 0.80,
		"Stop when reachability >= this value (paper ceiling: 0.97)")
	optimizeCmd.Flags().Int32("max-iterations", 3,
		"Max graph cycles (total LLM calls ≈ variants × iterations)")

	RootCmd.AddCommand(optimizeCmd)
}
