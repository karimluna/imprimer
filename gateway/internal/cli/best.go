package cli

import (
	"encoding/json"
	"fmt"
	"net/url"

	"github.com/spf13/cobra"
)

type bestResult struct {
	Task            string  `json:"task"`
	BestTemplate    string  `json:"best_template"`
	AvgReachability float32 `json:"avg_reachability"`
	AvgScore        float32 `json:"avg_score"`
	Evaluations     int32   `json:"evaluations"`
}

var bestCmd = &cobra.Command{
	Use:   "best",
	Short: "Query the registry for the best known prompt for a task",
	Long: `Returns the prompt template that achieved the highest average
reachability across all historical evaluations for the given task.

This is the feedback loop closing: the system learns over time which
prompts control the model most effectively for each task type.`,
	Example: `  imprimer best --task summarize
  imprimer best --task summarize --limit 20 --json`,
	RunE: func(cmd *cobra.Command, args []string) error {
		task, _ := cmd.Flags().GetString("task")
		limit, _ := cmd.Flags().GetInt("limit")

		if task == "" {
			return fmt.Errorf("--task is required")
		}

		path := fmt.Sprintf("/best?task=%s&limit=%d",
			url.QueryEscape(task), limit)

		c := NewImprimerClient(gatewayURL, apiKey)

		var result bestResult
		if err := c.get(path, &result); err != nil {
			return err
		}

		if outputJSON {
			raw, _ := json.MarshalIndent(result, "", "  ")
			fmt.Println(string(raw))
			return nil
		}

		fmt.Printf("\n  Task                %s\n", result.Task)
		fmt.Printf("  Evaluations         %d\n", result.Evaluations)
		fmt.Printf("  Avg reachability    %.4f\n", result.AvgReachability)
		fmt.Printf("  Avg score           %.4f\n\n", result.AvgScore)
		fmt.Printf("  Best prompt:\n  %s\n\n", result.BestTemplate)

		return nil
	},
}

func init() {
	bestCmd.Flags().String("task", "", "Task type to query")
	bestCmd.Flags().Int("limit", 10, "Number of recent evaluations to consider")

	RootCmd.AddCommand(bestCmd)
}
