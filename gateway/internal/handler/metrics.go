package handler

import (
	"net/http"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type taskMetrics struct {
	reachabilitySum   float64
	reachabilityCount int64
	judgeScoreSum     float64
	judgeScoreCount   int64
}

var (
	// Mutex to protect access to taskMetricsByTask map
	metricsMu         sync.Mutex
	taskMetricsByTask = map[string]*taskMetrics{}
	// Total number of prompt evaluations per task, including both judge and non-judge evaluations
	evaluationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "imprimer_evaluations_total",
			Help: "Total number of prompt evaluations per task.",
		}, []string{"task"},
	)
	// Average reachability-like score per task, averaged across all evaluations (judge and non-judge)
	avgReachability = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "imprimer_avg_reachability",
			Help: "Average reachability-like score per task.",
		}, []string{"task"},
	)
	// Average judge score per task, averaged across only judge-enabled evaluations
	avgJudgeScore = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "imprimer_avg_judge_score",
			Help: "Average judge score per task for judge-enabled evaluations.",
		}, []string{"task"},
	)
	// Latest optimization improvement per task, recorded after each optimization completes
	optimizationImprovement = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "imprimer_optimization_improvement",
			Help: "Latest optimization improvement per task.",
		}, []string{"task"},
	)
)

func MetricsHandler() http.Handler {
	return promhttp.Handler()
}

func RecordEvaluationMetrics(task string, reachability float64, judgeScore float64, judgeUsed bool) {
	evaluationsTotal.WithLabelValues(task).Inc()

	metricsMu.Lock()
	defer metricsMu.Unlock()

	stats, ok := taskMetricsByTask[task]
	if !ok {
		stats = &taskMetrics{}
		taskMetricsByTask[task] = stats
	}
	// Reachability represents
	stats.reachabilitySum += reachability
	stats.reachabilityCount++
	avgReachability.WithLabelValues(task).Set(stats.reachabilitySum / float64(stats.reachabilityCount))

	if judgeUsed {
		stats.judgeScoreSum += judgeScore
		stats.judgeScoreCount++
		avgJudgeScore.WithLabelValues(task).Set(stats.judgeScoreSum / float64(stats.judgeScoreCount))
	}
}

func RecordOptimizationImprovement(task string, improvement float64) {
	optimizationImprovement.WithLabelValues(task).Set(improvement)
}
