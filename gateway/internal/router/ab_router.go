package router

// ABRouter decides how to split traffic between
// prompt variants. For a while, we will use 50/50
// both variants run on every request.
// Later we will implement an epsilon-greedy exploration:
// exploit the known winner most of the time, explore
// the challenger sometimes.

type Strategy string

const (
	// runs both variants on every request and scores
	// both. Maximum information maximum cost.
	StrategyFull Strategy = "full"

	// StrategyExploit routes most traffic to the known
	// winner and a small fraction to challenger.
	StrategyExploit Strategy = "exploit"
)

func Route(task string) Strategy {
	return StrategyFull
}
