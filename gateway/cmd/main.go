package main

import (
	"log"
	"net/http"
	"os"

	"github.com/BalorLC3/Imprimer/gateway/internal/client"
	"github.com/BalorLC3/Imprimer/gateway/internal/handler"
	"github.com/BalorLC3/Imprimer/gateway/internal/middleware"
)

// Imprimer gateway; entry point for all external requests
// In Minsky's Society of Mind framing, this is the receptor layer:
// it receives stimuli from the outside world and routes them inward.
// It knows nothing about how prompts work or how models think.
// It only knows how to receive, authenticate, audit, and forward.

func main() {
	// Read engine address from environment.
	engineAddr := os.Getenv("ENGINE_ADDR")
	if engineAddr == "" {
		engineAddr = "localhost:50051"
	}

	engineClient, err := client.NewPythonClient(engineAddr)
	if err != nil {
		log.Fatalf("failed to connect to Python engine at %s: %v", engineAddr, err)
	}
	defer engineClient.Close()

	promptHandler := handler.NewPromptHandler(engineClient)
	bestHandler := handler.NewBestHandler(engineClient)
	optimizeHandler := handler.NewOptimizeHandler(engineClient)

	mux := http.NewServeMux()
	mux.Handle("/prompt", middleware.Auth(middleware.Audit(promptHandler)))
	mux.Handle("/best", middleware.Auth(middleware.Audit(bestHandler)))
	mux.Handle("/optimize", middleware.Auth(middleware.Audit(optimizeHandler)))

	log.Printf("Imprimer gateway listening on :8080 (engine at %s)", engineAddr)
	log.Fatal(http.ListenAndServe(":8080", mux))
}
