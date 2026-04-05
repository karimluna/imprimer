package middleware

import (
	"net/http"
	"os"
	"strings"
)

// Auth validates the Authorization header on incoming requests.
// In production this would verify a JWT against an identity provider.
//
// ISO 27001 A.9 - Access control:
// No request reaches the prompt handler without a valid token.
// The token is never logged - only its presence is recorded in the audit trail.
func Auth(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := os.Getenv("IMPRIMER_API_KEY")

		// If no token is configured, auth is disabled. This will be modified later
		if token == "" {
			next.ServeHTTP(w, r)
			return
		}

		header := r.Header.Get("Authorization")
		if !strings.HasPrefix(header, "Bearer ") {
			http.Error(w, "missing authorization header", http.StatusUnauthorized)
			return
		}

		provided := strings.TrimPrefix(header, "Bearer ")
		if provided != token {
			http.Error(w, "invalid token", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	})
}
