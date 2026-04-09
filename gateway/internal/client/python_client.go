package client

import (
	"context"

	gen "github.com/BalorLC3/Imprimer/gateway/gen"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// PythonClient wraps the generated gRPC stub
type PythonClient struct {
	conn *grpc.ClientConn
	stub gen.PromptEngineClient
}

func NewPythonClient(addr string) (*PythonClient, error) {
	// insecure.NewCredentials means no tls, which is fine for Docker
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &PythonClient{
		conn: conn,
		stub: gen.NewPromptEngineClient(conn),
	}, nil
}

// Call sends an EvaluateRequest to the Python engine and returns a response
func (c *PythonClient) Call(ctx context.Context, req *gen.EvaluateRequest) (*gen.EvaluateResponse, error) {
	return c.stub.EvaluatePrompt(ctx, req)
}

func (c *PythonClient) Close() error {
	return c.conn.Close()
}

func (c *PythonClient) Best(ctx context.Context, req *gen.BestRequest) (*gen.BestResponse, error) {
	return c.stub.BestVariant(ctx, req)
}

func (c *PythonClient) Optimize(ctx context.Context, req *gen.OptimizeRequest) (*gen.OptimizeResponse, error) {
	return c.stub.OptimizePrompt(ctx, req)
}
