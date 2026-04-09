"""
Imprimer engine, gRPC entrypoint.

This is the cognitive layer
"""
import grpc
from concurrent import futures

import imprimer_pb2
import imprimer_pb2_grpc

from core.chains.prompt_chain import run_variant, ModelBackend
from core.evaluator.scorer import score
from core.registry.prompt_store import init_db, save, EvalRecord, best_variant_for_task
from core.optimize.bayesian_search import optimize

from observability.tracer import log_eval, EvalTrace, reachability_gap_report
from security.injection_guard import scan_request, InjectionDetected
from utils.create_logger import get_logger

logger = get_logger(__name__)


class PromptEngineServicer(imprimer_pb2_grpc.PromptEngineServicer):

    def EvaluatePrompt(self, request, context):
        logger.info(
            f"trace={request.trace_id} "
            f"task={request.task} "
            f"backend={request.backend}"
        )

        # Security gate, scan before any LLM interaction
        try:
            scan_request(
                trace_id=request.trace_id,
                input_text=request.input,
                variant_a=request.variant_a,
                variant_b=request.variant_b,
            )
        except InjectionDetected as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return imprimer_pb2.EvaluateResponse()

        # Resolve backend
        backend_str = request.backend.lower() if request.backend else "ollama"
        try:
            backend = ModelBackend(backend_str)
        except ValueError:
            logger.warning(
                f"trace={request.trace_id} "
                f"unknown backend '{backend_str}', falling back to ollama"
            )
            backend = ModelBackend.OLLAMA

        # Activate both candidate minds
        result_a = run_variant(
            template=request.variant_a,
            input_text=request.input,
            task=request.task,
            backend=backend,
        )
        result_b = run_variant(
            template=request.variant_b,
            input_text=request.input,
            task=request.task,
            backend=backend,
        )

        score_a = score(result_a)
        score_b = score(result_b)
        winner = "a" if score_a.combined >= score_b.combined else "b"

        # reachability gap report
        gap_report = reachability_gap_report(
            trace_id=request.trace_id,
            reachability_a=score_a.reachability,
            reachability_b=score_b.reachability,
            winner=winner,
        )

        # persist to registry
        save(EvalRecord(
            trace_id=request.trace_id,
            task=request.task,
            backend=backend_str,
            variant_a=request.variant_a,
            variant_b=request.variant_b,
            winner=winner,
            reachability_a=score_a.reachability,
            reachability_b=score_b.reachability,
            score_a=score_a.combined,
            score_b=score_b.combined,
            latency_a_ms=result_a.latency_ms,
            latency_b_ms=result_b.latency_ms,
            gap_report=gap_report,
        ))

        # Structured audit trace
        log_eval(EvalTrace(
            trace_id=request.trace_id,
            task=request.task,
            backend=backend_str,
            winner=winner,
            reachability_a=score_a.reachability,
            reachability_b=score_b.reachability,
            score_a=score_a.combined,
            score_b=score_b.combined,
            latency_a_ms=result_a.latency_ms,
            latency_b_ms=result_b.latency_ms,
            variant_a=request.variant_a,
            variant_b=request.variant_b,
        ))

        logger.info(
            f"trace={request.trace_id} "
            f"winner={winner} "
            f"reachability_a={score_a.reachability} "
            f"reachability_b={score_b.reachability} "
            f"gap={gap_report[:60]}..."
        )

        return imprimer_pb2.EvaluateResponse(
            trace_id=request.trace_id,
            winner=winner,
            output_a=result_a.text,
            output_b=result_b.text,
            latency_a=result_a.latency_ms,
            latency_b=result_b.latency_ms,
            score_a=score_a.combined,
            score_b=score_b.combined,
        )
    
    def BestVariant(self, request, context):
        logger.info(f"task={request.task} limit={request.limit}")
        
        limit = request.limit if request.limit > 0 else 10
        result = best_variant_for_task(request.task, limit=limit)

        if not result:
            return imprimer_pb2.BestResponse(
                task=request.task,
                found=False
            )
        
        return imprimer_pb2.BestResponse(
            task=result["task"],
            best_template=result["best_template"],
            avg_reachability=result["avg_reachability"],
            avg_score=result["avg_score"],
            evaluations=result["evaluations_sampled"],
            found=True,
        )

    def OptimizePrompt(self, request, context): 
        logger.info(
            f"trace-optimize "
            f"task={request.task}"
            f"n_trials={request.n_trials}"
        )

        backend_str = request.backend.lower() if request.backend else "ollama"
        
        try:
            backend = ModelBackend(backend_str)
        except ValueError:
            logger.warning(
                f"trace={request.trace_id} "
                f"unknown backend '{backend_str}', falling back to ollama"
            )
            backend = ModelBackend.OLLAMA

        n_trials = request.n_trials if request.n_trials > 0 else 20

        result = optimize(
            task=request.task,
            base_prompt=request.base_prompt,
            input_example=request.input_example,
            expected_output=request.expected_output,
            n_trials=n_trials,
            backend=backend,
            storage=None,
            study_name=None
        )

        return imprimer_pb2.OptimizeResponse(
            best_prompt=result.best_prompt,
            best_score=result.best_score,
            best_reachability=result.best_reachability,
            baseline_score=result.baseline_score,
            improvement=result.improvement,
            trials_run=result.trials_run,
        )
            


def serve():
    init_db()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    imprimer_pb2_grpc.add_PromptEngineServicer_to_server(
        PromptEngineServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("Imprimer engine listening on :50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()