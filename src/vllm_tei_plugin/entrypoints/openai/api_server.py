# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from argparse import Namespace
from http import HTTPStatus
from typing import Optional

import uvloop
import vllm.envs as envs
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import State
from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import load_chat_template
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.api_server import (
    ErrorResponse,
    base,
    build_app,
    build_async_engine_client,
    init_app_state,
    load_log_config,
    logger,
    maybe_register_tokenizer_info_endpoint,
    setup_server,
    validate_json_request,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.utils import cli_env_setup, load_aware_call, with_cancellation
from vllm.utils import FlexibleArgumentParser

from vllm_tei_plugin.entrypoints.openai.serving_tei_embed import TeiServingEmbed
from vllm_tei_plugin.entrypoints.openai.serving_tei_rerank import TeiServingRerank
from vllm_tei_plugin.protocol import EmbedRequest, RerankRequest

tei_router = APIRouter(prefix="/tei")


def tei_embed(request: Request) -> Optional[TeiServingEmbed]:
    return request.app.state.tei_serving_embed


def tei_rerank(request: Request) -> Optional[TeiServingRerank]:
    return request.app.state.tei_serving_rerank


def extended_build_app(args: Namespace) -> FastAPI:
    app = build_app(args)
    app.include_router(tei_router)
    return app


async def extended_init_app_state(
    engine_client: EngineClient,
    vllm_config: VllmConfig,
    state: State,
    args: Namespace,
) -> None:
    await init_app_state(engine_client, vllm_config, state, args)
    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    model_config = vllm_config.model_config
    resolved_chat_template = load_chat_template(args.chat_template)

    enable_serving_reranking = (
        "classify" in model_config.supported_tasks
        and getattr(model_config.hf_config, "num_labels", 0) == 1
    )

    # Here is the difference
    state.tei_serving_embed = (
        TeiServingEmbed(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
            chat_template=resolved_chat_template,
            chat_template_content_format=args.chat_template_content_format,
        )
        if "embed" in model_config.supported_tasks
        else None
    )
    state.tei_serving_rerank = (
        TeiServingRerank(
            engine_client,
            model_config,
            state.openai_serving_models,
            request_logger=request_logger,
        )
        if ("embed" in model_config.supported_tasks or enable_serving_reranking)
        else None
    )


@tei_router.post(
    "/embed",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def embed(request: EmbedRequest, raw_request: Request):
    handler = tei_embed(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support TEI Embed API"
        )

    generator = await handler.embed(request, raw_request)

    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    return JSONResponse(content=generator)


@tei_router.post(
    "/rerank",
    dependencies=[Depends(validate_json_request)],
    responses={
        HTTPStatus.BAD_REQUEST.value: {"model": ErrorResponse},
        HTTPStatus.INTERNAL_SERVER_ERROR.value: {"model": ErrorResponse},
    },
)
@with_cancellation
@load_aware_call
async def rerank(request: RerankRequest, raw_request: Request):
    handler = tei_rerank(raw_request)
    if handler is None:
        return base(raw_request).create_error_response(
            message="The model does not support TEI Rerank API"
        )

    generator = await handler.rerank(request, raw_request)
    if isinstance(generator, ErrorResponse):
        return JSONResponse(content=generator.model_dump(), status_code=generator.code)
    return JSONResponse(content=generator)


async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""
    listen_address, sock = setup_server(args)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)


async def run_server_worker(
    listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Run a single API server worker."""

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    server_index = client_config.get("client_index", 0) if client_config else 0

    # Load logging config for uvicorn if specified
    log_config = load_log_config(args.log_config_file)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config

    async with build_async_engine_client(args, client_config) as engine_client:
        maybe_register_tokenizer_info_endpoint(args)
        app = extended_build_app(args)

        vllm_config = await engine_client.get_vllm_config()
        await extended_init_app_state(engine_client, vllm_config, app.state, args)

        logger.info("Starting vLLM API server %d on %s", server_index, listen_address)
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # NOTE: When the 'disable_uvicorn_access_log' value is True,
            # no access log will be output.
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    try:
        await shutdown_task
    finally:
        sock.close()


if __name__ == "__main__":
    # NOTE(simon):
    # This section should be in sync with vllm/entrypoints/cli/main.py for CLI
    # entrypoints.
    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
