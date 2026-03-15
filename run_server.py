#!/usr/bin/env python3
"""
运行脚本
Run Script

启动FastAPI服务
"""

import uvicorn
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="启动LLaMA2 LoRA推理服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--workers", type=int, default=1, help="工作进程数")
    parser.add_argument("--reload", action="store_true", help="是否启用热重载")
    parser.add_argument("--log-level", type=str, default="info", help="日志级别")

    args = parser.parse_args()

    logger.info(f"Starting server at {args.host}:{args.port}")

    uvicorn.run(
        "backend.api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
