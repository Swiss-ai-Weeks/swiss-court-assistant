import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Swiss Court Assistant API (and the built UI, if any).")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--reload", action="store_true", help="restart on code changes (watches src/ only)")
    args = ap.parse_args()
    uvicorn.run("swiss_court_assistant.server.app:app", host=args.host, port=args.port,
                reload=args.reload, reload_dirs=["src/swiss_court_assistant"] if args.reload else None)


if __name__ == "__main__":
    main()
