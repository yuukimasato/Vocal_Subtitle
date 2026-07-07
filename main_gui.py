#!/usr/bin/env python3
"""vocal-subtitle GUI 入口

启动 Web 图形界面，在浏览器中打开人声分离+字幕生成工具。

Usage:
    python main_gui.py
    python main_gui.py --port 8080
    python main_gui.py --no-browser
"""

import click


@click.command()
@click.option("--host", default="127.0.0.1", help="绑定的主机地址")
@click.option("--port", default=7860, help="监听端口")
@click.option("--no-browser", is_flag=True, help="不自动打开浏览器")
def main(host: str, port: int, no_browser: bool):
    """启动 Web GUI 服务"""
    import threading
    import time
    import webbrowser

    import uvicorn

    from vocal_subtitle.webui.app import create_app

    if not no_browser:
        def _open_browser():
            time.sleep(1.0)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open_browser, daemon=True).start()

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
