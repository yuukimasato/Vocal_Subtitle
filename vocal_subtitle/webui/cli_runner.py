"""GUI CLI 入口 — vocal-subtitle-gui 命令"""

import click


@click.command()
@click.option("--host", default="127.0.0.1", help="绑定的主机地址")
@click.option("--port", default=8613, help="监听端口")
@click.option(
    "--no-browser",
    is_flag=True,
    help="不自动打开浏览器",
)
@click.option(
    "--reload",
    is_flag=True,
    help="开发模式：自动重载（代码变更时）",
)
def main(host: str, port: int, no_browser: bool, reload: bool):
    """启动 Vocal Subtitle Web GUI

    在浏览器中打开图形界面，提供文件上传、Pipeline 进度
    可视化、字幕预览编辑和格式导出功能。
    """
    import uvicorn

    if not no_browser:
        import webbrowser
        import threading

        def _open_browser():
            import time
            time.sleep(1.0)  # 等待服务器启动
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open_browser, daemon=True).start()

    # 启动方式
    if reload:
        uvicorn.run(
            "vocal_subtitle.webui.app:create_app",
            host=host,
            port=port,
            factory=True,
            reload=True,
            reload_dirs=[str(__import__("pathlib").Path(__file__).parent.parent)],
        )
    else:
        from .app import create_app
        app = create_app()
        uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
