import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True)
class _ServerSpec:
    name: str
    module_path: Path


async def _run_protocol_checks(spec: _ServerSpec, *, supports_tool_tasks: bool) -> None:
    import anyio
    import mcp.types as types
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    logging_messages: list[str] = []
    progress_messages: list[str] = []
    task_status_notifications: list[str] = []

    async def logging_callback(params: types.LoggingMessageNotificationParams) -> None:
        data = params.data
        if isinstance(data, dict) and "msg" in data:
            logging_messages.append(str(data["msg"]))
        else:
            logging_messages.append(str(data))

    async def message_handler(msg: Any) -> None:
        if not isinstance(msg, types.ServerNotification):
            return
        if isinstance(msg.root, types.TaskStatusNotification):
            task_status_notifications.append(str(msg.root.params.statusMessage))

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(spec.module_path)],
        cwd=str(spec.module_path.parent.parent.parent),
        env={},
    )

    checks_completed = False

    try:
        async with stdio_client(server) as (read_stream, write_stream):
            session = ClientSession(
                read_stream,
                write_stream,
                logging_callback=logging_callback,
                message_handler=message_handler,
            )

            async with session:
                await session.initialize()

                # -----------------------------------------------------------------
                # Foreground: notifications/progress should fire when we provide a
                # progress_callback (ClientSession will inject _meta.progressToken).
                # -----------------------------------------------------------------
                async def progress_callback(
                    progress: float,
                    total: float | None,
                    message: str | None,
                ) -> None:
                    if message:
                        progress_messages.append(message)

                call_result = await session.call_tool(
                    name="progress_demo",
                    arguments={"steps": 3, "delay_ms": 50},
                    progress_callback=progress_callback,
                )

                assert call_result.isError is False
                if supports_tool_tasks:
                    assert progress_messages or logging_messages

                # -----------------------------------------------------------------
                # Tool task augmentation differs by implementation:
                # - official mcp.server.fastmcp does NOT create background tasks for
                #   tools/call when params.task is provided (returns CallToolResult)
                # - third-party fastmcp DOES create a task (returns CreateTaskResult)
                # -----------------------------------------------------------------
                task_req = types.ClientRequest(
                    types.CallToolRequest(
                        params=types.CallToolRequestParams(
                            name="progress_demo",
                            arguments={"steps": 3, "delay_ms": 80},
                            task=types.TaskMetadata(ttl=30_000),
                        )
                    )
                )

                if not supports_tool_tasks:
                    sync_result = await session.send_request(task_req, types.CallToolResult)
                    assert sync_result.isError is False
                    checks_completed = True
                    return

                created = await session.send_request(task_req, types.CreateTaskResult)
                task_id = created.task.taskId

                seen_status_messages: list[str] = []
                for _ in range(200):
                    get_req = types.ClientRequest(
                        types.GetTaskRequest(params=types.GetTaskRequestParams(taskId=task_id))
                    )
                    task = await session.send_request(get_req, types.GetTaskResult)
                    if task.statusMessage:
                        seen_status_messages.append(task.statusMessage)

                    if task.status in ("completed", "failed", "cancelled"):
                        break

                    await __import__("asyncio").sleep(0.05)

                assert task.status in ("completed", "failed", "cancelled")

                # If the server emits notifications/tasks/status, at least one
                # should contain a step message.
                if seen_status_messages:
                    assert any("step" in m for m in seen_status_messages)
                if task_status_notifications:
                    assert any("step" in (m or "") for m in task_status_notifications)

                checks_completed = True
    except* anyio.BrokenResourceError:
        if not checks_completed:
            raise


@pytest.mark.asyncio
async def test_protocol_progress_official_mcp_fastmcp() -> None:
    spec = _ServerSpec(
        name="official",
        module_path=Path(__file__).parent.parent
        / "scripts"
        / "protocol_servers"
        / "official_mcp_fastmcp_server.py",
    )
    await _run_protocol_checks(spec, supports_tool_tasks=False)


@pytest.mark.asyncio
async def test_protocol_progress_third_party_fastmcp() -> None:
    spec = _ServerSpec(
        name="fastmcp",
        module_path=Path(__file__).parent.parent
        / "scripts"
        / "protocol_servers"
        / "third_party_fastmcp_server.py",
    )
    await _run_protocol_checks(spec, supports_tool_tasks=True)
