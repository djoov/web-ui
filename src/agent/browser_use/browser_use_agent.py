from __future__ import annotations

import asyncio
import logging
import os
import json # Impor pustaka json

from browser_use.agent.gif import create_history_gif
from browser_use.agent.service import Agent, AgentHookFunc
from browser_use.agent.views import (
    ActionResult,
    AgentHistory,
    AgentHistoryList,
    AgentStepInfo,
    ToolCallingMethod,
    AgentOutput 
)
from browser_use.browser.views import BrowserStateHistory
from browser_use.utils import time_execution_async
from dotenv import load_dotenv
from browser_use.agent.message_manager.utils import is_model_without_tool_support

# Impor modul DSPy yang sudah kita buat
from .dspy_program import BrowserAgentModule

load_dotenv()
logger = logging.getLogger(__name__)

SKIP_LLM_API_KEY_VERIFICATION = (
        os.environ.get("SKIP_LLM_API_KEY_VERIFICATION", "false").lower()[0] in "ty1"
)


class BrowserUseAgent(Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Inisialisasi program DSPy
        self.dspy_program = BrowserAgentModule()

    def _set_tool_calling_method(self) -> ToolCallingMethod | None:
        # Karena kita menggunakan DSPy, metode tool calling tradisional menjadi kurang relevan.
        # Kita nonaktifkan agar tidak mengganggu.
        return None
        
    # --- PERBAIKAN UTAMA DI SINI ---
    # Kita TIDAK lagi meng-override metode step, karena logika utamanya sudah ada di kelas Agent (induk).
    # Sebagai gantinya, kita akan meng-override metode _get_model_output yang bertanggung jawab
    # untuk memanggil LLM.
    
    async def _get_model_output(self, messages: list, tools: list) -> AgentOutput:
        """
        Override metode ini untuk menggunakan DSPy sebagai ganti pemanggilan LLM langsung.
        """
        # Dapatkan status peramban terkini, yang tidak secara langsung dilewatkan ke metode ini.
        browser_state = await self.browser_context.get_state(config=self.settings.browser_state_config)
        main_content = await self.get_main_content(browser_state)
        history_str = self.state.history.to_prompt()

        # 1. Panggil program DSPy kita
        prediction = self.dspy_program(
            task=self.task,
            url=browser_state.url,
            title=browser_state.title,
            elements=[el.to_dict() for el in browser_state.interacted_element],
            content=main_content,
            history=history_str
        )

        # 2. Proses output dari DSPy
        action_json_str = prediction.action
        logger.info(f"🧠 DSPy Reasoning: {prediction.reasoning}")
        logger.info(f"🤖 DSPy Predicted Action: {action_json_str}")

        try:
            # Coba parse JSON yang dihasilkan oleh DSPy
            if action_json_str.strip().startswith("```json"):
                action_json_str = action_json_str.strip()[7:-3].strip()
            elif action_json_str.strip().startswith("```"):
                 action_json_str = action_json_str.strip()[3:-3].strip()
            
            action_data = json.loads(action_json_str)
            
            # Buat objek AgentOutput yang sesuai
            model_output = AgentOutput(
                current_state=browser_state,
                action=[action_data]
            )
            return model_output

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"❌ Failed to parse JSON from DSPy: {e}")
            # Jika gagal, buat aksi 'done' dengan pesan error
            return AgentOutput(
                current_state=browser_state,
                action=[{"done": {"result": f"Error: Failed to parse JSON from LLM: {action_json_str}"}}]
            )


    @time_execution_async("--run (agent)")
    async def run(
            self, max_steps: int = 100, on_step_start: AgentHookFunc | None = None,
            on_step_end: AgentHookFunc | None = None
    ) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""

        loop = asyncio.get_event_loop()

        from browser_use.utils import SignalHandler

        signal_handler = SignalHandler(
            loop=loop,
            pause_callback=self.pause,
            resume_callback=self.resume,
            custom_exit_callback=None,
            exit_on_second_int=True,
        )
        signal_handler.register()

        try:
            self._log_agent_run()

            if self.initial_actions:
                result = await self.multi_act(self.initial_actions, check_for_new_elements=False)
                self.state.last_result = result

            for step in range(max_steps):
                if self.state.paused:
                    signal_handler.wait_for_resume()
                    signal_handler.reset()

                if self.state.consecutive_failures >= self.settings.max_failures:
                    logger.error(f'❌ Stopping due to {self.settings.max_failures} consecutive failures')
                    break

                if self.state.stopped:
                    logger.info('Agent stopped')
                    break

                while self.state.paused:
                    await asyncio.sleep(0.2)
                    if self.state.stopped:
                        break

                if on_step_start is not None:
                    await on_step_start(self)

                step_info = AgentStepInfo(step_number=step, max_steps=max_steps)
                # Pemanggilan self.step() di sini akan secara internal memanggil _get_model_output kita
                await self.step(step_info)

                if on_step_end is not None:
                    await on_step_end(self)

                if self.state.history.is_done():
                    if self.settings.validate_output and step < max_steps - 1:
                        if not await self._validate_output():
                            continue

                    await self.log_completion()
                    break
            else:
                error_message = 'Failed to complete task in maximum steps'

                self.state.history.history.append(
                    AgentHistory(
                        model_output=None,
                        result=[ActionResult(error=error_message, include_in_memory=True)],
                        state=BrowserStateHistory(
                            url='',
                            title='',
                            tabs=[],
                            interacted_element=[],
                            screenshot=None,
                        ),
                        metadata=None,
                    )
                )

                logger.info(f'❌ {error_message}')

            return self.state.history

        except KeyboardInterrupt:
            logger.info('Got KeyboardInterrupt during execution, returning current history')
            return self.state.history

        finally:
            signal_handler.unregister()

            if self.settings.save_playwright_script_path:
                logger.info(
                    f'Agent run finished. Attempting to save Playwright script to: {self.settings.save_playwright_script_path}'
                )
                try:
                    keys = list(self.sensitive_data.keys()) if self.sensitive_data else None
                    self.state.history.save_as_playwright_script(
                        self.settings.save_playwright_script_path,
                        sensitive_data_keys=keys,
                        browser_config=self.browser.config,
                        context_config=self.browser_context.config,
                    )
                except Exception as script_gen_err:
                    logger.error(f'Failed to save Playwright script: {script_gen_err}', exc_info=True)

            await self.close()

            if self.settings.generate_gif:
                output_path: str = 'agent_history.gif'
                if isinstance(self.settings.generate_gif, str):
                    output_path = self.settings.generate_gif

                create_history_gif(task=self.task, history=self.state.history, output_path=output_path)