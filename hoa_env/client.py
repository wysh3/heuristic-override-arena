from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import HOAAction, HOAObservation


class HOAEnv(EnvClient[HOAAction, HOAObservation, State]):
    """Typed client for the Heuristic Override Arena environment."""

    def _step_payload(self, action: HOAAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[HOAObservation]:
        obs_payload = payload.get("observation", {})
        observation = HOAObservation(
            **obs_payload,
            done=payload.get("done", obs_payload.get("done", False)),
            reward=payload.get("reward", obs_payload.get("reward")),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
