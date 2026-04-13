import json
from typing import Any, Dict, Iterator, List, Optional

import requests
from sqlmodel import Session, select

from app.config import get_settings
from app.models import Routine, RoutineWorkout, User, Workout

try:
	from langchain.agents import create_agent
	from langchain.tools import tool
	from langchain_core.callbacks.manager import CallbackManagerForLLMRun
	from langchain_core.language_models.llms import LLM
	from langchain_core.outputs import GenerationChunk
except ImportError:
	create_agent = None
	tool = None
	CallbackManagerForLLMRun = Any
	GenerationChunk = Any

	class LLM:  # type: ignore[override]
		pass


class SundaeCustomLLM(LLM):
	"""Custom LLM wrapper for Sundae Bytes AI endpoints."""

	api_url: str
	api_key: str
	model_name: str
	temperature: float = 0.5
	max_tokens: int = 700
	timeout_seconds: int = 45

	def _call(
		self,
		prompt: str,
		stop: Optional[List[str]] = None,
		run_manager: Optional[CallbackManagerForLLMRun] = None,
		**kwargs: Any,
	) -> str:
		headers = {
			"Authorization": f"Bearer {self.api_key}",
			"Content-Type": "application/json",
		}
		payload = {
			"model": self.model_name,
			"messages": [{"role": "user", "content": prompt}],
			"temperature": self.temperature,
			"max_tokens": self.max_tokens,
		}

		if stop:
			payload["stop"] = stop

		for endpoint in self._candidate_endpoints:
			try:
				response = requests.post(
					endpoint,
					json=payload,
					headers=headers,
					timeout=self.timeout_seconds,
				)
				response.raise_for_status()
				return self._extract_text(response.json())
			except requests.exceptions.HTTPError:
				continue
			except requests.exceptions.RequestException as exc:
				return f"Error calling model: {exc}"

		return "Error calling model: no valid AI endpoint was found for this base URL."

	def _stream(
		self,
		prompt: str,
		stop: Optional[List[str]] = None,
		run_manager: Optional[CallbackManagerForLLMRun] = None,
		**kwargs: Any,
	) -> Iterator[GenerationChunk]:
		text = self._call(prompt=prompt, stop=stop, run_manager=run_manager, **kwargs)
		for part in text.split():
			yield GenerationChunk(text=f"{part} ")

	@property
	def _candidate_endpoints(self) -> List[str]:
		base = self.api_url.rstrip("/")
		return [f"{base}/v1/chat/completions", f"{base}/chat/completions", base]

	@staticmethod
	def _extract_text(result: Dict[str, Any]) -> str:
		choices = result.get("choices")
		if isinstance(choices, list) and choices:
			first_choice = choices[0]
			message = first_choice.get("message", {})
			content = message.get("content")
			if content:
				return str(content).strip()

		if result.get("response"):
			return str(result["response"]).strip()
		if result.get("text"):
			return str(result["text"]).strip()

		return json.dumps(result)

	@property
	def _llm_type(self) -> str:
		return "sundae_custom_ai"

	@property
	def _identifying_params(self) -> Dict[str, Any]:
		return {
			"api_url": self.api_url,
			"model_name": self.model_name,
			"temperature": self.temperature,
			"max_tokens": self.max_tokens,
		}


def _build_tools(db: Session, user_id: int):
	if tool is None:
		return []

	@tool
	def list_my_routines() -> str:
		"""List all routines that belong to the current user."""
		routines = db.exec(select(Routine).where(Routine.user_id == user_id)).all()
		if not routines:
			return "No routines found for this user."

		rows = [f"- {routine.id}: {routine.name}" for routine in routines]
		return "\n".join(rows)

	@tool
	def get_routine_details(routine_name: str) -> str:
		"""Get workouts inside one of the current user's routines by routine name."""
		target = routine_name.strip().lower()
		routines = db.exec(select(Routine).where(Routine.user_id == user_id)).all()
		selected = next((routine for routine in routines if routine.name.lower() == target), None)
		if selected is None:
			return "Routine not found for current user."

		rows = db.exec(
			select(RoutineWorkout, Workout)
			.join(Workout, Workout.id == RoutineWorkout.workout_id)
			.where(RoutineWorkout.routine_id == selected.id)
			.order_by(RoutineWorkout.order)
		).all()

		if not rows:
			return f"Routine '{selected.name}' has no workouts yet."

		lines = [f"Routine: {selected.name}"]
		for association, workout in rows:
			lines.append(f"{association.order}. {workout.title} ({workout.body_part}, {workout.level})")
		return "\n".join(lines)

	@tool
	def suggest_workouts(body_part: str = "", level: str = "", workout_type: str = "") -> str:
		"""Suggest workouts filtered by body part, level, and type."""
		query = select(Workout)
		if body_part.strip():
			query = query.where(Workout.body_part.ilike(body_part.strip()))
		if level.strip():
			query = query.where(Workout.level.ilike(level.strip()))
		if workout_type.strip():
			query = query.where(Workout.type.ilike(workout_type.strip()))

		workouts = db.exec(query.limit(20)).all()
		if not workouts:
			return "No workouts matched those filters."

		rows = [
			f"- {workout.title} | type: {workout.type} | body_part: {workout.body_part} | level: {workout.level}"
			for workout in workouts
		]
		return "\n".join(rows)

	return [list_my_routines, get_routine_details, suggest_workouts]


def _extract_assistant_message(result: Dict[str, Any]) -> str:
	messages = result.get("messages") if isinstance(result, dict) else None
	if isinstance(messages, list):
		for message in reversed(messages):
			content = getattr(message, "content", None)
			if content:
				return str(content)
			if isinstance(message, dict) and message.get("content"):
				return str(message["content"])
	return "I could not generate a response right now."


def generate_assistant_response(message: str, user: User, db: Session) -> Dict[str, str]:
	settings = get_settings()
	if not settings.ai_api_key:
		return {
			"response": "AI assistant is not configured yet. Add AI_API_KEY to your .env file.",
			"model_used": settings.ai_model_name,
		}

	if create_agent is None or tool is None:
		return {
			"response": "LangChain dependencies are missing. Install langchain, langchain-core, and langchain-community.",
			"model_used": settings.ai_model_name,
		}

	model = SundaeCustomLLM(
		api_url=settings.ai_base_url,
		api_key=settings.ai_api_key,
		model_name=settings.ai_model_name,
		temperature=settings.ai_temperature,
		max_tokens=settings.ai_max_tokens,
		timeout_seconds=settings.ai_request_timeout_seconds,
	)

	assistant = create_agent(
		model=model,
		tools=_build_tools(db, user.id),
		system_prompt=(
			"You are Workout Master Assistant. Help users plan and improve workouts. "
			"Prefer using tools when users ask about their routines or workout suggestions. "
			"Keep responses practical and concise."
		),
	)

	result = assistant.invoke(
		{"messages": [{"role": "user", "content": message}]},
		config={"configurable": {"thread_id": f"user-{user.id}"}},
	)

	return {
		"response": _extract_assistant_message(result),
		"model_used": settings.ai_model_name,
	}
