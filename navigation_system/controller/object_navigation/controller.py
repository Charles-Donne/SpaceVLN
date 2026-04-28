"""OVON-specific navigation controller wrapper."""

from __future__ import annotations

from datetime import datetime
import json
import math
import os
import re
import shutil

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from navigation_system.controller.action_compat import resolve_habitat_action
from navigation_system.controller.base_controller import BaseNavigationController
from navigation_system.controller.vlnce.controller import VLMNavigationController
from navigation_system.env.object_navigation.goal_task import (
    parse_object_goal_instruction,
)
from navigation_system.render.episode_visualization.navigation_visualizer import (
    NavigationVisualizer,
)
from navigation_system.runtime.storage.artifacts import (
    SaveManager,
    get_episode_detail_dir,
    get_episode_detail_path_candidates,
)
from navigation_system.vlm.contracts.schema import normalize_objectnav_subtask_payload
from navigation_system.vlm.contracts import DIRECTION_CONFIG
from navigation_system.vlm.object_navigation.runtime_factory import (
    build_ovon_navigation_model_stack,
)
from navigation_system.runtime.object_navigation.thresholds import (
    OVON_AUTOCOMPLETE_OPENING_M,
    OVON_AUTOCOMPLETE_SOLID_M,
    OVON_AUTOCOMPLETE_TOPK,
    OVON_FINAL_OBJECT_LANDMARK_TOPK,
    OVON_FINAL_OBJECT_STOP_DISTANCE_M,
    OVON_FORCED_EARLY_STOP_DISTANCE_M,
    OVON_FORCED_EARLY_STOP_MAX_DELTA_M,
    OVON_FORCED_EARLY_STOP_THINKING_POINTS,
    OVON_TARGET_BOX_THRESHOLD,
    OVON_TARGET_TEXT_THRESHOLD,
)
from navigation_system.vlm.interfaces import NavigationModelStackBuilder


class OVONObjectNavigationController(VLMNavigationController):
    """Separate controller entrypoint for OVON/object-navigation runs."""

    ACTION_SUBTASK_AUTOCOMPLETE_OPEN_DISTANCE_M = OVON_AUTOCOMPLETE_OPENING_M
    ACTION_SUBTASK_AUTOCOMPLETE_SOLID_DISTANCE_M = OVON_AUTOCOMPLETE_SOLID_M
    ACTION_SUBTASK_AUTOCOMPLETE_TOPK = OVON_AUTOCOMPLETE_TOPK
    FINAL_DESTINATION_MATCH_AUTOSTOP_STREAK = 2
    FINAL_DESTINATION_MATCH_AUTOSTOP_RADIUS_M = OVON_FINAL_OBJECT_STOP_DISTANCE_M
    STRICT_GLOBAL_STOP_DISTANCE_M = OVON_FINAL_OBJECT_STOP_DISTANCE_M
    FORCED_EARLY_STOP_DISTANCE_M = OVON_FORCED_EARLY_STOP_DISTANCE_M
    FORCED_EARLY_STOP_MAX_DELTA_M = OVON_FORCED_EARLY_STOP_MAX_DELTA_M
    FORCED_EARLY_STOP_THINKING_POINTS = OVON_FORCED_EARLY_STOP_THINKING_POINTS

    def _reset_vlm_episode_state(self) -> None:
        super()._reset_vlm_episode_state()
        self.ovon_forced_early_stop_subtask_history = []
        self.ovon_stop_gate_requested = False
        self.ovon_stop_gate_rejection_notice = ""

    def _get_landmark_detection_thresholds(
        self,
        landmark_query: Optional[str],
    ) -> Optional[Tuple[Optional[float], Optional[float]]]:
        if self._ovon_text_contains_goal_label(landmark_query):
            return float(OVON_TARGET_BOX_THRESHOLD), float(OVON_TARGET_TEXT_THRESHOLD)
        return super()._get_landmark_detection_thresholds(landmark_query)

    def reset_episode(self, episode_id: int = None, sample_index: int = None):
        """Reset OVON state while storing artifacts under sample-index paths."""
        self.ovon_sample_index = int(sample_index) if sample_index is not None else None
        self.ovon_storage_entry_id = (
            int(self.ovon_sample_index)
            if self.ovon_sample_index is not None
            else int(episode_id if episode_id is not None else 0)
        )
        self.ovon_storage_entry_kind = (
            "sample" if self.ovon_sample_index is not None else "episode"
        )

        if self.ovon_storage_entry_id is not None:
            for old_episode_dir in get_episode_detail_path_candidates(
                self.config.PATHS.RESULTS_DIR,
                self.ovon_storage_entry_id,
                entry_kind=self.ovon_storage_entry_kind,
            ):
                if os.path.exists(old_episode_dir):
                    print(f"[Reset] Removed previous episode data: {old_episode_dir}")
                    try:
                        shutil.rmtree(old_episode_dir)
                    except PermissionError as exc:
                        raise PermissionError(
                            "Cannot remove stale episode outputs before reset: "
                            f"{old_episode_dir}. This is usually caused by files created "
                            "by another user or by a Docker container running as root. "
                            "Fix the ownership or delete the stale directory first."
                        ) from exc

        BaseNavigationController.reset_episode(self, episode_id)
        if hasattr(self.visualizer, "set_storage_entry"):
            self.visualizer.set_storage_entry(
                self.ovon_storage_entry_id,
                self.ovon_storage_entry_kind,
            )

        self.save_manager = SaveManager(
            self.config.PATHS.RESULTS_DIR,
            self.current_episode_id,
            storage_entry_id=self.ovon_storage_entry_id,
            entry_kind=self.ovon_storage_entry_kind,
            save_waypoint_memory=self.runtime_options.save_waypoint_memory,
        )

        self._reset_vlm_episode_state()

        visualization_dir = os.path.join(self.episode_dir, "visualization")
        self.nav_visualizer = NavigationVisualizer(
            visualization_dir,
            save_step_images=self.runtime_options.save_navigation_step_images,
            keep_frames_for_gif=self.runtime_options.save_navigation_gif,
        )
        self.nav_visualizer.setup_maps_dir(self.episode_dir)

    @property
    def episode_dir(self) -> str:
        return get_episode_detail_dir(
            self.config.PATHS.RESULTS_DIR,
            int(getattr(self, "ovon_storage_entry_id", self.current_episode_id) or 0),
            entry_kind=str(getattr(self, "ovon_storage_entry_kind", "episode") or "episode"),
        )

    @property
    def current_episode_dir(self) -> str:
        return self.episode_dir

    def _ovon_is_final_object_stage(self) -> bool:
        current_subtask = getattr(self, "current_subtask", None) or {}
        if not isinstance(current_subtask, dict):
            return False

        subtask_landmark = self._get_subtask_landmark_field(current_subtask)
        return self._ovon_text_contains_goal_label(subtask_landmark)

    def _get_action_landmark_topk(self) -> int:
        # Final target-object stage keeps the narrower prompt context, but the
        # detection pipeline itself must still be allowed to observe landmarks.
        if self._ovon_is_final_object_stage():
            return int(OVON_FINAL_OBJECT_LANDMARK_TOPK)
        return int(self.ACTION_SUBTASK_AUTOCOMPLETE_TOPK)

    def _ovon_select_final_stage_landmark_entries(
        self,
        entries: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ordered_entries = self._sort_action_landmark_entries([
            dict(entry)
            for entry in (entries or [])
            if isinstance(entry, dict)
        ])
        if not self._ovon_is_final_object_stage():
            return ordered_entries

        # In OVON action, keep non-goal landmarks visible/detectable. Only bias
        # the ranking so the true target object appears earlier when present.
        ordered_entries.sort(
            key=lambda entry: (
                0 if self._ovon_text_contains_goal_label(entry.get("name")) else 1,
                self._safe_int(entry.get("selection_rank"))
                if self._safe_int(entry.get("selection_rank")) is not None
                else 1e9,
                -float(self._safe_float(entry.get("confidence")) or 0.0),
                self._safe_float(entry.get("distance_m"))
                if self._safe_float(entry.get("distance_m")) is not None
                else float("inf"),
                str(entry.get("name", "")),
            )
        )
        return ordered_entries

    def _get_latest_action_local_map_landmark_entries(self) -> List[Dict[str, Any]]:
        return self._ovon_select_final_stage_landmark_entries(
            super()._get_latest_action_local_map_landmark_entries()
        )

    def _get_action_landmark_prompt_entries(self, detection_step: Optional[int]) -> List[Dict[str, Any]]:
        return self._ovon_select_final_stage_landmark_entries(
            super()._get_action_landmark_prompt_entries(detection_step)
        )

    def _get_current_action_step_landmark_entries(self) -> List[Dict[str, Any]]:
        return self._ovon_select_final_stage_landmark_entries(
            super()._get_current_action_step_landmark_entries()
        )

    def __init__(
        self,
        config,
        config_path: str = "navigation_system/config/vlm/vlm_api_config.yaml",
        model_stack_builder: Optional[NavigationModelStackBuilder] = None,
        envs=None,
    ):
        super().__init__(
            config,
            config_path=config_path,
            model_stack_builder=model_stack_builder or build_ovon_navigation_model_stack,
            envs=envs,
        )

    @classmethod
    def _ovon_label_variants(cls, text: Optional[str]) -> Set[str]:
        normalized = cls._normalize_landmark_text(text)
        if not normalized:
            return set()
        normalized = " ".join(
            word for word in normalized.split() if word not in {"a", "an", "the"}
        )
        if not normalized:
            return set()

        variants = {normalized}
        words = normalized.split()
        if words:
            last = words[-1]
            if last.endswith("s") and len(last) > 1:
                variants.add(" ".join([*words[:-1], last[:-1]]).strip())
            else:
                variants.add(" ".join([*words[:-1], f"{last}s"]).strip())
        return {variant for variant in variants if variant}

    @classmethod
    def _ovon_landmark_part(cls, text: Optional[str]) -> str:
        cleaned = str(text or "").replace("’", "'").replace("`", "'").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"\([^)]*\)", " ", cleaned)
        if "'s" in cleaned:
            cleaned = cleaned.rsplit("'s", 1)[1]
        cleaned = re.sub(r"\b(?:goal|current|target object anchor|target anchor)\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"[^A-Za-z0-9\s-]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _ovon_goal_terms(self) -> Set[str]:
        goal, aliases = parse_object_goal_instruction(getattr(self, "current_instruction", ""))
        terms: Set[str] = set()
        for raw_term in (goal, *aliases):
            terms.update(self._ovon_label_variants(raw_term))
        return terms

    def _ovon_goal_object_name(self) -> str:
        goal, _aliases = parse_object_goal_instruction(getattr(self, "current_instruction", ""))
        return self._normalize_landmark_text(goal) or ""

    def _ovon_exact_goal_label(self) -> str:
        return self._ovon_goal_object_name()

    @staticmethod
    def _ovon_phrase_in_text(text: Optional[str], phrase: Optional[str]) -> bool:
        normalized_text = str(text or "").strip()
        phrase_text = str(phrase or "").strip()
        if not normalized_text or not phrase_text:
            return False
        text_tokens = normalized_text.split()
        phrase_tokens = phrase_text.split()
        if not text_tokens or not phrase_tokens:
            return False
        phrase_len = len(phrase_tokens)
        return any(
            text_tokens[idx: idx + phrase_len] == phrase_tokens
            for idx in range(0, len(text_tokens) - phrase_len + 1)
        )

    def _ovon_text_contains_goal_label(self, text: Optional[str]) -> bool:
        goal_terms = self._ovon_goal_terms()
        if not goal_terms:
            return False
        text_terms = {
            self._normalize_landmark_text(text),
            self._normalize_landmark_text(self._ovon_landmark_part(text)),
        }
        text_terms = {term for term in text_terms if term}
        if not text_terms:
            return False
        for text_term in text_terms:
            text_variants = self._ovon_label_variants(text_term) or {text_term}
            for text_variant in text_variants:
                if text_variant in goal_terms:
                    return True
                if any(self._ovon_phrase_in_text(text_variant, goal_term) for goal_term in goal_terms):
                    return True
        return False

    def _ovon_response_names_target_object(self, response: Dict[str, Any]) -> Tuple[bool, str]:
        subtask_landmark = str(response.get("subtask_landmark") or "").strip()
        if not self._ovon_text_contains_goal_label(subtask_landmark):
            return False, "subtask_landmark does not contain the OVON target object"
        return True, ""

    def _ovon_response_enters_final_object_stage(self, response: Optional[Dict[str, Any]]) -> bool:
        payload = dict(response or {})
        if not payload:
            return False
        return self._ovon_text_contains_goal_label(payload.get("subtask_landmark"))

    @staticmethod
    def _ovon_float(value: Optional[Any]) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _ovon_target_detection_within_distance(
        self,
        distance_threshold_m: float,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        entries: Sequence[Dict[str, Any]] = []
        try:
            entries = self._get_latest_action_local_map_landmark_entries()
        except Exception:
            entries = []

        for entry in entries or []:
            if not self._ovon_text_contains_goal_label(entry.get("name")):
                continue
            distance_m = self._ovon_float(entry.get("distance_m"))
            if distance_m is None:
                continue
            if distance_m <= float(distance_threshold_m):
                return True, dict(entry)
        return False, None

    def _ovon_previous_subtask_target_within_distance(
        self,
        distance_threshold_m: float,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        info = dict(getattr(self, "previous_subtask_landmark_final_info", {}) or {})
        if not info:
            return False, None

        entries = list(info.get("entries") or [])
        if not entries:
            entries = [info]

        for entry in entries:
            if not self._ovon_text_contains_goal_label(
                entry.get("raw_name") or entry.get("name")
            ):
                continue
            has_arrived = bool(entry.get("has_arrived", False))
            distance_m = self._ovon_float(
                entry.get("final_distance_m", entry.get("distance_m"))
            )
            stop_distance_m = self._ovon_float(entry.get("stop_distance_m"))
            if has_arrived:
                if distance_m is None:
                    return True, dict(entry)
                effective_arrival_threshold_m = float(distance_threshold_m)
                if stop_distance_m is not None and stop_distance_m > 0.0:
                    effective_arrival_threshold_m = max(
                        effective_arrival_threshold_m,
                        float(stop_distance_m),
                    )
                if distance_m <= effective_arrival_threshold_m:
                    return True, dict(entry)
            if distance_m is None:
                continue
            if distance_m <= float(distance_threshold_m):
                return True, dict(entry)
        return False, None

    def _ovon_build_stop_gate_notice(self, reason: str) -> str:
        exact_goal = self._ovon_exact_goal_label() or "the exact target object"
        reason_text = str(reason or "").strip()
        base_notice = (
            f"Set `global_landmark_arrival=true` on verify only when the current "
            f"`subtask_landmark` contains `{exact_goal}`, the previous executed action subtask also used "
            f"a landmark containing `{exact_goal}` as its `subtask_landmark`, and the Previous Subtask landmark summary shows "
            f"`{exact_goal}` within about {float(self.FORCED_EARLY_STOP_DISTANCE_M):.2f}m or explicitly as already reached. "
            f"Then judge from the current views and surrounding space whether "
            f"this is the real target object in a reasonable location rather than a misdetection. "
            f"If any of these is missing, keep `global_landmark_arrival=false` and continue approaching `{exact_goal}`."
        )
        if not reason_text:
            return base_notice
        return f"{base_notice} Current rejection reason: {reason_text}."

    def _ovon_previous_action_landmark_matches_goal(self) -> Tuple[bool, str]:
        previous_subtask = getattr(self, "current_subtask", None)
        if not isinstance(previous_subtask, dict):
            return False, "no previous action subtask is available for OVON stop gating"

        exact_goal = self._ovon_exact_goal_label()
        if not exact_goal:
            return False, "OVON target object is unavailable"

        previous_landmark = self._normalize_landmark_text(
            self._get_subtask_landmark_field(previous_subtask)
        )
        if not previous_landmark:
            return False, "previous action subtask landmark is empty"
        if not self._ovon_text_contains_goal_label(previous_landmark):
            return False, "previous action subtask landmark does not contain the OVON target object"
        return True, ""

    def _ovon_effective_global_landmark_arrival(
        self,
        response: Optional[Dict[str, Any]],
        *,
        log_rejection: bool = False,
        log_upgrade: bool = False,
    ) -> bool:
        _ = log_upgrade
        payload = dict(response or {})
        requested_stop = bool(payload.get("global_landmark_arrival", False))
        self.ovon_stop_gate_requested = bool(requested_stop)
        self.ovon_stop_gate_rejection_notice = ""
        if not requested_stop:
            return False

        stop_ok, stop_reason = self._ovon_response_names_target_object(payload)
        if not stop_ok:
            self.ovon_stop_gate_rejection_notice = self._ovon_build_stop_gate_notice(stop_reason)
            if log_rejection:
                print(f"[OVONStopGate] reject planner stop: {stop_reason}")
            return False

        previous_action_ok, previous_action_reason = self._ovon_previous_action_landmark_matches_goal()
        if not previous_action_ok:
            self.ovon_stop_gate_rejection_notice = self._ovon_build_stop_gate_notice(previous_action_reason)
            if log_rejection:
                print(f"[OVONStopGate] reject planner stop: {previous_action_reason}")
            return False

        previous_summary_ok, _previous_summary_entry = self._ovon_previous_subtask_target_within_distance(
            float(self.FORCED_EARLY_STOP_DISTANCE_M)
        )
        if not previous_summary_ok:
            rejection_reason = (
                "Previous Subtask landmark summary does not show the OVON target object "
                f"within {float(self.FORCED_EARLY_STOP_DISTANCE_M):.2f}m"
            )
            self.ovon_stop_gate_rejection_notice = self._ovon_build_stop_gate_notice(rejection_reason)
            if log_rejection:
                print(
                    f"[OVONStopGate] reject planner stop: {rejection_reason}"
                )
            return False

        return True

    def _sanitize_planner_response(self, response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        response = super()._sanitize_planner_response(response)
        if not response:
            return response

        response["next_waypoint_direction"] = self._ovon_normalize_direction_label(
            response.get("next_waypoint_direction")
        )
        response["next_waypoint"] = self._ovon_normalize_next_waypoint_text(
            response.get("next_waypoint"),
            response=response,
        )
        response["subtask_instruction"] = self._sanitize_subtask_instruction_text(
            response.get("subtask_instruction"),
            response.get("next_waypoint"),
            response.get("next_waypoint_direction"),
            keep_view_prefix=True,
        )

        exact_goal = self._ovon_exact_goal_label()

        valid_arrival = self._ovon_effective_global_landmark_arrival(
            response,
            log_rejection=True,
            log_upgrade=True,
        )
        response["global_landmark_arrival"] = bool(valid_arrival)
        response.pop("global_task_finish", None)
        return response

    def _normalize_controller_response_payload(
        self,
        response: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        return normalize_objectnav_subtask_payload(response)

    @staticmethod
    def _is_task_finished(response: Optional[Dict[str, Any]]) -> bool:
        return bool((response or {}).get("global_landmark_arrival", False))

    @staticmethod
    def _set_response_task_finished(response: Dict[str, Any], value: bool) -> None:
        response["global_landmark_arrival"] = bool(value)
        response.pop("global_task_finish", None)

    @staticmethod
    def _ovon_extract_image_index_from_direction_text(text: Optional[str]) -> Optional[int]:
        match = re.search(r"IMAGE\s*(\d+)", str(text or ""), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _ovon_direction_body_to_image_label(cls, text: Optional[str]) -> str:
        raw_text = str(text or "").strip()
        if not raw_text:
            return ""
        if raw_text.startswith("IMAGE") and "(" in raw_text and ")" in raw_text:
            return raw_text

        image_index = cls._ovon_extract_image_index_from_direction_text(raw_text)
        direction_body = raw_text
        if ":" in raw_text:
            maybe_prefix, maybe_body = raw_text.split(":", 1)
            if cls._ovon_extract_image_index_from_direction_text(maybe_prefix) is not None:
                direction_body = maybe_body.strip()

        if image_index is None:
            normalized_body = direction_body.lower().replace("°", "deg")
            for item in DIRECTION_CONFIG:
                item_name = str(item.get("name") or "").strip()
                body = re.sub(r"^IMAGE\s*\d+\s*:\s*", "", item_name, flags=re.IGNORECASE)
                body_norm = body.lower().replace("°", "deg")
                if body_norm == normalized_body:
                    image_index = cls._ovon_extract_image_index_from_direction_text(item_name)
                    direction_body = body
                    break

        if image_index is None:
            return raw_text
        if not direction_body:
            for item in DIRECTION_CONFIG:
                if int(item.get("step", -1) or -1) == int(image_index):
                    direction_body = re.sub(
                        r"^IMAGE\s*\d+\s*:\s*",
                        "",
                        str(item.get("name") or "").strip(),
                        flags=re.IGNORECASE,
                    )
                    break
        return f"IMAGE {int(image_index)} ({direction_body})".strip()

    def _ovon_normalize_direction_label(self, text: Optional[str]) -> str:
        return self._ovon_direction_body_to_image_label(text)

    def _ovon_normalize_next_waypoint_text(
        self,
        text: Optional[str],
        *,
        response: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        cleaned = self._sanitize_next_waypoint_text(text)
        if cleaned is None:
            return None
        cleaned = str(cleaned).strip()
        if not cleaned:
            return cleaned
        if "'s" in cleaned:
            return cleaned
        if " / " in cleaned:
            space_part, local_part = cleaned.split(" / ", 1)
            space_part = space_part.strip(" ,;:-")
            local_part = local_part.strip(" ,;:-")
            if space_part and local_part:
                return f"{space_part}'s {local_part}"

        current_waypoint = str((response or {}).get("current_waypoint") or "").strip()
        goal_object = self._ovon_goal_object_name()
        if goal_object and self._ovon_text_contains_goal_label(cleaned) and " - " in current_waypoint:
            space_part, _local = current_waypoint.split(" - ", 1)
            space_part = space_part.strip(" ,;:-")
            if space_part:
                return f"{space_part}'s {goal_object}"
        return cleaned

    def _ovon_persist_thinking_response_artifact(
        self,
        response: Dict[str, Any],
        cycle_info: Dict[str, Any],
    ) -> None:
        thinking_dir = str((cycle_info or {}).get("thinking_dir") or "").strip()
        if not thinking_dir:
            return
        try:
            artifact_payload = dict(response or {})
            artifact_payload.pop("global_task_finish", None)
            self._write_json_artifact(
                os.path.join(thinking_dir, "response.controller.json"),
                artifact_payload,
            )
        except OSError:
            return

    def _ovon_reset_forced_early_stop_subtask_history(self) -> None:
        self.ovon_forced_early_stop_subtask_history = []

    def _ovon_forced_early_stop_ready(
        self,
        response: Optional[Dict[str, Any]],
        *,
        proposed_subtask_id: int,
    ) -> Tuple[bool, str]:
        if not isinstance(response, dict):
            self._ovon_reset_forced_early_stop_subtask_history()
            return False, "no OVON response available for forced early-stop checking"

        exact_goal = self._ovon_exact_goal_label()
        current_landmark = self._normalize_landmark_text(
            self._get_subtask_landmark_field(response)
        )
        if not exact_goal or not self._ovon_text_contains_goal_label(current_landmark):
            self._ovon_reset_forced_early_stop_subtask_history()
            return False, "current subtask landmark does not contain the final target object"

        previous_action_ok, previous_action_reason = self._ovon_previous_action_landmark_matches_goal()
        if not previous_action_ok:
            self._ovon_reset_forced_early_stop_subtask_history()
            return False, previous_action_reason

        previous_summary_ok, _previous_summary_entry = self._ovon_previous_subtask_target_within_distance(
            float(self.FORCED_EARLY_STOP_DISTANCE_M)
        )
        if not previous_summary_ok:
            self._ovon_reset_forced_early_stop_subtask_history()
            return False, (
                "Previous Subtask landmark summary does not show the target object within "
                f"{float(self.FORCED_EARLY_STOP_DISTANCE_M):.2f}m"
            )

        target_seen, matched_entry = self._ovon_target_detection_within_distance(
            float(self.FORCED_EARLY_STOP_DISTANCE_M)
        )
        if not target_seen:
            self._ovon_reset_forced_early_stop_subtask_history()
            return False, "target object is not within the forced early-stop detection radius"

        pose_xy = self._extract_pose_xy(self._get_agent_pose())
        if pose_xy is None:
            self._ovon_reset_forced_early_stop_subtask_history()
            return False, "current pose is unavailable for forced early-stop checking"

        snapshot = {
            "subtask_id": int(proposed_subtask_id),
            "pose_xy": (float(pose_xy[0]), float(pose_xy[1])),
            "step": int(getattr(self, "current_step", 0) or 0),
            "distance_m": self._ovon_float((matched_entry or {}).get("distance_m")),
        }

        history: List[Dict[str, Any]] = list(
            getattr(self, "ovon_forced_early_stop_subtask_history", []) or []
        )
        if history:
            last_subtask_id = int(history[-1].get("subtask_id", -1) or -1)
            if proposed_subtask_id == last_subtask_id:
                history[-1] = snapshot
            elif proposed_subtask_id == last_subtask_id + 1:
                history.append(snapshot)
            else:
                history = [snapshot]
        else:
            history = [snapshot]

        required = int(self.FORCED_EARLY_STOP_THINKING_POINTS)
        history = history[-required:]
        self.ovon_forced_early_stop_subtask_history = history

        if len(history) < required:
            return False, (
                f"forced early-stop final-target thinking history "
                f"{len(history)}/{required}"
            )

        pair_movements: List[float] = []
        for prev_item, next_item in zip(history[:-1], history[1:]):
            prev_pose = prev_item.get("pose_xy")
            next_pose = next_item.get("pose_xy")
            if not prev_pose or not next_pose:
                self.ovon_forced_early_stop_subtask_history = [snapshot]
                return False, "missing historical pose for forced early-stop checking"
            pair_movements.append(
                float(
                    ((next_pose[0] - prev_pose[0]) ** 2 + (next_pose[1] - prev_pose[1]) ** 2) ** 0.5
                )
            )

        total_drift_m = float(
            ((history[-1]["pose_xy"][0] - history[0]["pose_xy"][0]) ** 2 +
             (history[-1]["pose_xy"][1] - history[0]["pose_xy"][1]) ** 2) ** 0.5
        )
        if any(move_m > float(self.FORCED_EARLY_STOP_MAX_DELTA_M) for move_m in pair_movements):
            self.ovon_forced_early_stop_subtask_history = [snapshot]
            return False, (
                "forced early-stop reset because consecutive final-target subtasks did not stay "
                f"within {float(self.FORCED_EARLY_STOP_MAX_DELTA_M):.2f}m"
            )
        if total_drift_m > float(self.FORCED_EARLY_STOP_MAX_DELTA_M):
            self.ovon_forced_early_stop_subtask_history = [snapshot]
            return False, (
                f"forced early-stop total drift {total_drift_m:.2f}m exceeds "
                f"{float(self.FORCED_EARLY_STOP_MAX_DELTA_M):.2f}m"
            )

        target_distance = self._ovon_float((matched_entry or {}).get("distance_m"))
        target_distance_text = (
            f"{float(target_distance):.2f}m"
            if target_distance is not None
            else "unknown distance"
        )
        return True, (
            f"exact final-target stage persisted across one executed subtask "
            f"({required} thinking calls), that subtask move stayed within "
            f"{float(self.FORCED_EARLY_STOP_MAX_DELTA_M):.2f}m, total drift={total_drift_m:.2f}m, "
            f"and target detection is within {target_distance_text}"
        )

    def _update_final_goal_destination_match_streak(
        self,
        response: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[str], int, Optional[float], bool, bool]:
        """Disable VLNCE-style implicit final-waypoint autostop for OVON."""
        self._reset_final_goal_destination_match_state()
        waypoint_chain = response.get("waypoint_chain") or response.get("waypoint_sequence") or ""
        next_destination = self._get_next_waypoint_field(response)
        last_chain_node = self._extract_last_waypoint_chain_node(waypoint_chain)
        response["final_waypoint_chain_goal"] = last_chain_node or ""
        response["final_waypoint_destination_match_streak"] = 0
        response["final_waypoint_destination_anchor_distance_m"] = None
        response["final_waypoint_destination_anchor_radius_m"] = float(self.STRICT_GLOBAL_STOP_DISTANCE_M)
        response["final_waypoint_destination_anchor_region_stable"] = False
        return False, last_chain_node, str(next_destination).strip() or None, 0, None, False, False

    def _apply_thinking_cycle_result(
        self,
        response: Dict[str, Any],
        cycle_info: Dict[str, Any],
        mode: str,
    ) -> bool:
        mode_key = str(mode).strip().lower()
        response["global_landmark_arrival"] = self._ovon_effective_global_landmark_arrival(
            response,
            log_rejection=True,
            log_upgrade=True,
        )
        if mode_key == "initial" and bool(response.get("global_landmark_arrival", False)):
            print(
                "[OVONStopGate] reject initial-planning stop: OVON must execute at least one "
                "action subtask and stop only on a later verify call."
            )
            response["global_landmark_arrival"] = False
        if bool(response.get("global_landmark_arrival", False)):
            self._ovon_reset_forced_early_stop_subtask_history()
        else:
            proposed_subtask_id = 1 if mode_key == "initial" else int(self.subtask_count) + 1
            forced_stop_ok, forced_reason = self._ovon_forced_early_stop_ready(
                response,
                proposed_subtask_id=int(proposed_subtask_id),
            )
            if forced_stop_ok:
                exact_goal = self._ovon_exact_goal_label()
                response["global_landmark_arrival"] = True
                if exact_goal:
                    response["subtask_landmark"] = exact_goal
                    response["next_waypoint"] = exact_goal
                print(
                    "[OVONEarlyStop] "
                    f"{forced_reason}. Stop on this thinking call instead of looping."
                )
        task_finished = bool(response.get("global_landmark_arrival", False))
        self._ovon_persist_thinking_response_artifact(response, cycle_info)
        if mode_key == "initial" and bool(task_finished):
            phase = str(cycle_info.get("phase", "initial"))
            self._apply_postplanning_space_area_update(
                response=response,
                phase=phase,
                thinking_dir=cycle_info.get("thinking_dir"),
                refresh_direction_views=True,
            )
            self._record_current_position_from_thinking_response(response)
            self.current_subtask = response
            self.subtask_count = 1
            self.subtask_attempt = 0
            self._print_subtask_info(response, is_initial=True)
            print("[DONE] Global task complete at initial planning")
            return True
        return super()._apply_thinking_cycle_result(
            response=response,
            cycle_info=cycle_info,
            mode=mode,
        )

    def _run_thinking_cycle(
        self,
        mode: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
        response, prompt, cycle_info = super()._run_thinking_cycle(mode)
        mode_key = str(mode).strip().lower()
        if mode_key != "verify" or not response or not cycle_info:
            return response, prompt, cycle_info

        if not bool(getattr(self, "ovon_stop_gate_requested", False)):
            return response, prompt, cycle_info
        retry_notice = str(getattr(self, "ovon_stop_gate_rejection_notice", "") or "").strip()
        if not retry_notice:
            return response, prompt, cycle_info

        print("[OVONStopGate] verify stop request was rejected; re-query once with a stop-guard notice")
        verify_subtask = dict(self.current_subtask or {})
        verify_subtask["subtask_instruction"] = self._build_previous_subtask_instruction_summary(
            self.current_subtask
        )
        retry_response, retry_prompt = self.planner.verify_and_replan(
            instruction=self.current_instruction,
            current_subtask=verify_subtask,
            observation_images=list(cycle_info.get("image_paths") or []),
            direction_names=list(cycle_info.get("direction_names") or []),
            global_map_image=cycle_info.get("global_map_image"),
            local_map_image=None,
            detected_landmarks=list(cycle_info.get("detected_landmarks") or []),
            waypoint_summary=cycle_info.get("waypoint_summary"),
            previous_subtask_landmark_summary=cycle_info.get("previous_subtask_landmark_summary"),
            obstacle_distances=getattr(self, "latest_obstacle_distances", None),
            verify_replan_prompt_notice=self._merge_prompt_notices(
                str(getattr(self, "verify_replan_prompt_notice", "") or "").strip(),
                retry_notice,
            ),
            save_dir=cycle_info.get("thinking_dir"),
        )
        if not retry_response:
            return response, prompt, cycle_info

        raw_retry_response = self._get_latest_planner_raw_response_payload(retry_response)
        retry_response = self._sanitize_planner_response(retry_response)
        thinking_dir = str(cycle_info.get("thinking_dir") or "").strip()
        if thinking_dir:
            try:
                self._persist_response_artifacts(
                    save_dir=thinking_dir,
                    raw_payload=raw_retry_response,
                    controller_payload=retry_response,
                )
            except OSError:
                pass

        updated_cycle_info = dict(cycle_info)
        self.latest_thinking_cycle_info = dict(updated_cycle_info)
        return retry_response, retry_prompt, updated_cycle_info

    def _should_autostop_from_goal_distance(self) -> bool:
        """OVON global STOP follows the planner's explicit final-arrival flag only."""
        current_subtask = getattr(self, "current_subtask", None)
        if not isinstance(current_subtask, dict):
            return False

        if self._ovon_effective_global_landmark_arrival(
            current_subtask,
            log_rejection=False,
            log_upgrade=False,
        ):
            current_subtask["global_landmark_arrival"] = True
            return True
        return False

    def _attempt_goal_distance_autostop(self) -> bool:
        current_subtask = getattr(self, "current_subtask", None)
        planner_stop_ok = False
        if isinstance(current_subtask, dict):
            planner_stop_ok = self._ovon_effective_global_landmark_arrival(
                current_subtask,
                log_rejection=False,
                log_upgrade=False,
            )
            if planner_stop_ok:
                current_subtask["global_landmark_arrival"] = True
        if not planner_stop_ok:
            return False

        print("[OVONAutoStop] planner final-arrival flag is true; issuing STOP immediately.")

        result = self.step_with_vlm(
            resolve_habitat_action("STOP"),
            action_name="OVON_AUTO_GOAL_STOP",
            save_vis=True,
            enable_landmark_detection=False,
        )
        if result.get("done", False):
            print("[OVONAutoStop] STOP executed and episode finished.")
            return True
        print("[OVONAutoStop] STOP was issued but episode did not finish; continue.")
        return False

    def _normalize_final_env_metrics(self, env_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        metrics = dict(env_metrics or self.latest_info or {})
        if bool(getattr(self, "final_stop_skipped_due_to_done", False)):
            metrics["_final_success_inferred"] = False
        return metrics

    def _reset_custom_landmark_state(self) -> None:
        """OVON resets landmark memory at every new subtask, same as VLNCE."""
        VLMNavigationController._reset_custom_landmark_state(self)

    def _should_autocomplete_subtask_during_action_step(
        self,
        step_landmark_entries: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        landmark_match_terms = self._get_current_subtask_autocomplete_match_terms()
        if not landmark_match_terms:
            return None

        final_object_stage = self._ovon_is_final_object_stage()
        effective_topk = self._get_action_landmark_topk()
        ordered_entries = [
            dict(entry)
            for entry in self._sort_action_landmark_entries(step_landmark_entries or [])
        ]

        if final_object_stage:
            ordered_entries = [
                entry
                for entry in ordered_entries
                if self._ovon_text_contains_goal_label(entry.get("name"))
            ]
            landmark_match_terms = [
                term for term in landmark_match_terms
                if self._ovon_text_contains_goal_label(term)
            ]
            if not landmark_match_terms:
                return None

        matches: List[Dict[str, Any]] = []
        for entry in ordered_entries[: max(1, effective_topk)]:
            if not self._entry_reaches_action_arrival_threshold(
                entry,
                landmark_match_terms=landmark_match_terms,
            ):
                continue
            matches.append({
                "name": str(entry.get("name") or landmark_match_terms[0]),
                "distance_m": float(entry.get("distance_m")),
                "confidence": float(entry.get("confidence", 0.0) or 0.0),
                "angle_deg": entry.get("angle_deg"),
                "is_opening_like": bool(self._is_opening_like_landmark_entry(entry)),
                "stop_distance_m": float(self._autocomplete_stop_distance_m(entry)),
                "source": "vis" if str(entry.get("source", "mem") or "mem") == "vis" else "mem",
                "display_id": self._safe_int(entry.get("display_id")),
                "instance_idx": self._safe_int(entry.get("instance_idx")),
                "class_total": self._safe_int(entry.get("class_total")),
                "selection_rank": self._safe_int(entry.get("selection_rank")),
                "instance_uid": self._safe_int(entry.get("instance_uid")),
            })

        if not matches:
            return None

        matches.sort(
            key=lambda item: (
                float(item.get("distance_m", 1e9)),
                -float(item.get("confidence", 0.0)),
                str(item.get("name", "")),
            )
        )
        return matches[0]

    def _save_navigation_result(self, total_steps: int, env_metrics: Dict = None) -> str:
        """Save OVON results without VLNCE-only metrics such as OSR/nDTW."""

        def check_inf_nan(value):
            if isinstance(value, (int, float)) and (math.isinf(value) or math.isnan(value)):
                return 0
            return value

        metrics_source = dict(env_metrics if env_metrics else (self.latest_info if self.latest_info else {}))
        episode_timing_summary = self._build_episode_timing_summary()
        result = {
            "episode_id": self.current_episode_id,
            "instruction": self.current_instruction,
            "total_steps": total_steps,
            "subtask_count": self.subtask_count,
            "episode_duration_s": episode_timing_summary["episode_duration_s"],
            "failed_api_total_duration_s": episode_timing_summary["failed_api_total_duration_s"],
            "failed_retry_wait_duration_s": episode_timing_summary["failed_retry_wait_duration_s"],
            "failed_wasted_duration_s": episode_timing_summary["failed_wasted_duration_s"],
            "success": int(check_inf_nan(metrics_source.get("success", 0))),
            "spl": float(check_inf_nan(metrics_source.get("spl", 0.0))),
            "soft_spl": float(check_inf_nan(metrics_source.get("soft_spl", 0.0))),
            "distance_to_goal": float(check_inf_nan(metrics_source.get("distance_to_goal", -1.0))),
            "path_length": float(check_inf_nan(metrics_source.get("path_length", 0.0))),
            "thinking_api_summary": episode_timing_summary["thinking_api_summary"],
            "action_api_summary": episode_timing_summary["action_api_summary"],
            "timestamp": datetime.now().isoformat(),
        }
        result["sr"] = result["success"]
        result["ne"] = result["distance_to_goal"]
        if getattr(self, "ovon_sample_index", None) is not None:
            result["sample_index"] = int(self.ovon_sample_index)
        return self.save_manager.save_result(result)
