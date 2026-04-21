"""OVON-specific navigation controller wrapper."""

from __future__ import annotations

import re

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from navigation_system.controller.vlnce.controller import VLMNavigationController
from navigation_system.env.object_navigation.goal_task import (
    OBJECT_SPACE_PRIORS,
    parse_object_goal_instruction,
)
from navigation_system.vlm.object_navigation.runtime_factory import (
    build_ovon_navigation_model_stack,
)
from navigation_system.runtime.object_navigation.thresholds import (
    OVON_AUTOCOMPLETE_OPENING_M,
    OVON_AUTOCOMPLETE_SOLID_M,
    OVON_AUTOCOMPLETE_TOPK,
    OVON_FINAL_OBJECT_STOP_DISTANCE_M,
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

    def _ovon_is_final_object_stage(self) -> bool:
        current_subtask = getattr(self, "current_subtask", None) or {}
        if not isinstance(current_subtask, dict):
            return False

        subtask_landmark = self._get_subtask_landmark_field(current_subtask)
        next_waypoint = self._get_next_waypoint_field(current_subtask)
        waypoint_tail = self._extract_last_waypoint_chain_node(
            current_subtask.get("waypoint_chain") or current_subtask.get("waypoint_sequence") or ""
        )
        return any(
            self._ovon_text_exactly_matches_goal(item)
            for item in (subtask_landmark, next_waypoint, waypoint_tail)
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
        normalized = cls._normalize_landmark_candidate(text)
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
        return self._normalize_landmark_candidate(goal) or ""

    def _ovon_text_exactly_matches_goal(self, text: Optional[str]) -> bool:
        goal_terms = self._ovon_goal_terms()
        if not goal_terms:
            return False
        variants = self._ovon_label_variants(self._ovon_landmark_part(text))
        variants.update(self._ovon_label_variants(text))
        return bool(variants and goal_terms.intersection(variants))

    def _ovon_response_names_target_object(self, response: Dict[str, Any]) -> Tuple[bool, str]:
        subtask_landmark = str(response.get("subtask_landmark") or "").strip()
        if not self._ovon_text_exactly_matches_goal(subtask_landmark):
            return False, "subtask_landmark does not exactly match the OVON target object"

        next_waypoint = self._get_next_waypoint_field(response)
        waypoint_tail = self._extract_last_waypoint_chain_node(
            response.get("waypoint_chain") or response.get("waypoint_sequence") or ""
        )
        if not (
            self._ovon_text_exactly_matches_goal(next_waypoint)
            or self._ovon_text_exactly_matches_goal(waypoint_tail)
        ):
            return False, "next_waypoint / waypoint_chain tail does not exactly name the OVON target object"
        return True, ""

    def _ovon_response_has_plausible_space_context(self, response: Dict[str, Any]) -> Tuple[bool, str]:
        goal_name = self._ovon_goal_object_name()
        if not goal_name:
            return True, ""

        likely_spaces = list(OBJECT_SPACE_PRIORS.get(goal_name, []))
        if not likely_spaces:
            return True, ""

        context_fields = [
            response.get("current_waypoint"),
            response.get("next_waypoint"),
            response.get("waypoint_chain"),
            response.get("task_progress"),
            response.get("subtask_instruction"),
        ]
        context_text = " ".join(str(item or "") for item in context_fields)
        normalized_context = self._normalize_landmark_candidate(context_text) or ""
        if not normalized_context:
            return False, "missing space context for final target-object stop"

        normalized_spaces = [
            self._normalize_landmark_candidate(space_name)
            for space_name in likely_spaces
        ]
        if any(space_name and space_name in normalized_context for space_name in normalized_spaces):
            return True, ""

        return (
            False,
            f"final response does not place target object in a plausible target space "
            f"({', '.join(likely_spaces)})",
        )

    @staticmethod
    def _ovon_float(value: Optional[Any]) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _ovon_distance_to_goal_m(self) -> Optional[float]:
        latest_info = getattr(self, "latest_info", None)
        if not isinstance(latest_info, dict):
            return None
        distance_to_goal = self._ovon_float(latest_info.get("distance_to_goal", -1.0))
        if distance_to_goal is None or distance_to_goal < 0.0:
            return None
        return float(distance_to_goal)

    def _ovon_target_detection_within_strict_stop(self) -> bool:
        entries: Sequence[Dict[str, Any]] = []
        try:
            entries = self._get_latest_action_local_map_landmark_entries()
        except Exception:
            entries = []

        for entry in entries or []:
            if not self._ovon_text_exactly_matches_goal(entry.get("name")):
                continue
            distance_m = self._ovon_float(entry.get("distance_m"))
            if distance_m is None:
                continue
            if distance_m <= float(self.STRICT_GLOBAL_STOP_DISTANCE_M):
                return True
        return False

    def _ovon_strict_stop_distance_satisfied(self) -> Tuple[bool, str]:
        distance_to_goal = self._ovon_distance_to_goal_m()
        if distance_to_goal is not None:
            if distance_to_goal <= float(self.STRICT_GLOBAL_STOP_DISTANCE_M):
                return True, ""
            return (
                False,
                f"distance_to_goal={distance_to_goal:.2f}m is greater than "
                f"strict stop threshold {float(self.STRICT_GLOBAL_STOP_DISTANCE_M):.2f}m",
            )

        if self._ovon_target_detection_within_strict_stop():
            return True, ""

        return False, "no target-object distance evidence within strict stop threshold"

    def _ovon_validate_global_landmark_arrival(
        self,
        response: Optional[Dict[str, Any]],
        *,
        require_flag: bool = True,
        log_rejection: bool = False,
    ) -> bool:
        if not response:
            return False
        if require_flag and not bool(response.get("global_landmark_arrival", False)):
            return False

        names_ok, name_reason = self._ovon_response_names_target_object(response)
        if not names_ok:
            if log_rejection and bool(response.get("global_landmark_arrival", False)):
                print(f"[OVONStrictStop] Reject global_landmark_arrival=true: {name_reason}")
            return False

        space_ok, space_reason = self._ovon_response_has_plausible_space_context(response)
        if not space_ok:
            if log_rejection and bool(response.get("global_landmark_arrival", False)):
                print(f"[OVONStrictStop] Reject global_landmark_arrival=true: {space_reason}")
            return False

        distance_ok, distance_reason = self._ovon_strict_stop_distance_satisfied()
        if not distance_ok:
            if log_rejection and bool(response.get("global_landmark_arrival", False)):
                print(f"[OVONStrictStop] Reject global_landmark_arrival=true: {distance_reason}")
            return False

        return True

    def _sanitize_planner_response(self, response: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        response = super()._sanitize_planner_response(response)
        if not response:
            return response

        valid_arrival = self._ovon_validate_global_landmark_arrival(
            response,
            require_flag=True,
            log_rejection=True,
        )
        response["global_landmark_arrival"] = bool(valid_arrival)
        response.pop("global_task_finish", None)
        return response

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
        response["global_landmark_arrival"] = self._ovon_validate_global_landmark_arrival(
            response,
            require_flag=True,
            log_rejection=True,
        )
        response["global_task_finish"] = (
            str(mode).strip().lower() != "initial"
            and bool(response.get("global_landmark_arrival", False))
        )
        return super()._apply_thinking_cycle_result(
            response=response,
            cycle_info=cycle_info,
            mode=mode,
        )

    def _should_autostop_from_goal_distance(self) -> bool:
        """OVON global STOP must be planner-confirmed, target-exact, and within 0.5m."""
        current_subtask = getattr(self, "current_subtask", None)
        if not isinstance(current_subtask, dict):
            return False
        if not bool(current_subtask.get("global_landmark_arrival", False)):
            return False
        return self._ovon_validate_global_landmark_arrival(
            current_subtask,
            require_flag=True,
            log_rejection=False,
        )

    def _should_autocomplete_subtask_during_action_step(
        self,
        step_landmark_entries: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        candidate_names = self._get_current_subtask_autocomplete_candidates()
        if not candidate_names:
            return None

        final_object_stage = self._ovon_is_final_object_stage()
        effective_topk = 1 if final_object_stage else int(self.ACTION_SUBTASK_AUTOCOMPLETE_TOPK)
        ordered_entries = [
            dict(entry)
            for entry in self._sort_action_landmark_entries(step_landmark_entries or [])
        ]

        if final_object_stage:
            ordered_entries = [
                entry
                for entry in ordered_entries
                if self._ovon_text_exactly_matches_goal(entry.get("name"))
            ]
            candidate_names = [name for name in candidate_names if self._ovon_text_exactly_matches_goal(name)]
            if not candidate_names:
                goal_name = self._ovon_goal_object_name()
                if goal_name:
                    candidate_names = [goal_name]

        matches: List[Dict[str, Any]] = []
        for entry in ordered_entries[: max(1, effective_topk)]:
            if not self._entry_reaches_action_arrival_threshold(
                entry,
                candidate_names=candidate_names,
            ):
                continue
            matches.append({
                "name": str(entry.get("name") or candidate_names[0]),
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
            if final_object_stage:
                return None

            dest_room, dest_object = self._parse_subtask_destination()
            subtask_landmark = self._normalize_landmark_candidate(
                self._get_subtask_landmark_field(getattr(self, "current_subtask", None))
            )
            if not self._current_area_matches_stair_destination(dest_room, dest_object, subtask_landmark):
                return None

            relaxed_matches: List[Dict[str, Any]] = []
            for entry in ordered_entries[: max(1, effective_topk)]:
                if not self._entry_reaches_action_arrival_threshold(
                    entry,
                    candidate_names=candidate_names,
                ):
                    continue
                relaxed_matches.append({
                    "name": str(entry.get("name") or candidate_names[0]),
                    "distance_m": float(entry.get("distance_m")),
                    "confidence": float(entry.get("confidence", 0.0) or 0.0),
                    "angle_deg": entry.get("angle_deg"),
                    "is_opening_like": bool(self._is_opening_like_landmark_entry(entry)),
                    "stop_distance_m": float(self._autocomplete_stop_distance_m(entry)),
                    "structure_matched": True,
                    "source": "vis" if str(entry.get("source", "mem") or "mem") == "vis" else "mem",
                    "display_id": self._safe_int(entry.get("display_id")),
                    "instance_idx": self._safe_int(entry.get("instance_idx")),
                    "class_total": self._safe_int(entry.get("class_total")),
                    "selection_rank": self._safe_int(entry.get("selection_rank")),
                    "instance_uid": self._safe_int(entry.get("instance_uid")),
                })

            if not relaxed_matches:
                return None

            relaxed_matches.sort(
                key=lambda item: (
                    float(item.get("distance_m", 1e9)),
                    -float(item.get("confidence", 0.0)),
                    str(item.get("name", "")),
                )
            )
            return relaxed_matches[0]

        matches.sort(
            key=lambda item: (
                float(item.get("distance_m", 1e9)),
                -float(item.get("confidence", 0.0)),
                str(item.get("name", "")),
            )
        )
        return matches[0]
