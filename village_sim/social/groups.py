"""Work parties, social groups, and group management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from numpy.random import Generator


@dataclass
class WorkParty:
    """A temporary group formed for a collaborative activity."""

    leader_id: int
    member_ids: list[int] = field(default_factory=list)
    activity: str = ""
    target_resource_id: Optional[int] = None
    target_position: Optional[tuple[int, int]] = None
    duration_days: int = 1

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def effectiveness(self, villagers_by_id: dict[int, "Villager"]) -> float:  # noqa: F821
        """Group effectiveness based on composition."""
        if not self.member_ids:
            return 0.0
        total_skill = 0.0
        for vid in self.member_ids:
            v = villagers_by_id.get(vid)
            if v is not None:
                total_skill += v.memory.skill_level(self.activity, v.traits.intelligence)
        avg_skill = total_skill / len(self.member_ids)
        # Diminishing returns from group size
        size_bonus = 1.0
        for i in range(1, len(self.member_ids)):
            size_bonus += 0.15 * (0.8 ** i)
        return avg_skill * size_bonus / 100.0


@dataclass
class SocialGroup:
    """A persistent social grouping (friends, council, guild)."""

    group_id: int
    name: str = ""
    member_ids: list[int] = field(default_factory=list)
    purpose: str = ""  # "hunting_party", "council", "friends", "craft_guild"
    influence: float = 0.0

    def add_member(self, villager_id: int) -> None:
        if villager_id not in self.member_ids:
            self.member_ids.append(villager_id)

    def remove_member(self, villager_id: int) -> None:
        if villager_id in self.member_ids:
            self.member_ids.remove(villager_id)


class GroupManager:
    """Manages work parties and social groups."""

    def __init__(self) -> None:
        self.work_parties: list[WorkParty] = []
        self.social_groups: list[SocialGroup] = []
        self._next_group_id: int = 0

    def form_work_party(
        self,
        leader_id: int,
        activity: str,
        member_ids: list[int],
        target_resource_id: Optional[int] = None,
        target_position: Optional[tuple[int, int]] = None,
    ) -> WorkParty:
        """Create a new work party."""
        party = WorkParty(
            leader_id=leader_id,
            member_ids=[leader_id] + [m for m in member_ids if m != leader_id],
            activity=activity,
            target_resource_id=target_resource_id,
            target_position=target_position,
        )
        self.work_parties.append(party)
        return party

    def dissolve_work_party(self, party: WorkParty) -> None:
        if party in self.work_parties:
            self.work_parties.remove(party)

    def resolve_work_parties(
        self,
        villagers: list["Villager"],  # noqa: F821
        decision_engine: "DecisionEngine",  # noqa: F821
        relationships: "RelationshipManager",  # noqa: F821
        rng: Generator,
    ) -> None:
        """
        Resolve work party formation for the day.

        Villagers who chose group activities try to recruit others.
        """
        # Clear yesterday's parties
        self.work_parties.clear()

        villager_map = {v.id: v for v in villagers if v.is_alive}
        assigned: set[int] = set()

        # Find potential leaders (villagers doing group-eligible activities)
        from village_sim.economy.activities import ACTIVITIES

        for v in villagers:
            if not v.is_alive or v.id in assigned:
                continue
            act = ACTIVITIES.get(v.current_activity)
            if act is None or act.min_group_size <= 1:
                continue

            # Try to recruit from friends and nearby villagers
            friends = relationships.get_friends(v.id)
            candidates = [
                vid for vid in friends
                if vid in villager_map
                and vid not in assigned
                and villager_map[vid].is_alive
            ]

            recruited: list[int] = [v.id]
            assigned.add(v.id)

            for cand_id in candidates:
                cand = villager_map[cand_id]
                trust = relationships.get_or_create(v.id, cand_id).trust
                if decision_engine.evaluate_cooperation_request(
                    cand, v, v.current_activity, trust
                ):
                    recruited.append(cand_id)
                    assigned.add(cand_id)
                    cand.current_activity = v.current_activity
                    if len(recruited) >= 5:  # cap group size
                        break

            if len(recruited) >= act.min_group_size:
                self.form_work_party(
                    leader_id=v.id,
                    activity=v.current_activity,
                    member_ids=recruited,
                )

    def get_party_for(self, villager_id: int) -> Optional[WorkParty]:
        """Get the work party a villager belongs to, if any."""
        for party in self.work_parties:
            if villager_id in party.member_ids:
                return party
        return None

    def form_social_group(
        self, founder_id: int, purpose: str, initial_members: list[int]
    ) -> SocialGroup:
        group = SocialGroup(
            group_id=self._next_group_id,
            name=f"{purpose}_{self._next_group_id}",
            member_ids=list(initial_members),
            purpose=purpose,
        )
        self._next_group_id += 1
        self.social_groups.append(group)
        return group
