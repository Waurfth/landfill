"""Sentiment contagion, status dynamics, and knowledge spreading."""

from __future__ import annotations

from numpy.random import Generator

from village_sim.core.config import (
    SENTIMENT_CONTAGION_RATE,
    SOCIAL_INFLUENCE_RADIUS,
    STATUS_FROM_AGE,
    STATUS_FROM_SKILL,
    STATUS_FROM_SOCIAL,
    STATUS_FROM_WEALTH,
)


class InfluenceSystem:
    """Handles sentiment contagion, status calculation, and knowledge spread."""

    def spread_sentiment(
        self,
        villagers: list["Villager"],  # noqa: F821
        relationships: "RelationshipManager",  # noqa: F821
    ) -> None:
        """
        After social interactions, sentiment spreads through the social network.

        Each villager's sentiment is pulled toward the weighted average of their
        social contacts' sentiment. The pull strength depends on emotional stability.
        """
        # Pre-compute sentiment values (don't update in-place during iteration)
        sentiment_updates: dict[int, float] = {}

        alive = [v for v in villagers if v.is_alive]
        villager_map = {v.id: v for v in alive}

        for v in alive:
            rels = relationships.get_all_for(v.id)
            if not rels:
                continue

            weighted_sentiment = 0.0
            total_weight = 0.0

            for rel in rels:
                other_id = rel.villager_b_id if rel.villager_a_id == v.id else rel.villager_a_id
                other = villager_map.get(other_id)
                if other is None:
                    continue

                # Weight by relationship strength and proximity
                dist = abs(v.current_position[0] - other.current_position[0]) + \
                       abs(v.current_position[1] - other.current_position[1])
                if dist > SOCIAL_INFLUENCE_RADIUS:
                    continue

                weight = max(0.0, rel.affinity + rel.trust) * rel.familiarity
                if weight > 0:
                    weighted_sentiment += other.current_sentiment * weight
                    total_weight += weight

            if total_weight > 0:
                social_avg = weighted_sentiment / total_weight
                # Pull toward social average; magnitude depends on emotional stability
                stability = v.traits.emotional_stability / 100.0
                pull = SENTIMENT_CONTAGION_RATE * (1.0 - stability * 0.7)
                delta = (social_avg - v.current_sentiment) * pull
                # Sociability amplifies influence
                delta *= 0.5 + 0.5 * (v.traits.sociability / 100.0)
                sentiment_updates[v.id] = delta

        # Apply updates
        for vid, delta in sentiment_updates.items():
            v = villager_map.get(vid)
            if v is not None:
                v.current_sentiment = max(0, min(100, v.current_sentiment + delta))

    def calculate_status(
        self,
        villager: "Villager",  # noqa: F821
        family_wealth: float,
        relationships: "RelationshipManager",  # noqa: F821
        max_wealth: float = 1.0,
    ) -> float:
        """
        Calculate a villager's social status (0-1).

        Based on wealth, skill, social connections, and age.
        """
        # Wealth component
        wealth_score = min(1.0, family_wealth / max(1.0, max_wealth)) * STATUS_FROM_WEALTH

        # Skill component (average of top 3 skills)
        skills = list(villager.memory.skill_experience.values())
        skills.sort(reverse=True)
        top_skills = skills[:3] if skills else [0]
        avg_skill = sum(top_skills) / len(top_skills)
        skill_score = min(1.0, avg_skill / 50.0) * STATUS_FROM_SKILL

        # Social component (number and quality of relationships)
        rels = relationships.get_all_for(villager.id)
        social_score = 0.0
        if rels:
            avg_affinity = sum(max(0, r.affinity) for r in rels) / len(rels)
            social_score = min(1.0, avg_affinity * 2 + len(rels) / 20.0) * STATUS_FROM_SOCIAL

        # Age component (peaks around 40-55)
        age = villager.age_years
        if age < 20:
            age_score = age / 20.0 * 0.3
        elif age < 40:
            age_score = 0.3 + 0.7 * ((age - 20) / 20.0)
        elif age < 60:
            age_score = 1.0
        else:
            age_score = max(0.5, 1.0 - (age - 60) / 30.0)
        age_score *= STATUS_FROM_AGE

        return wealth_score + skill_score + social_score + age_score

    def spread_knowledge(
        self,
        source: "Villager",  # noqa: F821
        target: "Villager",  # noqa: F821
        relationships: "RelationshipManager",  # noqa: F821
        rng: Generator,
    ) -> bool:
        """
        Transfer knowledge during social interaction.
        Returns True if knowledge was successfully transferred.
        """
        rel = relationships.get_or_create(source.id, target.id)

        # Success chance based on traits and relationship
        chance = (
            0.1
            + 0.2 * (source.traits.sociability / 100.0)
            + 0.2 * (source.traits.empathy / 100.0)
            + 0.2 * (target.traits.intelligence / 100.0)
            + 0.2 * max(0, rel.trust)
            + 0.1 * rel.familiarity
        )

        if rng.random() > chance:
            return False

        # Try to transfer a random knowledge type
        topics = ["resource", "recipe", "medicinal"]
        rng.shuffle(topics)
        for topic in topics:
            if target.memory.learn_from(
                source.memory, topic,
                target.traits.intelligence,
                target.traits.sociability,
                max(0, rel.trust),
            ):
                return True
        return False

    def spread_opinion(
        self,
        source: "Villager",  # noqa: F821
        target: "Villager",  # noqa: F821
        about_id: int,
        opinion_value: float,
        relationships: "RelationshipManager",  # noqa: F821
        rng: Generator,
    ) -> None:
        """Spread an opinion about a third villager (gossip)."""
        rel_source_target = relationships.get_or_create(source.id, target.id)
        if rel_source_target.trust < 0.1:
            return  # Target doesn't trust source enough to believe gossip

        # Influence on target's opinion of the subject
        influence = (
            rel_source_target.trust * 0.5
            + rel_source_target.familiarity * 0.3
            + (source.traits.sociability / 100.0) * 0.2
        )

        if rng.random() < influence:
            rel_target_subject = relationships.get_or_create(target.id, about_id)
            shift = opinion_value * influence * 0.3
            rel_target_subject.affinity = max(
                -1.0, min(1.0, rel_target_subject.affinity + shift)
            )
