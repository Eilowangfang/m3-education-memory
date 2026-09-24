from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .database import connect, initialize
from .memory_writer import rebuild_mastery_states
from .time_simulation import SIMULATION_SEED_VERSION, simulated_time


def rebase_simulated_times(
    db_path: str | Path,
    *,
    anchor_date: date | str,
    timezone_name: str = "Asia/Shanghai",
    seed: str = "m3-education-memory",
) -> dict:
    """Rebase imported FERMAT attempts to seven months ending at anchor_date."""
    anchor = date.fromisoformat(anchor_date) if isinstance(anchor_date, str) else anchor_date
    connection = connect(db_path)
    initialize(connection)
    rows = connection.execute(
        """SELECT attempt_id,dataset_revision,shard,row_index,grade,Time
           FROM attempts
           WHERE shard IS NOT NULL AND row_index IS NOT NULL"""
    ).fetchall()
    changed = 0
    with connection:
        for row in rows:
            timestamp = simulated_time(
                grade=row["grade"],
                dataset_revision=row["dataset_revision"],
                shard=row["shard"],
                row_index=row["row_index"],
                anchor_date=anchor,
                timezone=timezone_name,
                seed=seed,
            ).isoformat()
            connection.execute(
                """UPDATE attempts SET Time=?,simulation_seed_version=?
                   WHERE attempt_id=?""",
                (timestamp, SIMULATION_SEED_VERSION, row["attempt_id"]),
            )
            node_id = f"attempt:{row['attempt_id']}"
            node = connection.execute(
                "SELECT properties_json FROM memory_nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if node:
                properties = json.loads(node["properties_json"] or "{}")
                properties["Time"] = timestamp
                connection.execute(
                    "UPDATE memory_nodes SET properties_json=? WHERE node_id=?",
                    (json.dumps(properties, ensure_ascii=False), node_id),
                )
            connection.execute(
                """UPDATE episodic_memories
                   SET content=replace(content,?,?) WHERE attempt_id=?""",
                (row["Time"], timestamp, row["attempt_id"]),
            )
            changed += int(timestamp != row["Time"])

        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('simulation_anchor_date',?)",
            (anchor.isoformat(),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('simulation_seed_version',?)",
            (SIMULATION_SEED_VERSION,),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('timezone',?)",
            (timezone_name,),
        )
        connection.execute("DELETE FROM metadata WHERE key='simulation_year'")

        model_prompts = connection.execute(
            """SELECT DISTINCT model,prompt_version FROM analysis_runs
               WHERE status='completed'"""
        ).fetchall()
        mastery_states = 0
        for item in model_prompts:
            mastery_states += rebuild_mastery_states(
                connection,
                model=item["model"],
                prompt_version=item["prompt_version"],
                change_reason=f"time_rebase:{anchor.isoformat()}",
            )

    bounds = connection.execute(
        "SELECT MIN(Time) AS earliest,MAX(Time) AS latest FROM attempts"
    ).fetchone()
    connection.close()
    return {
        "anchor_date": anchor.isoformat(),
        "attempts": len(rows),
        "changed": changed,
        "earliest": bounds["earliest"],
        "latest": bounds["latest"],
        "mastery_states": mastery_states,
        "simulation_seed_version": SIMULATION_SEED_VERSION,
    }
