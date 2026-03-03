# Cognitive Decay Matrix (CDM) Engine

## The Flaw in EdTech
Most AI tutors assume static mastery, ignoring the biological reality of memory decay. Once a student "learns" a topic, conventional systems consider it permanently mastered, failing to account for how knowledge naturally degrades over time without reinforcement.

## Our First Principles Solution
We built a deterministic Math/DAG Engine decoupled from a stateless local LLM renderer. The system relies on hard mathematical principles to track and predict knowledge retention, using the LLM exclusively as a stateless rendering engine to construct educational content and evaluate free-text student answers based on the deterministic engine's state.

## The Physics
The CDM engine explicitly uses the **Ebbinghaus exponential decay formula** to calculate real-time memory weights dynamically:

$$ W_{current} = W_{initial} \times e^{(-\lambda \times \Delta t)} $$

Where $\lambda$ is the decay rate constant and $\Delta t$ represents the elapsed time. This produces a live decayed weight computed on the fly that actively controls the student's path and progression through the Knowledge Graph.

## Reverse DFS Diagnosis
When a student fails on an advanced concept, the system does not arbitrarily assign blame. Instead, it executes a **Reverse Depth-First Search (DFS) Diagnosis**. It traverses all transitive prerequisites of the failed node, computing live weights on the fly. Any foundational node whose decay has crossed the critical threshold is flagged as a "Collapsed Foundation". This provides an exact deterministic, mathematical diagnosis of the root cause of the failure, allowing the system to prioritise relearning that missing prerequisite.

## How to Test (The Time Machine)
To demonstrate the physics of the engine without waiting weeks, the dashboard includes a **+30 days debug button**. This acts as a "Time Machine", subtracting exactly 30 days of elapsed time from the timestamps in the database to instantly simulate physical memory degradation.

By observing the live topological visualizer, clicking this button will show healthy nodes rapidly deteriorating and their actual statuses updating to Collapsed Foundations in real-time.
