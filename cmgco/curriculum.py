"""Model-Aware Curriculum Learning module.

Strictly follows Section III-B (Eq. 2-5) of the CMGCO manuscript:
- Phase I (Bootstrap): Full-data bootstrap for epochs 1 through e_0 (default 20)
- Phase II (Active-Set Expansion):
    * Sample readiness score: s_i = (|y_hat_i ∩ y_i| + eps) / (|y_hat_i ∪ y_i| + eps)
    * Monotonic admission: once admitted to active set, never removed
    * Reassessment interval: every K epochs (default K=5)
- Phase III (Full-Data Refinement): At e >= e_all (default 220), force-all admission
"""

from typing import List, Dict, Callable


class ModelAwareCurriculum:
    """Manages sample admission into active training set."""

    def __init__(
        self,
        all_ids: List[str],
        bootstrap_epochs: int = 20,
        readiness_threshold: float = 0.5,
        update_interval: int = 5,
        force_all_epoch: int = 220,
    ):
        if not all_ids:
            raise ValueError("Training split is empty")
        self.all_ids = list(all_ids)
        self.bootstrap_epochs = int(bootstrap_epochs)
        self.readiness_threshold = float(readiness_threshold)
        self.update_interval = int(update_interval)
        self.force_all_epoch = int(force_all_epoch)

        self.initialized = False
        self.active_ids = list(self.all_ids)
        self.pending_ids: List[str] = []
        self.last_update_epoch = None

    def maybe_update(self, epoch: int, readiness_evaluator: Callable[[List[str]], Dict[str, float]]) -> dict:
        """Evaluate readiness and update active set according to current epoch.

        Args:
            epoch: Current training epoch (1-indexed)
            readiness_evaluator: Function that evaluates a list of sample IDs without
                                 data augmentation and returns {sample_id: readiness_score}
        """
        epoch = int(epoch)

        # Phase I: complete the configured number of full-data bootstrap epochs.
        if epoch <= self.bootstrap_epochs:
            if not self.initialized:
                self.active_ids = list(self.all_ids)
                self.pending_ids = []
            return {
                "phase": "bootstrap",
                "updated": False,
                "active_count": len(self.active_ids),
                "pending_count": 0,
            }

        # Phase III: Full-Data Refinement
        if epoch >= self.force_all_epoch:
            changed = len(self.active_ids) != len(self.all_ids)
            self.initialized = True
            self.active_ids = list(self.all_ids)
            self.pending_ids = []
            self.last_update_epoch = epoch
            return {
                "phase": "full_refinement",
                "updated": changed,
                "active_count": len(self.active_ids),
                "pending_count": 0,
            }

        # Phase II: Active-Set Expansion
        # Check if this epoch is an update epoch
        first_update_epoch = self.bootstrap_epochs + 1
        is_update_epoch = (epoch == first_update_epoch) or (
            (epoch - first_update_epoch) % self.update_interval == 0
        )

        if not is_update_epoch:
            return {
                "phase": "active_expansion",
                "updated": False,
                "active_count": len(self.active_ids),
                "pending_count": len(self.pending_ids),
            }

        # Candidate pool to evaluate
        if not self.initialized:
            eval_pool = list(self.all_ids)
        else:
            eval_pool = list(self.pending_ids)

        if not eval_pool:
            return {
                "phase": "active_expansion",
                "updated": False,
                "active_count": len(self.active_ids),
                "pending_count": 0,
            }

        scores = readiness_evaluator(eval_pool)
        admitted = [
            sid for sid in eval_pool if float(scores.get(sid, 0.0)) >= self.readiness_threshold
        ]

        if not self.initialized:
            # Initial active/pending split immediately after e_0 bootstrap epochs.
            self.active_ids = admitted
            admitted_set = set(admitted)
            self.pending_ids = [sid for sid in self.all_ids if sid not in admitted_set]
            self.initialized = True
            if not self.active_ids:
                raise RuntimeError(
                    "The paper-defined readiness threshold admitted no samples at "
                    f"epoch {epoch}; refusing to introduce an undocumented top-k fallback."
                )
        else:
            # Monotonic active-set update: A_e = A_{e-1} ∪ Q_e
            active_set = set(self.active_ids)
            active_set.update(admitted)
            self.active_ids = [sid for sid in self.all_ids if sid in active_set]
            self.pending_ids = [sid for sid in self.all_ids if sid not in active_set]

        self.last_update_epoch = epoch
        return {
            "phase": "active_expansion",
            "updated": True,
            "admitted_this_epoch": len(admitted),
            "active_count": len(self.active_ids),
            "pending_count": len(self.pending_ids),
        }

    def state_dict(self):
        return {
            "initialized": self.initialized,
            "active_ids": list(self.active_ids),
            "pending_ids": list(self.pending_ids),
            "last_update_epoch": self.last_update_epoch,
            "all_ids": list(self.all_ids),
        }

    def load_state_dict(self, state):
        self.initialized = bool(state.get("initialized", False))
        self.active_ids = list(state.get("active_ids", []))
        self.pending_ids = list(state.get("pending_ids", []))
        self.last_update_epoch = state.get("last_update_epoch")

