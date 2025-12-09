import numpy as np
from logging import INFO, WARNING
from flwr.common import log, ArrayRecord, Message, ConfigRecord, MetricRecord
from flwr.server import Grid
from typing import Optional, Iterable, Tuple
from strategy_mixin import StrategyMixin
from flwr.serverapp.strategy import (
    FedProx, FedTrimmedAvg, FedMedian, MultiKrum,
    FedAdam, FedAvgM, FedYogi, DifferentialPrivacyServerSideFixedClipping
)

# Custom Strategy Classes that inherit from StrategyMixin and respective strategies
class CustomFedProx(StrategyMixin, FedProx):
    pass
class CustomFedTrimmedAvg(StrategyMixin, FedTrimmedAvg):
    pass
class CustomFedMedian(StrategyMixin, FedMedian):
    pass
class CustomMultiKrum(StrategyMixin, MultiKrum):
    pass
class CustomFedAdam(StrategyMixin, FedAdam):
    pass
class CustomFedAvgM(StrategyMixin, FedAvgM):
    pass
class CustomFedYogi(StrategyMixin, FedYogi):
    pass

# Custom Differential Privacy Strategy with logging of update norms
class CustomDifferentialPrivacyServerSideFixedClipping(StrategyMixin, DifferentialPrivacyServerSideFixedClipping):
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        return super(StrategyMixin, self).configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        
        replies_list = list(replies)
        current_ndarrays = self.current_arrays.to_numpy_ndarrays()
        log(INFO, f"NORME L2 UPDATE ROUND {server_round} (PRE-CLIPPING)")
        update_norms = []

        for r in replies_list:
            if r.has_error():
                continue
            
            record = next(iter(r.content.array_records.values()), None)
            if record is None:
                log(WARNING, f"Client {r.metadata.src_node_id} non ha inviato ArrayRecord.")
                continue
                
            reply_ndarrays = record.to_numpy_ndarrays()
            
            # delta (update) = model_client - model_server
            update_ndarrays = [
                (reply - curr) for reply, curr in zip(reply_ndarrays, current_ndarrays)
            ]
            
            flat_update = np.concatenate([arr.flatten() for arr in update_ndarrays])
            norm = np.linalg.norm(flat_update)
            update_norms.append(norm)
            log(INFO, f"Client {r.metadata.src_node_id}: L2 Norm = {norm:.4f}")

        if update_norms:
            log(INFO, f"Norm stats: "
                        f"Min: {np.min(update_norms):.4f}, "
                        f"Max: {np.max(update_norms):.4f}, "
                        f"Mediana: {np.median(update_norms):.4f}, "
                        f"90-percentile: {np.percentile(update_norms, 90):.4f}")
        log(INFO, "---------------------------------------------")
            
        return DifferentialPrivacyServerSideFixedClipping.aggregate_train(self, server_round, replies_list)