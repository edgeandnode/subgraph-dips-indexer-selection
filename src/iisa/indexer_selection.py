"""
Indexer selection algorithm for IISA.

Selects the best indexers for subgraphs based on weighted scoring of multiple
metrics including latency, uptime, success rate, and economic security.
"""

import logging
from types import MappingProxyType
from typing import NewType, Optional, TypedDict, cast

import numpy as np
import pandas as pd

__all__ = [
    "IndexerSelector",
    "DeploymentId",
    "IndexerId",
    "IpfsHashStr",
    "EthAddressStr",
    "QueryIdStr",
]

# Type aliases for domain concepts
QueryIdStr = NewType("QueryIdStr", str)
IpfsHashStr = NewType("IpfsHashStr", str)
DeploymentId = IpfsHashStr
EthAddressStr = NewType("EthAddressStr", str)
IndexerId = EthAddressStr

# Module-level logger
logger = logging.getLogger(__name__)

NON_ZERO_UPTIME_SUCCESS_RATE_SCORE_THRESHOLD = 0.97

# Indexers scoring below this threshold are candidates for replacement.
# They will only be replaced if a significantly better candidate is available.
MIN_INDEXER_SCORE = 0.15

# Minimum score improvement required to justify replacing an underperforming indexer.
# Candidate must score at least (current_score + REPLACEMENT_MARGIN) to replace.
REPLACEMENT_MARGIN = 0.50


class WeightsDict(TypedDict, total=False):
    """
    A dictionary containing weights for each metric used in the weighted score calculation.
    """

    stake_to_fees: float
    base_price_per_epoch: float
    lat_lin_reg_coefficient: float
    uptime_score: float
    success_rate: float
    price_per_entity: float


DEFAULT_WEIGHTS = cast(
    WeightsDict,
    MappingProxyType(
        {
            "stake_to_fees": 0.30,
            "base_price_per_epoch": 0.25,
            "lat_lin_reg_coefficient": 0.20,
            "uptime_score": 0.15,
            "success_rate": 0.05,
            "price_per_entity": 0.05,
        }
    ),
)


class IndexerSelector:
    """
    IndexerSelector is responsible for processing the data from the DataManager class,
    including score calculations, normalization of scores, using custom weightings
    to get an overall weighted score, and selecting the best indexers for subgraphs.
    It also handles indexers that are blocked, replacing under-performing indexers,
    and periodically optimizing indexer groups based on quality of service.

    This class has a job lifetime, meaning it is instantiated and used for the specific
    task of adding or replacing an indexer from being assigned to a subgraph_id, then it dies.
    After death the class can be re-instantiated again immediately to add or replace another
    indexer from being assigned to another subgraph_id. This class will figure out weather to
    add or replace an indexer depending on the number of existing
    indexers serving data on the subgraph in question and the quality
    of the existing indexers serving data on a subgraph compared to
    the best alternative.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        deployment_id: DeploymentId,
        existing_agreements: Optional[dict[DeploymentId, list[IndexerId]]] = None,
        pending_agreements: Optional[dict[DeploymentId, list[IndexerId]]] = None,
        declined_indexers: Optional[dict[DeploymentId, IndexerId]] = None,
        indexer_denylist: Optional[list[IndexerId]] = None,
        weights: Optional[WeightsDict] = None,
        target_size: int = 3,
        optimistic_dips_fees: Optional[dict[str, float]] = None,
        price_ceiling: Optional[float] = None,
    ):
        """
        Initialize the IndexerSelector class with data, deployment ID, existing agreements,
        and an indexer denylist.

        Args:
            target_size: The target number of indexers to assign to this deployment.
            optimistic_dips_fees: Per-indexer expected DIPs fees in GRT per 30 days,
                keyed by checksummed hex address. Used to adjust stake_to_fees at
                request time so indexers with accepted agreements get deprioritised.
            price_ceiling: Maximum GRT per 30 days that the payer will accept.
                Used as the normalisation ceiling for pricing scores so that
                outlier prices cannot compress legitimate price differentiation.
        """
        # Initialize class variables with provided parameters
        self.data = pd.DataFrame(history)
        self.deployment_id = deployment_id
        self.existing_agreements = existing_agreements or {}
        self.pending_agreements = pending_agreements or {}
        self.declined_indexers = declined_indexers or {}
        self.indexer_denylist = indexer_denylist or []
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.target_size = target_size
        self.optimistic_dips_fees = optimistic_dips_fees or {}
        self.price_ceiling = price_ceiling
        self.current_group: list[IndexerId] = []
        self.initial_group: list[IndexerId] = []

        self._process_data()

        logger.info(
            "IndexerSelector completed: deployment=%s target=%d initial=%d final=%d",
            self.deployment_id,
            self.target_size,
            len(self.initial_group),
            len(self.current_group),
        )

    def update_indexer_denylist_cancel_indexing_agreements(self, indexer_denylist):
        """
        Cancels all outstanding indexing agreements for indexers on the denylist.

        Note:
        - Does not reassign indexers after cancellation. The
          periodic process_subgraph loop detects under-staffed
          subgraphs and reassigns.
          TODO: if all indexers on a subgraph get denied at once
          there may be extra latency before reassignment.
        - New indexers still need time to accept and sync, so
          the loop latency may not matter much in practice.

        :param indexer_denylist: A list of indexers that have been denied.
        :return: A dictionary where keys are denied indexers and values are lists of subgraphs
                 from which they were removed.
        """
        #
        self.indexer_denylist = indexer_denylist

        cancelled_agreements = {}

        for subgraph, indexers in self.existing_agreements.items():
            for indexer in indexers:
                if indexer in indexer_denylist:
                    # If indexer not already in cancelled_agreements, create new key-value
                    if indexer not in cancelled_agreements:
                        cancelled_agreements[indexer] = []
                    # Add subgraphs the blocked indexer loses.
                    cancelled_agreements[indexer].append(subgraph)

        return cancelled_agreements

    def get_indexer_selections(self):
        """
        Returns the indexers that have recently been assigned to or removed from the subgraph.

        This method compares the initial and current groups of indexers to determine
        which indexers have been added or removed.
        Note:
            If no indexers were added or removed, the respective dictionary will be empty.

        :return: A tuple of two dictionaries:
                - added_dict: subgraph_id -> list of added indexers
                - cancelled_dict: subgraph_id -> list of removed indexers
        """
        # Compare initial and current groups to determine changes
        added = set(self.current_group) - set(self.initial_group)
        cancelled = set(self.initial_group) - set(self.current_group)

        # Create dictionaries with subgraph_id as key and list of indexers as value
        added_dict = {self.deployment_id: list(added)} if added else {}
        cancelled_dict = {self.deployment_id: list(cancelled)} if cancelled else {}

        # Return two separate dictionaries
        return added_dict, cancelled_dict

    def _process_data(self):
        """
        Process data by normalizing metrics and calculating weighted scores.
        """
        # Apply optimistic DIPs fees to adjust stake_to_fees before normalisation.
        # When dipper reports expected fees from accepted agreements, we add them
        # to the Redpanda-derived query fees so stake_to_fees can differentiate
        # indexers even before on-chain payment claims appear.
        if self.optimistic_dips_fees and "total_query_fees" in self.data.columns:
            dips_adjustment = self.data["indexer"].map(self.optimistic_dips_fees).fillna(0.0)
            effective_fees = self.data["total_query_fees"].fillna(0.0) + dips_adjustment

            if "last_known_slashable_stake" in self.data.columns:
                self.data["stake_to_fees"] = self.data[
                    "last_known_slashable_stake"
                ] / effective_fees.replace(0.0, float("nan"))
                # Drop pre-normalised column so _normalize_metrics recomputes it
                self.data.drop(
                    columns=["norm_stake_to_fees"],
                    errors="ignore",
                    inplace=True,
                )

            adjusted_count = (dips_adjustment > 0).sum()
            logger.info(
                "deployment=%s applied optimistic DIPs fees to %d/%d indexers",
                self.deployment_id,
                adjusted_count,
                len(self.data),
            )

        # Get the current group of indexers for the subgraph using '_get_current_group'
        self.current_group = self._get_current_group()
        self.initial_group = list(self.current_group)
        logger.debug(
            "deployment=%s current_group=%s (%d indexers)",
            self.deployment_id,
            [addr[:10] for addr in self.current_group],
            len(self.current_group),
        )

        # Normalize metrics and calculate scores
        self.data = self._normalize_and_score()

        if not self.data.empty and "weighted_score" in self.data.columns:
            top = self.data.nlargest(5, "weighted_score")[["indexer", "weighted_score"]]
            logger.debug(
                "deployment=%s top_5_scores: %s",
                self.deployment_id,
                [
                    (row["indexer"][:10], round(row["weighted_score"], 4))
                    for _, row in top.iterrows()
                ],
            )

        # Call _assign_indexers_to_subgraph to assign/replace/remove an indexer on the subgraph.
        self._assign_indexers_to_subgraph()

    def _get_current_group(self):
        """
        Get the current group of indexers assigned to a subgraph.

        :return: Indexers assigned to self.deployment_id, or [].
        """
        # Check if the subgraph_id exists in the agreements and return the corresponding indexers
        return self.existing_agreements.get(self.deployment_id, [])

    def _normalize_and_score(self):
        """
        Normalize metrics assessing indexer quality and calculate weighted scores.

        This method attempts to normalize the data and calculate weighted scores.

        :return: The processed DataFrame.
        """
        try:
            normalized_data = _normalize_metrics(self.data, price_ceiling=self.price_ceiling)
            logger.debug(
                "deployment=%s normalized %d indexers",
                self.deployment_id,
                len(normalized_data),
            )
        except Exception as e:
            logger.error(
                "Unexpected error when trying normalize_metrics(self.data): %s",
                e,
            )
            normalized_data = self.data

        try:
            normalized_data["weighted_score"] = normalized_data.apply(
                lambda row: _calculate_weighted_score(row, self.weights), axis=1
            )
        except Exception as e:
            logger.error(f"Unexpected error when trying calculate_weighted_score: {e}")
            normalized_data["weighted_score"] = np.nan

        return normalized_data

    def _assign_indexers_to_subgraph(self):
        """
        Assign indexers to subgraph based on weighted scores and decentralization requirements.

        Use the methods _add_indexers_to_group and _replace_underperforming_indexers to
        assign indexers to the subgraph in question.
        """
        action = (
            "add"
            if len(self.current_group) < self.target_size
            else "remove"
            if len(self.current_group) > self.target_size
            else "replace_check"
        )
        logger.info(
            "deployment=%s assigning: current_size=%d target_size=%d action=%s",
            self.deployment_id,
            len(self.current_group),
            self.target_size,
            action,
        )
        # Under-staffed: add indexers
        if len(self.current_group) < self.target_size:
            self._add_indexers_to_group()

        # Over-staffed: remove worst indexers
        if len(self.current_group) > self.target_size:
            self._remove_indexers_from_group()

        # At target: check for underperforming replacements
        if len(self.current_group) == self.target_size:
            self._replace_underperforming_indexers()

    def _add_indexers_to_group(self):
        """
        Add indexers to the group to meet the required number of indexers.
        """
        # While the group has less than target_size indexers, select the best indexer to add
        while len(self.current_group) < self.target_size:
            next_indexer = self._find_best_replacement_or_select_best_indexer()

            # Add the best indexer to the group
            if next_indexer:
                self.current_group.append(next_indexer)
                logger.info(
                    "deployment=%s added indexer %s to group (%d/%d)",
                    self.deployment_id,
                    next_indexer[:10],
                    len(self.current_group),
                    self.target_size,
                )

            # If there are no indexers available, do nothing.
            else:
                logger.info(
                    "deployment=%s no more candidates available, group is %d/%d",
                    self.deployment_id,
                    len(self.current_group),
                    self.target_size,
                )
                break

    def _meets_decentralization_requirements(
        self, new_indexer: IndexerId, replacing_indexer: Optional[IndexerId] = None
    ) -> bool:
        """
        Check if adding/replacing an indexer meets decentralisation requirements.

        When adding: checks if current_group + [new_indexer] meets requirements.
        When replacing: checks if (current_group - replacing_indexer)
        + [new_indexer] meets requirements.

        The resulting group must have at least 2 unique organizations and 2 unique locations.
        This check applies when the resulting group would have 2+ indexers.

        Args:
            new_indexer: The indexer being added or used as replacement.
            replacing_indexer: If provided, the indexer being replaced (for swap scenarios).

        Returns:
            True if decentralization requirements are met, False otherwise.
        """
        # Build the resulting group
        if replacing_indexer:
            # Replacement scenario: remove old, add new
            new_group = [i for i in self.current_group if i != replacing_indexer] + [new_indexer]
        else:
            # Addition scenario: just add new
            new_group = self.current_group + [new_indexer]

        # If resulting group has fewer than 2 indexers, no decentralisation check needed
        if len(new_group) < 2:
            return True

        # Get unique locations and organizations for the resulting group
        locations = self.data[self.data["indexer"].isin(new_group)]["destination_loc"].unique()
        orgs = self.data[self.data["indexer"].isin(new_group)]["org"].unique()

        # Return True if decentralisation requirements are met
        meets = len(locations) >= 2 and len(orgs) >= 2
        if not meets:
            logger.debug(
                "deployment=%s decentralization check failed for %s: "
                "locations=%d orgs=%d (need 2 each)",
                self.deployment_id,
                new_indexer[:10],
                len(locations),
                len(orgs),
            )
        return meets

    def _remove_indexers_from_group(self):
        """
        Remove worst indexers until group has target_size indexers.
        """
        while len(self.current_group) > self.target_size:
            indexer_scores = []
            for indexer in self.current_group:
                row = self.data[self.data["indexer"] == indexer]
                score = (
                    row["weighted_score"].iloc[0]
                    if not row.empty and "weighted_score" in row.columns
                    else 0.0
                )
                indexer_scores.append((indexer, score))

            # Sort by score, worst (lowest) first
            indexer_scores.sort(key=lambda x: x[1])

            for indexer, score in indexer_scores:
                temp_group = self.current_group.copy()
                temp_group.remove(indexer)

                if self._meets_decentralization_requirements_indexer_removal(temp_group):
                    self.current_group.remove(indexer)
                    logger.info(
                        "deployment=%s removed worst indexer %s (score=%.4f, group now %d/%d)",
                        self.deployment_id,
                        indexer[:10],
                        score,
                        len(self.current_group),
                        self.target_size,
                    )
                    break
            else:
                break

    def _meets_decentralization_requirements_indexer_removal(self, group):
        """
        Check if the group meets decentralisation requirements after removing an indexer.

        The group must have at least 2 unique organizations and 2 unique locations.
        """
        if len(group) < 2:
            return False

        locations = self.data[self.data["indexer"].isin(group)]["destination_loc"].unique()
        orgs = self.data[self.data["indexer"].isin(group)]["org"].unique()

        # Return 'True' if decentralisation requirements are hit
        if len(locations) >= 2 and len(orgs) >= 2:
            return True

        # Otherwise 'False'
        return False

    def _replace_underperforming_indexers(self):
        """
        Replace indexers scoring below MIN_INDEXER_SCORE when a significantly better
        candidate is available.

        Only considers indexers with weighted_score < MIN_INDEXER_SCORE for replacement.
        A replacement occurs only if the candidate scores at least REPLACEMENT_MARGIN
        higher than the current indexer (e.g., current=0.10, candidate must be >0.60).

        This approach:
        - Keeps indexers performing adequately (>= MIN_INDEXER_SCORE) stable
        - Only replaces poor performers when there's a meaningfully better option
        - Avoids churn from marginal improvements
        - Never leaves gaps (keeps bad indexer if no good replacement exists)

        Iterates until no more beneficial replacements are found.
        """
        # Track indexers added in this call - don't replace them
        added_this_call: set[IndexerId] = set()

        while True:
            best_swap: Optional[tuple[IndexerId, IndexerId, float]] = None

            for existing_indexer in self.current_group:
                # Don't replace indexers we just added in this call
                if existing_indexer in added_this_call:
                    continue

                # Get current indexer's score
                indexer_data = self.data[self.data["indexer"] == existing_indexer]
                if indexer_data.empty:
                    continue

                current_score = indexer_data["weighted_score"].iloc[0]

                # Only consider replacing indexers below the minimum threshold
                if current_score >= MIN_INDEXER_SCORE:
                    logger.debug(
                        "deployment=%s indexer %s score=%.4f "
                        ">= threshold=%.2f, no replacement needed",
                        self.deployment_id,
                        existing_indexer[:10],
                        current_score,
                        MIN_INDEXER_SCORE,
                    )
                    continue

                # Find best candidate for replacing this specific indexer
                candidate = self._find_best_replacement_or_select_best_indexer(
                    replacing_indexer=existing_indexer
                )

                if not candidate:
                    continue

                # Get candidate's score
                candidate_data = self.data[self.data["indexer"] == candidate]
                if candidate_data.empty:
                    continue

                candidate_score = candidate_data["weighted_score"].iloc[0]

                # Only replace if candidate is significantly better
                if candidate_score > current_score + REPLACEMENT_MARGIN:
                    improvement = candidate_score - current_score
                    if best_swap is None or improvement > best_swap[2]:
                        best_swap = (existing_indexer, candidate, improvement)

            if best_swap:
                old_indexer, new_indexer, improvement = best_swap
                self.current_group.remove(old_indexer)
                self.current_group.append(new_indexer)
                added_this_call.add(new_indexer)
                logger.info(
                    "deployment=%s replaced indexer %s with %s (improvement=%.4f)",
                    self.deployment_id,
                    old_indexer[:10],
                    new_indexer[:10],
                    improvement,
                )
            else:
                logger.debug(
                    "deployment=%s no more beneficial replacements found",
                    self.deployment_id,
                )
                break  # No more beneficial replacements available

    def _find_best_replacement_or_select_best_indexer(
        self, replacing_indexer: Optional[IndexerId] = None
    ) -> Optional[IndexerId]:
        """
        Find the best indexer to add to or replace in the current group.

        Used for both:
        - Adding an indexer when group is below target_size (replacing_indexer=None)
        - Finding a replacement for a specific indexer (replacing_indexer=<indexer_id>)

        Will not select an indexer if:
        1. Already in the current group
        2. On the denylist
        3. Has pending agreements not yet accepted
        4. Previously declined an agreement for this subgraph

        Args:
            replacing_indexer: If provided, the indexer being considered for replacement.
                              This affects the decentralization check (simulates the swap).

        Returns:
            The best indexer ID, or None if no suitable candidate exists.
        """

        def flatten_list_of_lists(list_of_lists):
            flattened_list = []
            for sublist in list_of_lists:
                for item in sublist:
                    flattened_list.append(item)
            return flattened_list

        unpickable_indexers = set(
            self.current_group
            + self.indexer_denylist
            + flatten_list_of_lists(self.pending_agreements.values())
            + self.declined_indexers.get(self.deployment_id, [])
        )
        logger.debug(
            "deployment=%s unpickable: %d in_group, %d denylisted, %d pending, %d declined",
            self.deployment_id,
            len(self.current_group),
            len(self.indexer_denylist),
            sum(len(v) for v in self.pending_agreements.values()),
            len(self.declined_indexers.get(self.deployment_id, [])),
        )

        # The candidates we could select are those that are not unpickable
        candidates = self.data[~self.data["indexer"].isin(unpickable_indexers)].copy()

        # Sort the candidates by weighted score, highest score first
        candidates.sort_values(by="weighted_score", ascending=False, inplace=True)
        logger.debug(
            "deployment=%s candidates: %d eligible out of %d total (excluded %d unpickable)",
            self.deployment_id,
            len(candidates),
            len(self.data),
            len(unpickable_indexers),
        )

        # Iterate through candidates, prefer one that meets decentralization requirements
        for indexer in candidates["indexer"]:
            if self._meets_decentralization_requirements(
                indexer, replacing_indexer=replacing_indexer
            ):
                score = candidates[candidates["indexer"] == indexer]["weighted_score"].iloc[0]
                logger.debug(
                    "deployment=%s selected %s (score=%.4f, meets decentralization)",
                    self.deployment_id,
                    indexer[:10],
                    score,
                )
                return indexer

        # Fallback: return best candidate even if it doesn't meet decentralization
        # Decentralization is best-effort - if constraints cannot be met, still return an indexer
        if not candidates.empty:
            fallback = candidates["indexer"].iloc[0]
            fallback_score = candidates["weighted_score"].iloc[0]
            logger.debug(
                "deployment=%s no candidate meets decentralization, "
                "falling back to best scorer %s (score=%.4f)",
                self.deployment_id,
                fallback[:10],
                fallback_score,
            )
            return fallback

        logger.info(
            "deployment=%s zero candidates remain after filtering %d total indexers",
            self.deployment_id,
            len(self.data),
        )
        return None  # Only when truly zero candidates available


def _normalize_metrics(
    merged: pd.DataFrame,
    price_ceiling: Optional[float] = None,
) -> pd.DataFrame:
    """
    Normalize metrics to create comparable scores across dimensions.

    Each metric is normalized to [0, 1] where 1 = better performance.
    Metrics where lower is better (latency, price) are inverted.

    Args:
        merged: DataFrame containing indexer metrics.
        price_ceiling: Maximum GRT/30d the payer will accept. When
            provided, used as the normalisation ceiling for pricing
            scores instead of the observed max price. This prevents
            a single outlier price from compressing legitimate price
            differentiation among other indexers.
    """
    if merged.empty:
        new_columns = [
            "norm_lat_lin_reg_coefficient",
            "norm_uptime_score",
            "norm_stake_to_fees",
            "norm_success_rate",
            "norm_base_price_per_epoch",
            "norm_price_per_entity",
        ]
        for col in new_columns:
            merged[col] = pd.Series(dtype=float)
        return merged

    # Normalise latency linear regression score
    if "norm_lat_lin_reg_coefficient" not in merged.columns:
        if "Latency Coefficient + Error Confidence Interval" in merged.columns:
            merged["norm_lat_lin_reg_coefficient"] = 1 - _normalize_generic(
                merged["Latency Coefficient + Error Confidence Interval"]
            )  # lower is better
        else:
            merged["norm_lat_lin_reg_coefficient"] = np.nan

    # Normalise uptime score
    if "norm_uptime_score" not in merged.columns:
        if "% up_x" in merged.columns:
            merged["norm_uptime_score"] = _normalize_uptime_and_success_rate(
                merged["% up_x"]
            )  # higher is better
        else:
            merged["norm_uptime_score"] = np.nan

    # Normalise stake to fees ratio (higher = more capacity = better).
    # Indexers with zero fees get NaN (infinite ratio = maximum capacity).
    # Fill NaN above the max finite value so they normalise to 1.0.
    if "norm_stake_to_fees" not in merged.columns:
        if "stake_to_fees" in merged.columns:
            stf = merged["stake_to_fees"].copy()
            finite_max = stf.max()
            fill_value = (finite_max + 1.0) if pd.notna(finite_max) else 1.0
            stf = stf.fillna(fill_value)
            merged["norm_stake_to_fees"] = _normalize_generic(stf)
        else:
            merged["norm_stake_to_fees"] = np.nan

    # Normalise success rate score
    if "norm_success_rate" not in merged.columns:
        if "average_status" in merged.columns:
            merged["norm_success_rate"] = _normalize_uptime_and_success_rate(
                merged["average_status"]
            )  # higher is better
        else:
            merged["norm_success_rate"] = np.nan

    # Normalize base price per epoch (lower is better).
    # When price_ceiling is provided, use it instead of the observed max
    # so outlier prices cannot compress differentiation among others.
    if "norm_base_price_per_epoch" not in merged.columns:
        if "base_price_per_epoch" in merged.columns:
            prices = (
                pd.to_numeric(merged["base_price_per_epoch"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0)
            )
            ceiling = (
                price_ceiling if price_ceiling is not None and price_ceiling > 0 else prices.max()
            )
            if ceiling > 0 and ceiling > prices.min():
                merged["norm_base_price_per_epoch"] = (1 - (prices / ceiling)).clip(lower=0)
            else:
                merged["norm_base_price_per_epoch"] = 0.5
        else:
            merged["norm_base_price_per_epoch"] = np.nan

    # Normalize price per entity (lower is better).
    # price_ceiling applies to base epoch price; entity pricing uses
    # observed max since there is no separate budget field for it.
    if "norm_price_per_entity" not in merged.columns:
        if "price_per_entity" in merged.columns:
            prices = (
                pd.to_numeric(merged["price_per_entity"], errors="coerce").fillna(0.0).clip(lower=0)
            )
            ceiling = prices.max()
            if ceiling > 0 and ceiling > prices.min():
                merged["norm_price_per_entity"] = 1 - (prices / ceiling)
            else:
                merged["norm_price_per_entity"] = 0.5
        else:
            merged["norm_price_per_entity"] = np.nan

    # Fill NaN values with 0 for all norm_ columns
    norm_columns = [col for col in merged.columns if col.startswith("norm_")]
    merged[norm_columns] = merged[norm_columns].fillna(0)

    return merged


def _normalize_generic(series: pd.Series) -> pd.Series:
    """
    Perform a generic min-max normalization on a pandas Series.

    This function normalizes the input series to a range between 0 and 1 using min-max scaling.
    It handles edge cases such as constant series or series with NaN values.

    Note:
    - If the input series is empty or contains only one unique value, it returns a series of 0.5.

    :param series: The input series to be normalized.
    :return: A new series with normalized values between 0 and 1.
    """
    min_val = series.min()
    max_val = series.max()

    # Normalize to between 0 and 1 range
    normalized = (series - min_val) / (max_val - min_val)

    # Handle any potential NaN or inf values
    normalized = normalized.fillna(0)

    return normalized


def _normalize_uptime_and_success_rate(series: pd.Series) -> pd.Series:
    """
    Normalize either uptime or success rate data using a piecewise linear scaling method.

    This function applies a custom normalization to uptime / success rate data, emphasizing
    high performance. Uptime between 0% and 97% of the best indexers uptime results in a
    score of 0, while uptime between 97% and 100% of the best indexers uptime results in a
    linear score scaling from 0 to 1. So for example 98.5% of the best indexers uptime would
    result in a normalized score of 0.5. The same calculation applies to success rate.

    :param series: The input series containing uptime or success rate data.
    :returns: A new series with normalized values between 0 and 1.
    """
    # Find the best uptime/success rate score in the series first
    best = series.max()

    # Threshold whereby indexers that have less uptime/success rate than this get no score.
    threshold = best * NON_ZERO_UPTIME_SUCCESS_RATE_SCORE_THRESHOLD

    # Linear score between the threshold and the best.
    normalized = series.apply(
        lambda x: max(
            0,
            min(1, (x - threshold) / (best - threshold)),
        )
    )

    # Reindex and fill NaN's with 0.
    normalized = normalized.reindex(series.index).fillna(0)

    return normalized


def _calculate_weighted_score(row: pd.Series, weights: dict) -> float:
    """
    Calculate a weighted score for an indexer based on multiple normalized metrics.

    This function computes a single score by combining multiple performance metrics,
    each weighted according to predefined weights. NaN values and missing metrics
    are treated as 0, but all weights contribute to the total score.

    :param row: A series containing normalized metric values for an indexer.
                Expected to have columns prefixed with 'norm_'.
    :param weights: A dictionary mapping metric names to their respective weights.
                    Keys should match the suffix of the 'norm_' columns in the row.
    :return: The calculated weighted score.
    :raises ValueError: If the total weight is 0.
    """
    weighted_sum = 0
    weight_total = 0
    missing_columns = []

    for metric, weight in weights.items():
        column_name = f"norm_{metric}"

        # Append any missing columns to the list
        if column_name not in row.index:
            missing_columns.append(column_name)
            continue

        value = row.get(column_name, np.nan)  # Uses np.nan if column is missing

        # So long as the column has a value that isn't nan, then:
        if not pd.isna(value):
            weighted_sum += value * weight
            weight_total += weight

    if missing_columns:
        logger.warning(f"Missing columns in input data: {', '.join(missing_columns)}")

    if weight_total == 0:
        logger.error("Total sum of weights is 0. Sum of weights should be non-zero, ideally 1.")
        raise ValueError("Total weight cannot be 0.")

    return weighted_sum / weight_total
