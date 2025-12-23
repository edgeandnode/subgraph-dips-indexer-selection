import logging
from types import MappingProxyType
from typing import Optional, TypedDict, cast

import numpy as np
import pandas as pd

from ..typing import DeploymentId, IndexerId

# Module-level logger
logger = logging.getLogger(__name__)

NON_ZERO_UPTIME_SUCCESS_RATE_SCORE_THRESHOLD = 0.97


class WeightsDict(TypedDict, total=False):
    """
    A dictionary containing weights for each metric used in the weighted score calculation.
    """

    lat_lin_reg_coefficient: float
    uptime_score: float
    existing_dips_agreements: float
    stake_to_fees_iqr_deviation: float
    success_rate: float
    avg_sync_duration: float
    indexing_agreement_acceptance_latency: float
    # "initial/ongoing_sync_price": float # 0.09 <- future weight, above weights will change slightly when implemented


DEFAULT_WEIGHTS = cast(
    WeightsDict,
    MappingProxyType(
        {
            "lat_lin_reg_coefficient": 0.2424,
            "uptime_score": 0.1667,
            "existing_dips_agreements": 0.1212,
            "stake_to_fees_iqr_deviation": 0.1023,
            "success_rate": 0.0625,
            "avg_sync_duration": 0.0625,
            "indexing_agreement_acceptance_latency": 0.2424,
        }
    ),
)


class DataProcessor:
    """
    DataProcessor is responsible for processing the data from the DataManager class,
    including score calculations, normalization of scores, using custom weightings
    to get an overall weighted score, and selecting the best indexers for subgraphs.
    It also handles indexers that are blocked, replacing under-performing indexers,
    and periodically optimizing indexer groups based on quality of service.

    This class has a job lifetime, meaning it is instantiated and used for the specific
    task of adding or replacing an indexer from being assigned to a subgraph_id, then it dies.
    After death the class can be re-instantiated again immediately to add or replace another
    indexer from being assigned to another subgraph_id. This class will figure out weather to
    add or replace an indexer depending on the number of existing indexers serving data on the
    subgraph in question and the quality of the existing indexers serving data on a subgraph compared
    to the best alternative.
    """

    def __init__(
        self,
        history: pd.DataFrame,
        deployment_id: DeploymentId,
        existing_agreements: Optional[dict[DeploymentId, list[IndexerId]]] = None,
        pending_agreements: Optional[dict[DeploymentId, list[IndexerId]]] = None,
        declined_indexers: Optional[dict[DeploymentId, IndexerId]] = None,
        blocklist: Optional[list[IndexerId]] = None,
        weights: Optional[WeightsDict] = None,
    ):
        """
        Initialize the DataProcessor class with data, deployment ID, existing agreements,
        and an indexer blocklist.
        """
        # Initialize class variables with provided parameters
        self.data = pd.DataFrame(history)
        self.deployment_id = deployment_id
        self.existing_agreements = existing_agreements or {}
        self.pending_agreements = pending_agreements or {}
        self.declined_indexers = declined_indexers or {}
        self.blocklist = blocklist or []
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

        # Process the data, we can then call update_blacklist_cancel_indexing_agreements,
        # or get_indexer_selections later after this constructor has finished running.
        self._process_data()

    def update_blocklist_cancel_indexing_agreements(self, blocklist):
        """
        Cancels all outstanding indexing agreements for indexers on the blacklist.

        Note:
        - This method does not currently attempt to reassign indexers to the subgraph after
          cancellation of the indexing agreement from the blocked indexers. Instead we can loop through
          all of the subgraphs while calling the process_subgraph function. Which will detect when a subgraph
          has less than the threshold number of indexers assigned to it and reassign an appropriate indexer.
          We would do this loop at frequent intervals anyway, because it will be important to reassign indexing
          agreements to high quality indexers after an indexers quality has slipped based on their updated
          weighted_score. # TODO we could address the above note, as if all indexers on a subgraph got
          blocked simultaneously, there could be a longer than necessary latency while we reassign new indexers.
        - Although it would likely take some time for new indexers to accept the agreements and finish syncing, so this
          additional latency while we wait for the for loop to get to the subgraph, might not be a huge issue.

        :param blocklist: A list of indexers that have been blocked.
        :return: A dictionary where keys are blocked indexers and values are lists of subgraphs
                 from which they were removed.
        """
        #
        self.blocklist = blocklist

        cancelled_agreements = {}

        for subgraph, indexers in self.existing_agreements.items():
            for indexer in indexers:
                if indexer in blocklist:
                    # If indexer not already in cancelled_agreements, create new key-value
                    if indexer not in cancelled_agreements:
                        cancelled_agreements[indexer] = []
                    # Add subgraphs that the blocked indexer will be cancelled from receiving DIPS for.
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
                - added_dict: A dictionary where the key is the subgraph_id and the value is a list of newly added indexers
                - cancelled_dict: A dictionary where the key is the subgraph_id and the value is a list of removed indexers
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
        # Update the number of existing agreements for each indexer
        self.data = self._fetch_number_of_indexer_agreements()

        # Get the current group of indexers for the subgraph using '_get_current_group'
        self.current_group = self._get_current_group()
        self.initial_group = list(self.current_group)

        # Normalize metrics and calculate scores
        self.data = self._normalize_and_score()

        # Call _assign_indexers_to_subgraph to assign/replace/remove an indexer on the subgraph.
        self._assign_indexers_to_subgraph()

    def _fetch_number_of_indexer_agreements(self):
        """
        Fetch and update the number of existing agreements for each indexer based on current assignments.

        This method updates the 'existing_dips_agreements' field in the df to reflect the number of
        current agreements each indexer has, as specified in the existing_agreements attribute passed by the rust server.
        """
        agreement_counts = {}
        # Count the occurrences of each indexer in existing agreements
        for subgraph_indexers in self.existing_agreements.values():
            for indexer in subgraph_indexers:
                if indexer in agreement_counts:
                    agreement_counts[indexer] += 1
                else:
                    agreement_counts[indexer] = 1

        # Update 'existing_dips_agreements' for all indexers at once
        self.data["existing_dips_agreements"] = (
            self.data["indexer"].map(agreement_counts).fillna(0).astype(int)
        )

        return self.data

    def _get_current_group(self):
        """
        Get the current group of indexers assigned to a subgraph (data from self.existing_agreements).

        :return: A list containing the indexer assigned to 'self.subgraph_id', or an empty list if no indexer is assigned.
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
            normalized_data = _normalize_metrics(self.data)
        except Exception as e:
            logger.error(
                f"Unexpected error when trying normalize_metrics(self.data): {e}"
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
        # If the current indexer group has less than 3 indexers, call '_add_indexers_to_group'
        if len(self.current_group) < 3:
            self._add_indexers_to_group()

        # If the current indexer group has more than 3 indexers, call '_remove_indexers_from_group'
        if len(self.current_group) > 3:
            self._remove_indexers_from_group()

        # Otherwise, call '_replace_underperforming_indexers' which will search for a suitable replacement
        if len(self.current_group) == 3:
            self._replace_underperforming_indexers()

    def _add_indexers_to_group(self):
        """
        Add indexers to the group to meet the required number of indexers.
        """
        # While the group has less than 3 indexers, select the best indexer to add using _find_best_replacement_or_select_best_indexer
        while len(self.current_group) < 3:
            next_indexer = self._find_best_replacement_or_select_best_indexer()

            # Add the best indexer to the group
            if next_indexer:
                self.current_group.append(next_indexer)

            # If there are no indexers available, do nothing.
            else:
                break

    def _meets_decentralization_requirements(self, new_indexer):
        """
        Check if adding the new indexer meets decentralisation requirements.

        This method is called either when adding indexers to a group with less than 3 indexers,
        or when finding a replacement for an existing indexer in a group of 3 or more.

        The final group must have at least 2 unique organizations and 2 unique locations.
        """
        # If the current group has fewer than 2 indexers, no decentralisation check is needed.
        if len(self.current_group) < 2:
            return True

        # Create a new group including the new indexer
        new_group = self.current_group + [new_indexer]

        # Get unique locations and organizations for the new group
        locations = self.data[self.data["indexer"].isin(new_group)][
            "destination_loc"
        ].unique()
        orgs = self.data[self.data["indexer"].isin(new_group)]["org"].unique()

        # Return 'True' if decentralisation requirements are hit
        if len(locations) >= 2 and len(orgs) >= 2:
            return True

        # Otherwise 'False'
        return False

    def _remove_indexers_from_group(self):
        """
        Remove the worst indexers from the current group until the current group only has 3 indexers.
        """
        while len(self.current_group) > 3:
            indexer_scores = []
            for indexer in self.current_group:
                # Calculate each indexers score as if the indexer had 1 less indexing agreement
                score = self._calculate_indexer_score(indexer)
                indexer_scores.append((indexer, score))

            # Sort indexers by score, worst (lowest score) first
            indexer_scores.sort(key=lambda x: x[1], reverse=False)

            for indexer, _ in indexer_scores:
                temp_group = self.current_group.copy()
                temp_group.remove(indexer)

                if self._meets_decentralization_requirements_indexer_removal(
                    temp_group
                ):
                    self.current_group.remove(indexer)
                    break
            else:
                break

    def _calculate_indexer_score(self, indexer):
        """
        Calculate the score for an individual indexer as if they had one less indexing agreement.
        """
        # Check if the indexer exists in self.data
        indexer_data = self.data[self.data["indexer"] == indexer]

        if indexer_data.empty:
            logger.warning(
                f"Indexer {indexer} not found in self.data. Returning lowest possible score."
            )
            return 0

        # Create a copy of the data for this indexer
        indexer_data = indexer_data.copy()

        # Reduce the indexer's agreement count by 1
        indexer_data["existing_dips_agreements"] = (
            indexer_data["existing_dips_agreements"] - 1
        ).clip(lower=0)

        # Normalize only the necessary metrics for this indexer
        normalized_data = _normalize_metrics(indexer_data)

        # Calculate the weighted score for this indexer
        score = _calculate_weighted_score(normalized_data.iloc[0], self.weights)

        return score

    def _meets_decentralization_requirements_indexer_removal(self, group):
        """
        Check if the group meets decentralisation requirements after removing an indexer.

        The group must have at least 2 unique organizations and 2 unique locations.
        """
        if len(group) < 2:
            return False

        locations = self.data[self.data["indexer"].isin(group)][
            "destination_loc"
        ].unique()
        orgs = self.data[self.data["indexer"].isin(group)]["org"].unique()

        # Return 'True' if decentralisation requirements are hit
        if len(locations) >= 2 and len(orgs) >= 2:
            return True

        # Otherwise 'False'
        return False

    def _replace_underperforming_indexers(self):
        """
        Replace underperforming indexers if the group score can be improved by more than 10%.
        This method updates the current_group but does not modify the DataFrame, as the
        DataProcessor instance is short-lived and the DataFrame state isn't used after processing.
        """
        worst_indexer = None
        worst_score_improvement = None
        best_replacement = None

        # For each indexer in the current group
        for indexer in self.current_group:
            # Check the most appropriate replacement indexer to replace the indexer in question.
            new_indexer = self._find_best_replacement_or_select_best_indexer()

            if new_indexer:
                # Create a temp copy of the current group, remove the old indexer from it, add the new indexer.
                temp_group = self.current_group.copy()
                temp_group.remove(indexer)
                temp_group.append(new_indexer)

                # Calculate group score of old group as if the removed indexer had 1 less indexing agreement.
                group_score_before = self._calculate_group_score(
                    self.current_group, indexer_to_exclude=indexer
                )

                # Calculate group score of new group as if the replacement indexer had 1 more indexing agreement.
                group_score_after = self._calculate_group_score(
                    temp_group, indexer_to_include=new_indexer
                )

                # Calculate how much better the new group is than the old group.
                score_improvement = group_score_after - group_score_before

                # If new group is >= 10% better than old group
                if score_improvement >= group_score_before * 0.1:
                    # And score improvement is the best available, take note of the indexer to be replaced
                    # and the indexer to do the replacement.
                    if (
                        worst_score_improvement is None
                        or score_improvement > worst_score_improvement
                    ):
                        worst_score_improvement = score_improvement
                        worst_indexer = indexer
                        best_replacement = new_indexer

        # Once the best replacement has been found, remove old indexer from group & add new indexer to group.
        if best_replacement and worst_indexer:
            self.current_group.remove(worst_indexer)
            self.current_group.append(best_replacement)

    def _find_best_replacement_or_select_best_indexer(self):
        """
        This function is used when either:

            - Finding the best replacement for an indexer in the current group.
              (assuming the group has reached capacity)

            - Selecting the best indexer to add to the current group.
              (assuming the group capacity has not yet been reached)

        Will not attempt to assign an indexing agreement to an indexer under the following conditions:
        1. The indexer is already in the current group.
        2. The indexer is blocked.
        3. The indexer has pending agreements that they have not yet accepted.
        4. The indexer has previously declined an indexing agreement for this subgraph.

        Note:
        - declined_indexers is intended to contain only those indexers that declined within the last
        x days (x=10 seems like a good starting point) and which subgraph they declined.

        Example of declined_indexers structure:
        {
            "subgraph1": ["indexer1", "indexer2"],
            "subgraph2": ["indexer1"]
        }
        In the example above we would not attempt to offer an indexing agreement to:
            - indexer1 for either subgraph1 or subgraph2.
            - indexer2 for subgraph1

        :return: The best indexer, or None if no suitable candidate is found.
        """

        def flatten_list_of_lists(list_of_lists):
            """
            In the context being used here:
            - This function returns a list of indexers that have pending agreements.
            """
            flattened_list = []
            for sublist in list_of_lists:
                for item in sublist:
                    flattened_list.append(item)
            return flattened_list

        unpickable_indexers = set(
            self.current_group
            + self.blocklist
            + flatten_list_of_lists(self.pending_agreements.values())
            + self.declined_indexers.get(self.deployment_id, [])
        )

        # The candidates we could select are those that are not unpickable
        candidates = self.data[~self.data["indexer"].isin(unpickable_indexers)].copy()

        # Sort the candidates by weighted score, highest score first.
        candidates.sort_values(by="weighted_score", ascending=False, inplace=True)

        # Iterate through the list of candidates, return the first (best) candidate that meets decentralization requirements
        for indexer in candidates["indexer"]:
            if self._meets_decentralization_requirements(indexer):
                return indexer

        return None

    def _calculate_group_score(
        self, group, indexer_to_exclude=None, indexer_to_include=None
    ):
        """
        Temporarily adjust the number of indexing agreements for specified indexers and calculate
        the average weighted score of the new indexer group.

        This method is intended to have only one of [indexer_to_exclude, indexer_to_include] passed
        into it at a time, at most.
        """
        try:
            if indexer_to_exclude:
                # Temporarily adjust the data to reflect the indexer losing an agreement
                self.data.loc[
                    self.data["indexer"] == indexer_to_exclude,
                    "existing_dips_agreements",
                ] -= 1
                self._recalculate_metrics_and_scores()

            if indexer_to_include:
                # Temporarily adjust the data to reflect the indexer gaining an agreement
                self.data.loc[
                    self.data["indexer"] == indexer_to_include,
                    "existing_dips_agreements",
                ] += 1
                self._recalculate_metrics_and_scores()

            # Calculate the average weighted score of the new indexer group
            score = self.data[self.data["indexer"].isin(group)]["weighted_score"].mean()

        finally:
            if indexer_to_exclude:
                # Revert the temporary change
                self.data.loc[
                    self.data["indexer"] == indexer_to_exclude,
                    "existing_dips_agreements",
                ] += 1
                self._recalculate_metrics_and_scores()

            if indexer_to_include:
                # Revert the temporary change
                self.data.loc[
                    self.data["indexer"] == indexer_to_include,
                    "existing_dips_agreements",
                ] -= 1
                self._recalculate_metrics_and_scores()

        return score

    def _recalculate_metrics_and_scores(self):
        """
        Helper method to recalculate metrics and scores.
        """
        self.data = _normalize_metrics(self.data)
        self.data["weighted_score"] = self.data.apply(
            _calculate_weighted_score, axis=1, weights=self.weights
        )


def _normalize_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize various metrics in the merged DataFrame to create comparable scores across different dimensions.

    This function takes the merged DataFrame containing various indexer metrics and normalizes them,
    to create standardized scores. It handles different types of metrics, applying appropriate
    normalization techniques for each.

    Note:
    - Each metric is normalized to a scale of 0 to 1, where 1 represents better performance.
    - Some metrics are inverted (1 - normalized value) if lower values are better (e.g., latency).
    - The function handles missing data by assigning a neutral score of 0.5 to NaN values.
    - Different normalization techniques are used based on the nature of each metric:
        - Generic min-max normalization for most metrics
        - Special normalization for uptime and success rate to emphasize high performance
        - Logistic function for acceptance latency

    :param merged: The input DataFrame containing various indexer metrics.
    :return: The input DataFrame with additional columns for normalized metrics:
        - 'norm_lat_lin_reg_coefficient': Normalized latency linear regression coefficient
        - 'norm_uptime_score': Normalized uptime score
        - 'norm_existing_dips_agreements': Normalized score for existing DIP agreements
        - 'norm_stake_to_fees_iqr_deviation': Normalized stake-to-fees ratio deviation
        - 'norm_success_rate': Normalized success rate
        - 'norm_avg_sync_duration': Normalized average sync duration
        - 'norm_indexing_agreement_acceptance_latency': Normalized acceptance latency
    """
    if merged.empty:
        new_columns = [
            "norm_lat_lin_reg_coefficient",
            "norm_uptime_score",
            "norm_existing_dips_agreements",
            "norm_stake_to_fees_iqr_deviation",
            "norm_success_rate",
            "norm_avg_sync_duration",
            "norm_indexing_agreement_acceptance_latency",
        ]
        for col in new_columns:
            merged[col] = pd.Series(dtype=float)
        return merged

    # Normalise latency linear regression score
    if "Latency Coefficient + Error Confidence Interval" in merged.columns:
        merged["norm_lat_lin_reg_coefficient"] = 1 - _normalize_generic(
            merged["Latency Coefficient + Error Confidence Interval"]
        )  # lower is better
    else:
        merged["norm_lat_lin_reg_coefficient"] = np.nan

    # Normalise uptime score
    if "% up_x" in merged.columns:
        merged["norm_uptime_score"] = _normalize_uptime_and_success_rate(
            merged["% up_x"]
        )  # higher is better
    else:
        merged["norm_uptime_score"] = np.nan

    # Normalise the number of indexing agreements each indexer has
    if "existing_dips_agreements" in merged.columns:
        merged["norm_existing_dips_agreements"] = 1 - _normalize_generic(
            merged["existing_dips_agreements"]
        )  # lower is better
    else:
        merged["norm_existing_dips_agreements"] = np.nan

    # Normalise stake to fees ratio
    if "stake_to_fees_iqr_deviation" in merged.columns:
        merged["norm_stake_to_fees_iqr_deviation"] = _normalize_generic(
            merged["stake_to_fees_iqr_deviation"]
        )  # higher is better
    else:
        merged["norm_stake_to_fees_iqr_deviation"] = np.nan

    # Normalise success rate score
    if "average_status" in merged.columns:
        merged["norm_success_rate"] = _normalize_uptime_and_success_rate(
            merged["average_status"]
        )  # higher is better
    else:
        merged["norm_success_rate"] = np.nan

    # Normalize avg_sync_duration
    if "avg_sync_duration" in merged.columns:
        merged["norm_avg_sync_duration"] = 1 - _normalize_generic(
            merged["avg_sync_duration"]
        )  # lower is better
    else:
        merged["norm_avg_sync_duration"] = np.nan

    # Normalize indexing_agreement_acceptance_latency
    if "indexing_agreement_acceptance_latency" in merged.columns:
        merged["norm_indexing_agreement_acceptance_latency"] = (
            _normalize_indexing_agreement_acceptance_latency(
                merged["indexing_agreement_acceptance_latency"]
            )
        )  # lower is better
    else:
        merged["norm_indexing_agreement_acceptance_latency"] = np.nan

    # Fill NaN values with 0 for all norm_ columns except norm_indexing_agreement_acceptance_latency
    norm_columns = [
        col
        for col in merged.columns
        if col.startswith("norm_")
        and col != "norm_indexing_agreement_acceptance_latency"
    ]
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


def _normalize_indexing_agreement_acceptance_latency(
    latency_series: pd.Series,
    l: float = 1.002,  # noqa: E741
    k: float = 1,
    x0: float = 6,
) -> pd.Series:
    """
    Normalize indexing agreement acceptance latency using a piecewise function:
    logistic for x ≤ x0, linear for x > x0.

    Note:
    - Indexing agreement acceptance latency should be measured in hours to 2 d.p, not minutes or seconds.
    - Lower latency results in higher normalized values.
    - Negative latency values are clipped to 0 before normalization.
    - Large latency values are clipped to a maximum of 8 hours, after this the score is 0 anyway.

    :param latency_series: The input series containing latency data in hours.
    :param l: The logistic function's maximum value. Defaults to 1.002.
    :param k: The steepness of the curve. Defaults to 1.
    :param x0: The x-value of the sigmoid's midpoint. Defaults to 6 hours.
    :return: A new series with normalized values between 0 and 1.
    """

    def logistic(x):
        """
        This function creates the smooth transition from high scores
        for low latencies to low scores for high latencies.

        x: time in hours
        """
        return l / (1 + np.exp(k * (x - x0)))

    # x0 is the midpoint of the logistic function, we need to find the gradient of the slope through that point
    def slope_at_x0():
        """
        Calculate the slope of the logistic function at x0.
        """
        h = 1e-6
        return (logistic(x0 + h) - logistic(x0 - h)) / (2 * h)

    m = slope_at_x0()

    def piecewise_function(x):
        """
        Apply a piecewise function: logistic for x ≤ x0, linear for x > x0.
        """
        return np.where(x <= x0, logistic(x), logistic(x0) + m * (x - x0))

    # Replace negative values with 0 (as negative latency doesn't make sense)
    latency_series = latency_series.clip(lower=0)

    # Configure max input latency and clip the series so all values are <= the max value.
    max_latency = 8
    clipped_latency = np.clip(latency_series, 0, max_latency)

    # Apply the piecewise function to normalize acceptance latency
    normalized = pd.Series(piecewise_function(clipped_latency)).round(3)

    # Handle NaN's
    normalized = normalized.fillna(0)

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
        logger.error(
            "Total sum of weights is 0. Sum of weights should be non-zero, ideally 1."
        )
        raise ValueError("Total weight cannot be 0.")

    return weighted_sum / weight_total
