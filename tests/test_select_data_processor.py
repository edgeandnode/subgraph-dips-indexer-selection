from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from iisa.indexer_selection import (
    DataProcessor,
    DeploymentId,
    IndexerId,
    _calculate_weighted_score,
    _normalize_generic,
    _normalize_metrics,
    _normalize_uptime_and_success_rate,
)


def process_subgraph(
    history,
    deployment_id,
    existing_agreements,
    indexer_denylist=None,
):
    processor = DataProcessor(
        history,
        deployment_id,
        existing_agreements=existing_agreements,
        indexer_denylist=indexer_denylist,
    )
    return processor.get_indexer_selections()


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "indexer": ["A", "B", "C"],
            "deployment_hash": ["hash1", "hash2", "hash3"],
            "score": [0.8, 0.6, 0.7],
        }
    )


@pytest.fixture
def mock__combined_query_results(faker):
    return pd.DataFrame(
        {
            "query_id": [faker.query_id() for _ in range(3)],
            "deployment_hash": [faker.deployment_id() for _ in range(3)],
            "indexer": [faker.indexer_id() for _ in range(3)],
            "indexer_network": ["net1", "net2", "net3"],
            "org": ["hetzner", "amazon aws", "google"],
            "fee": [0.1, 0.2, 0.3],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "blocks_behind": [1, 2, 3],
            "response_time_ms": [100, 200, 300],
            "status": ["200 OK", "200 OK", "200 OK"],
            "day_partition": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "subgraph_network": ["network1", "network2", "network3"],
            "url": [faker.url() for _ in range(3)],
            "origin_loc": ["0,20", "40,40", "60,60"],
            "destination_loc": ["20,40", "40,60", "60,80"],
            "loc": ["0,20", "40,40", "60,60"],
            "distance_miles": [100, 200, 300],
            "sampled_query_id_hashed_mod_integer_root": [0, 1, 2],
        }
    )


@pytest.fixture
def mock__provider(faker, mock__combined_query_results):
    provider = MagicMock()
    provider.return_value.fetch_initial_query_results.return_value = pd.DataFrame(
        {
            "deployment_hash": [faker.deployment_id() for _ in range(3)],
            "indexer": [faker.indexer_id() for _ in range(3)],
            "num_rows": [1000, 2000, 3000],
        }
    )
    provider.return_value.fetch_combined_query_results.return_value = mock__combined_query_results
    provider.return_value.fetch_initial_stake_to_fees.return_value = pd.DataFrame(
        {
            "indexer": [faker.indexer_id() for _ in range(3)],
            "stake_to_fees": [1.0, 2.0, 3.0],
        }
    )
    return provider


class TestProcessSubgraph:
    """
    This class verifies the process_subgraph function creates a DataProcessor
    instance and returns the expected results for added/cancelled indexers.
    """

    @pytest.mark.skip(reason="Flaky test: high dependency on internal details")
    @patch("iisa.indexer_selection.DataProcessor")
    def test_process_subgraph(self, mock__data_processor, sample_data, mock__provider):
        """
        Test process_subgraph creates a DataProcessor and returns expected results.

        Expected results:
        1. processor.added_indexers
        2. processor.cancelled_indexers
        """
        # Set up mock DataProcessor instance
        mock_instance = mock__data_processor.return_value
        mock_instance.added_indexers = [
            ("indexer1", "test_subgraph"),
            ("indexer2", "test_subgraph"),
        ]
        mock_instance.cancelled_indexers = [("indexer3", "test_subgraph")]

        # Define test input parameters
        deployment_id = "test_subgraph"
        existing_agreements = {
            "indexer1": ["subgraph1"],
            "indexer2": ["subgraph2"],
            "indexer3": ["test_subgraph"],
        }
        indexer_denylist = ["indexer_denylisted_indexer"]

        # Process the subgraph
        added, cancelled = process_subgraph(
            sample_data,
            deployment_id,
            existing_agreements,
        )

        # Verify an instance of DataProcessor was created with expected parameters
        mock__data_processor.assert_called_once_with(
            history=sample_data,
            deployment_id=deployment_id,
            existing_agreements=existing_agreements,
            indexer_denylist=indexer_denylist,
            weights=None,
        )

        # Verify the function returns the expected added and cancelled indexer pairs
        assert added == [("indexer1", "test_subgraph"), ("indexer2", "test_subgraph")]
        assert cancelled == [("indexer3", "test_subgraph")]

        # Verify pairs are associated with the expected respective subgraphs
        assert all(pair[1] == deployment_id for pair in added)
        assert all(pair[1] == deployment_id for pair in cancelled)


class TestDataProcessor:
    """
    Unit tests for the DataProcessor class.
    """

    @pytest.fixture
    def sample_data(self):
        """
        Fixture to create a sample DataFrame for testing.
        """
        return pd.DataFrame(
            {
                "indexer": ["A", "B", "C"],
                "deployment_hash": ["hash1", "hash2", "hash3"],
                "score": [0.8, 0.6, 0.7],
                "destination_loc": ["loc1", "loc2", "loc3"],
                "org": ["org1", "org2", "org3"],
                "weighted_score": [0.9, 0.7, 0.8],
                "lat_lin_reg_coefficient": [0.1, 0.2, 0.3],
                "uptime_score": [0.9, 0.8, 0.7],
                "stake_to_fees": [0.1, 0.2, 0.3],
                "success_rate": [0.95, 0.90, 0.85],
                "base_price_per_epoch": [100, 200, 300],
                "price_per_entity": [0.1, 0.2, 0.3],
            }
        )

    @pytest.mark.skip(reason="Flaky test: high dependency on internal details")
    def test_data_processor_constructor(self, sample_data):
        """
        Test the initialization of the DataProcessor class.

        This test verifies:
        1. The constructor correctly sets all instance variables with provided parameters.
        2. Default values are applied when optional parameters are not provided.
        3. The _process_data method is called once.
        4. The indexer_denylist is properly applied.
        5. The 'data' DataFrame maintains its original content, while adding the new columns.
        6. Optional parameters (existing_agreements, indexer_denylist) default empty if not set.

        The test uses mock objects and patch decorators for _process_data
        and derive_timestamps to avoid actual data fetching and ensure consistent test behavior.
        """
        # Define test input parameters
        deployment_id = DeploymentId("test_subgraph")
        existing_agreements = {
            DeploymentId("subgraph1"): [IndexerId("A")],
            DeploymentId("subgraph2"): [IndexerId("B")],
        }
        indexer_denylist = [IndexerId("D")]

        # Create a DataProcessor instance
        processor = DataProcessor(
            history=sample_data,
            deployment_id=deployment_id,
            existing_agreements=existing_agreements,
            indexer_denylist=indexer_denylist,
        )

        # Verify that all instance variables are set correctly
        assert processor.deployment_id == deployment_id
        assert processor.existing_agreements == existing_agreements
        assert processor.indexer_denylist == indexer_denylist

        # Verify default values for optional parameters
        processor_default = DataProcessor(
            history=sample_data,
            deployment_id=deployment_id,
        )
        assert processor_default.existing_agreements == {}
        assert processor_default.indexer_denylist == []

    @pytest.mark.parametrize(
        "initial_group, current_group, expected_added, expected_cancelled",
        [
            (
                ["A", "B"],  # initial_group
                ["A", "C"],  # current_group
                {"test_subgraph": ["C"]},  # expected_added
                {"test_subgraph": ["B"]},  # expected_cancelled
            ),
            (
                [],  # initial_group
                ["A", "B"],  # current_group
                {"test_subgraph": ["A", "B"]},  # expected_added
                {},  # expected_cancelled (no cancellations)
            ),
            (
                ["A", "B", "C"],  # initial_group
                [],  # current_group
                {},  # expected_added (no additions)
                {"test_subgraph": ["A", "B", "C"]},  # expected_cancelled
            ),
            (
                ["A", "B"],  # initial_group
                ["A", "B"],  # current_group
                {},  # expected_added (no additions)
                {},  # expected_cancelled (no cancellations)
            ),
            (
                ["A"],  # initial_group
                ["B"],  # current_group
                {"test_subgraph": ["B"]},  # expected_added
                {"test_subgraph": ["A"]},  # expected_cancelled
            ),
        ],
    )
    def test_get_indexer_selections(
        self,
        sample_data,
        initial_group,
        current_group,
        expected_added,
        expected_cancelled,
        mock__provider,
    ):
        """
        This test verifies the get_indexer_selections method correctly identifies the
        recent added and cancelled indexers.
        """
        with patch("iisa.indexer_selection.DataProcessor._process_data"):
            # Create a DataProcessor instance
            processor = DataProcessor(
                history=sample_data,
                deployment_id=DeploymentId("test_subgraph"),
            )

        processor.initial_group = initial_group
        processor.current_group = current_group

        # Call the method under test
        added, cancelled = processor.get_indexer_selections()

        # Sort the lists within the dictionaries
        added_sorted = {k: sorted(v) for k, v in added.items()}
        cancelled_sorted = {k: sorted(v) for k, v in cancelled.items()}
        expected_added_sorted = {k: sorted(v) for k, v in expected_added.items()}
        expected_cancelled_sorted = {k: sorted(v) for k, v in expected_cancelled.items()}

        # Verify the results by comparing sorted dictionaries
        assert added_sorted == expected_added_sorted, (
            f"Expected added: {expected_added_sorted}, but got: {added_sorted}"
        )
        assert cancelled_sorted == expected_cancelled_sorted, (
            f"Expected cancelled: {expected_cancelled_sorted}, but got: {cancelled_sorted}"
        )

    def test_get_indexer_selections_empty_groups(self, sample_data, mock__provider):
        """
        Test get_indexer_selections method when both initial_group and current_group are empty.

        This test verifies that the method handles the scenario where both the initial_group
        and current_group are empty (represented as an empty list and an empty set respectively).
        It ensures that the method returns empty lists for both added and cancelled indexers
        when there are no indexers in either group.
        """
        with patch("iisa.indexer_selection.DataProcessor._process_data"):
            processor = DataProcessor(
                history=sample_data,
                deployment_id=DeploymentId("test_subgraph"),
            )

        processor.initial_group = []
        processor.current_group = set()

        added, cancelled = processor.get_indexer_selections()

        # Verify that no indexers were added or cancelled.
        assert added == {}
        assert cancelled == {}

    @pytest.fixture
    def processor(self, sample_data, mock__provider):
        return DataProcessor(
            history=sample_data,
            deployment_id=DeploymentId("test_subgraph"),
        )

    def test_get_current_group_normal_case(self, processor):
        """
        Test _get_current_group with multiple indexers assigned to the subgraph.
        """
        processor.existing_agreements = {
            "test_subgraph": ["A", "B", "D"],
            "other_subgraph": ["A", "C"],
            "another_subgraph": ["D"],
        }
        result = processor._get_current_group()
        expected = ["A", "B", "D"]
        assert set(result) == set(expected)

    def test_get_current_group_no_assigned_indexers(self, processor):
        """
        Test _get_current_group when no indexers are assigned to the subgraph.
        """
        processor.existing_agreements = {
            "A": ["other_subgraph"],
            "B": ["another_subgraph"],
            "C": ["yet_another_subgraph"],
        }
        result = processor._get_current_group()
        assert result == []

    def test_get_current_group_empty_agreements(self, processor):
        """
        Test _get_current_group with empty existing_agreements.
        """
        processor.existing_agreements = {}
        result = processor._get_current_group()
        assert result == []

    def test_get_current_group_subgraph_not_in_agreements(self, processor, mock__provider):
        """
        Test _get_current_group when the subgraph 'test_subgraph' is not in any agreement.
        """
        processor.existing_agreements = {
            "A": ["other_subgraph1", "other_subgraph2"],
            "B": ["other_subgraph3", "other_subgraph4"],
        }
        result = processor._get_current_group()
        assert result == []

    @patch("iisa.indexer_selection._normalize_metrics")
    @patch("iisa.indexer_selection._calculate_weighted_score")
    def test_normalize_and_score(
        self, mock_calculate_score, mock_normalize, sample_data, mock__provider
    ):
        """
        Test the _normalize_and_score method.

        This test verifies that:
        1. The method calls normalize_metrics with the correct input.
        2. It applies calculate_weighted_score to each row of the normalized data.
        3. The resulting DataFrame contains a 'weighted_score' column with expected values.
        4. The method handles the data flow correctly, passing results between functions.
        5. The weights used in calculate_weighted_score match the expected structure.
            - They are passed as a dictionary
            - They contain all expected metric keys
            - The sum of weights is approximately 1.0
        6. The number and type of arguments passed to calculate_weighted_score are correct.
        7. The method produces the expected output structure and values.

        Note: This test does not verify specific weight values or exception handling for
        normalization and score calculation, as these are implementation details that may change.
        """
        # Create a DataProcessor instance
        with patch("iisa.indexer_selection.DataProcessor._process_data"):
            processor = DataProcessor(
                history=sample_data,
                deployment_id=DeploymentId("test_subgraph"),
            )

        # Set up mock return values
        normalized_data = sample_data.copy()
        for metric in [
            "stake_to_fees",
            "base_price_per_epoch",
            "lat_lin_reg_coefficient",
            "uptime_score",
            "success_rate",
            "price_per_entity",
        ]:
            normalized_data[f"norm_{metric}"] = normalized_data.get(metric, 0.5)
        mock_normalize.return_value = normalized_data
        mock_calculate_score.return_value = 0.8

        # Call the _normalize_and_score method
        result = processor._normalize_and_score()

        # Verify normalize_metrics was called with correct input
        mock_normalize.assert_called_once()
        pd.testing.assert_frame_equal(mock_normalize.call_args[0][0], sample_data)

        # Verify calculate_weighted_score was called for each row
        assert mock_calculate_score.call_count == len(sample_data)

        # Check weights structure
        for call_args in mock_calculate_score.call_args_list:
            args, kwargs = call_args
            assert len(args) == 2
            assert isinstance(args[1], dict)
            weights = args[1]
            expected_metrics = [
                "stake_to_fees",
                "base_price_per_epoch",
                "lat_lin_reg_coefficient",
                "uptime_score",
                "success_rate",
                "price_per_entity",
            ]
            assert all(metric in weights for metric in expected_metrics)
            assert pytest.approx(sum(weights.values())) == 1.0

        # Verify 'weighted_score' column exists and contains expected values
        assert "weighted_score" in result.columns
        expected_scores = pd.Series(
            [0.8] * len(sample_data), name="weighted_score", index=result.index
        )
        pd.testing.assert_series_equal(result["weighted_score"], expected_scores)

    @pytest.mark.skip(reason="Flaky test: high dependency on internal details")
    def test_assign_indexers_to_subgraph(self, sample_data, mock__provider):
        """
        Test the _assign_indexers_to_subgraph method of DataProcessor.

        This test verifies:
        1. The method calls _add_indexers_to_group when there are fewer than 3 indexers.
        2. The method calls _replace_underperforming_indexers when there are 3 or more indexers.
        """
        with patch("iisa.indexer_selection.DataProcessor._add_indexers_to_group") as mock_add:
            with patch(
                "iisa.indexer_selection.DataProcessor._replace_underperforming_indexers)"
            ) as mock_replace:
                processor = DataProcessor(
                    history=sample_data,
                    deployment_id=DeploymentId("test_subgraph"),
                )

                # Test with fewer than 3 indexers
                processor.current_group = ["A", "B"]
                processor._assign_indexers_to_subgraph()
                assert mock_add.call_count > 0
                mock_replace.assert_not_called()

                # Reset mocks
                mock_add.reset_mock()
                mock_replace.reset_mock()

                # Test with 3 or more indexers
                processor.current_group = ["A", "B", "C"]
                processor._assign_indexers_to_subgraph()
                mock_add.assert_not_called()
                mock_replace.assert_called_once()

    @pytest.mark.parametrize(
        "initial_group, expected_calls, expected_final_group",
        [
            (
                [],  # initial_group
                3,  # expected_calls
                ["B", "C", "D"],  # expected_final_group
            ),
            (
                ["A"],  # initial_group
                2,  # expected_calls
                ["A", "B", "C"],  # expected_final_group
            ),
            (
                ["A", "B"],  # initial_group
                1,  # expected_calls
                ["A", "B", "B"],  # expected_final_group
            ),
            (
                ["A", "B", "C"],  # initial_group
                0,  # expected_calls
                ["A", "B", "C"],  # expected_final_group
            ),
        ],
    )
    def test_add_indexers_to_group(
        self,
        sample_data,
        initial_group,
        expected_calls,
        expected_final_group,
        mock__provider,
    ):
        """
        Test the _add_indexers_to_group method of DataProcessor.

        This test verifies:
        1. The method adds indexers to the group until there are 3 indexers in the group.
        2. The method stops adding indexers if no suitable candidates are found.
        3. The method behaves correctly with different initial group sizes.
        """
        processor = DataProcessor(
            history=sample_data,
            deployment_id=DeploymentId("test_subgraph"),
        )

        with patch(
            "iisa.indexer_selection.DataProcessor._find_best_replacement_or_select_best_indexer"
        ) as mock_select:
            mock_select.side_effect = ["B", "C", "D", None]
            processor.current_group = initial_group.copy()

            processor._add_indexers_to_group()

            assert processor.current_group == expected_final_group
            assert mock_select.call_count == expected_calls

            # Check intermediate states
            for i in range(expected_calls):
                mock_select.assert_any_call()

        # Test when no suitable indexers are found
        with patch(
            "iisa.indexer_selection.DataProcessor._find_best_replacement_or_select_best_indexer",
            return_value=None,
        ):
            processor.current_group = ["A"]
            processor._add_indexers_to_group()
            assert processor.current_group == ["A"]

    def test_meets_decentralization_requirements(self, mock__provider):
        """
        Test the _meets_decentralization_requirements method of DataProcessor.

        This test verifies:
        1. Resulting group with < 2 indexers always passes (no check needed).
        2. Resulting group with 2+ indexers needs 2+ unique locations AND 2+ unique orgs.
        3. The replacing_indexer parameter correctly simulates swap scenarios.

        Note:
        _meets_decentralization_requirements accepts new_indexer and optional replacing_indexer.
        """
        processor = DataProcessor(
            history=pd.DataFrame(
                {
                    "indexer": ["A", "B", "C", "D"],
                    "destination_loc": ["loc1", "loc1", "loc2", "loc3"],
                    "org": ["org1", "org1", "org2", "org3"],
                }
            ),
            deployment_id=DeploymentId("test_subgraph"),
        )

        # Test adding first indexer (resulting group has 1 indexer - no check needed)
        processor.current_group = []
        assert processor._meets_decentralization_requirements("A")

        # Test adding second indexer - same location and org (fails decentralization)
        processor.current_group = ["A"]
        assert not processor._meets_decentralization_requirements("B")  # A,B both loc1/org1

        # Test adding second indexer - different location and org (passes)
        processor.current_group = ["A"]
        assert processor._meets_decentralization_requirements("C")  # A=loc1/org1, C=loc2/org2

        # Test with 2 indexers, adding third with different location and org
        processor.current_group = ["A", "B"]
        assert processor._meets_decentralization_requirements("C")  # C adds loc2 and org2

        # Test with 2 indexers that already meet requirements, adding any third is fine
        processor.current_group = ["A", "C"]
        assert processor._meets_decentralization_requirements("D")  # Adds loc3 and org3
        assert processor._meets_decentralization_requirements("B")  # Already have 2 locs/orgs

        # Test replacement scenario: replacing A (loc1/org1) with C (loc2/org2) in group [A, B]
        # Results in [B, C] = loc1/org1 + loc2/org2 = 2 locs, 2 orgs (passes)
        processor.current_group = ["A", "B"]
        assert processor._meets_decentralization_requirements("C", replacing_indexer="A")

        # Test replacement scenario: replacing C (loc2/org2) with B (loc1/org1) in group [A, C]
        # Results in [A, B] = loc1/org1 + loc1/org1 = 1 loc, 1 org (fails)
        processor.current_group = ["A", "C"]
        assert not processor._meets_decentralization_requirements("B", replacing_indexer="C")

        # Test with duplicate indexer (edge case)
        processor.current_group = ["A", "A"]
        assert not processor._meets_decentralization_requirements("A")

    def test_meets_decentralization_requirements_edge_cases(self, mock__provider):
        """
        Test _meets_decentralization_requirements with various edge cases.
        """
        processor = DataProcessor(
            history=pd.DataFrame(
                {
                    "indexer": ["A", "B", "C", "D", "E", "F"],
                    "destination_loc": ["loc1", "loc1", "loc2", "loc2", "loc3", "loc3"],
                    "org": ["org1", "org2", "org1", "org2", "org3", "org1"],
                }
            ),
            deployment_id=DeploymentId("test_subgraph"),
        )

        # Test with empty current group (adding first indexer)
        processor.current_group = []
        assert processor._meets_decentralization_requirements("A")

        # Test adding second indexer with different org (A=loc1/org1, B=loc1/org2)
        # Same location but different org - fails (needs 2 locs AND 2 orgs)
        processor.current_group = ["A"]
        assert not processor._meets_decentralization_requirements("B")  # Same loc

        # Test adding second indexer with different location and org
        processor.current_group = ["A"]
        assert processor._meets_decentralization_requirements("D")  # A=loc1/org1, D=loc2/org2

        # Test with indexer 'A' selected twice due to some error, adding diverse indexer
        processor.current_group = ["A", "A"]
        assert processor._meets_decentralization_requirements("D")  # D=loc2/org2 adds diversity

        # Test with many indexers already in group (decentralization already met)
        processor.current_group = ["A", "B", "C", "D", "E", "F"]
        assert processor._meets_decentralization_requirements("F")

        # Test adding same indexer that's already in group (fails - duplicates don't add diversity)
        processor.current_group = ["A", "B"]  # loc1/org1 + loc1/org2 = 1 loc, 2 orgs (fails)
        assert not processor._meets_decentralization_requirements("A")

        # Test that two diverse indexers pass
        processor.current_group = ["A", "D"]  # loc1/org1 + loc2/org2 = 2 locs, 2 orgs
        assert processor._meets_decentralization_requirements("E")  # Adds more diversity

    def test_replace_underperforming_indexers_replaces_low_scorer(self, mock__provider):
        """
        Test replacement when indexer scores below MIN_INDEXER_SCORE and
        candidate exceeds current + REPLACEMENT_MARGIN.
        """
        history = pd.DataFrame(
            {
                "indexer": ["A", "B", "C", "D"],
                "destination_loc": ["loc1", "loc2", "loc3", "loc4"],
                "org": ["org1", "org2", "org3", "org4"],
            }
        )
        processor = DataProcessor(
            history=history,
            deployment_id=DeploymentId("test_subgraph"),
        )

        # Manually set weighted_score after initialization
        # A=0.10 (below MIN_INDEXER_SCORE=0.15), D=0.70 (> 0.10 + 0.50 = 0.60)
        processor.data.loc[processor.data["indexer"] == "A", "weighted_score"] = 0.10
        processor.data.loc[processor.data["indexer"] == "B", "weighted_score"] = 0.50
        processor.data.loc[processor.data["indexer"] == "C", "weighted_score"] = 0.50
        processor.data.loc[processor.data["indexer"] == "D", "weighted_score"] = 0.70

        processor.current_group = ["A", "B", "C"]
        processor._replace_underperforming_indexers()

        # A (0.10) should be replaced with D (0.70) since 0.70 > 0.10 + 0.50
        assert "D" in processor.current_group
        assert "A" not in processor.current_group
        assert len(processor.current_group) == 3

    def test_replace_underperforming_indexers_keeps_adequate_performers(self, mock__provider):
        """
        Test that indexers scoring >= MIN_INDEXER_SCORE are not replaced,
        even if better candidates exist.
        """
        history = pd.DataFrame(
            {
                "indexer": ["A", "B", "C", "D"],
                "destination_loc": ["loc1", "loc2", "loc3", "loc4"],
                "org": ["org1", "org2", "org3", "org4"],
            }
        )
        processor = DataProcessor(
            history=history,
            deployment_id=DeploymentId("test_subgraph"),
        )

        # A=0.20 >= MIN_INDEXER_SCORE, so not eligible for replacement
        processor.data.loc[processor.data["indexer"] == "A", "weighted_score"] = 0.20
        processor.data.loc[processor.data["indexer"] == "B", "weighted_score"] = 0.50
        processor.data.loc[processor.data["indexer"] == "C", "weighted_score"] = 0.50
        processor.data.loc[processor.data["indexer"] == "D", "weighted_score"] = 0.90

        processor.current_group = ["A", "B", "C"]
        processor._replace_underperforming_indexers()

        # No replacement - all indexers are above MIN_INDEXER_SCORE (0.15)
        assert processor.current_group == ["A", "B", "C"]

    def test_replace_underperforming_indexers_margin_not_met(self, mock__provider):
        """
        Test that no replacement occurs when candidate doesn't exceed
        current + REPLACEMENT_MARGIN.
        """
        history = pd.DataFrame(
            {
                "indexer": ["A", "B", "C", "D"],
                "destination_loc": ["loc1", "loc2", "loc3", "loc4"],
                "org": ["org1", "org2", "org3", "org4"],
            }
        )
        processor = DataProcessor(
            history=history,
            deployment_id=DeploymentId("test_subgraph"),
        )

        # A=0.10 (below threshold), D=0.55 (< 0.10 + 0.50 = 0.60)
        processor.data.loc[processor.data["indexer"] == "A", "weighted_score"] = 0.10
        processor.data.loc[processor.data["indexer"] == "B", "weighted_score"] = 0.50
        processor.data.loc[processor.data["indexer"] == "C", "weighted_score"] = 0.50
        processor.data.loc[processor.data["indexer"] == "D", "weighted_score"] = 0.55

        processor.current_group = ["A", "B", "C"]
        processor._replace_underperforming_indexers()

        # No replacement - D (0.55) doesn't exceed A (0.10) + REPLACEMENT_MARGIN (0.50)
        assert processor.current_group == ["A", "B", "C"]

    def test_replace_underperforming_indexers_multiple_swaps(self, mock__provider):
        """
        Test iterative replacement when multiple indexers are below threshold.
        """
        # Need 5 indexers with diverse locations/orgs
        history = pd.DataFrame(
            {
                "indexer": ["A", "B", "C", "D", "E"],
                "destination_loc": ["loc1", "loc2", "loc3", "loc4", "loc5"],
                "org": ["org1", "org2", "org3", "org4", "org5"],
            }
        )
        processor = DataProcessor(
            history=history,
            deployment_id=DeploymentId("test_subgraph"),
        )

        # A=0.05, B=0.08 (both below MIN_INDEXER_SCORE)
        # D=0.80 > 0.05+0.50, E=0.75 > 0.08+0.50
        processor.data.loc[processor.data["indexer"] == "A", "weighted_score"] = 0.05
        processor.data.loc[processor.data["indexer"] == "B", "weighted_score"] = 0.08
        processor.data.loc[processor.data["indexer"] == "C", "weighted_score"] = 0.50
        processor.data.loc[processor.data["indexer"] == "D", "weighted_score"] = 0.80
        processor.data.loc[processor.data["indexer"] == "E", "weighted_score"] = 0.75

        processor.current_group = ["A", "B", "C"]
        processor._replace_underperforming_indexers()

        # A and B should be replaced with D and E
        assert "A" not in processor.current_group
        assert "B" not in processor.current_group
        assert "C" in processor.current_group
        assert len(processor.current_group) == 3

    def test_replace_underperforming_indexers_skips_newly_added(self, mock__provider):
        """
        Test that newly added indexers are not eligible for replacement in the same call.
        """
        history = pd.DataFrame(
            {
                "indexer": ["A", "B", "C", "D", "E"],
                "destination_loc": ["loc1", "loc2", "loc3", "loc4", "loc5"],
                "org": ["org1", "org2", "org3", "org4", "org5"],
            }
        )
        processor = DataProcessor(
            history=history,
            deployment_id=DeploymentId("test_subgraph"),
        )

        # A=0.05 (below threshold), D and E are good replacements
        processor.data.loc[processor.data["indexer"] == "A", "weighted_score"] = 0.05
        processor.data.loc[processor.data["indexer"] == "B", "weighted_score"] = 0.50
        processor.data.loc[processor.data["indexer"] == "C", "weighted_score"] = 0.50
        processor.data.loc[processor.data["indexer"] == "D", "weighted_score"] = 0.80
        processor.data.loc[processor.data["indexer"] == "E", "weighted_score"] = 0.85

        processor.current_group = ["A", "B", "C"]
        processor._replace_underperforming_indexers()

        # A replaced with best candidate (D or E), newly added indexer not re-evaluated
        assert "A" not in processor.current_group
        assert len(processor.current_group) == 3

    def test_find_best_replacement_or_select_best_indexer(self, mock__provider):
        """
        Test the _find_best_replacement_or_select_best_indexer method of DataProcessor.

        This test verifies:
        1. The method returns the best replacement that meets decentralization requirements.
        2. The method returns None when no suitable replacement is found.
        3. The method skips indexers already on the denylist.
        """
        processor = DataProcessor(
            history=pd.DataFrame(
                {
                    "indexer": ["A", "B", "C", "D", "E"],
                    "weighted_score": [0.9, 0.8, 0.7, 0.6, 0.5],
                    "destination_loc": ["loc1", "loc2", "loc3", "loc4", "loc5"],
                    "org": ["org1", "org2", "org3", "org4", "org5"],
                }
            ),
            deployment_id=DeploymentId("test_subgraph"),
        )

        processor.current_group = ["A", "B", "C"]
        processor.indexer_denylist = ["E"]

        with patch(
            "iisa.indexer_selection.DataProcessor._meets_decentralization_requirements"
        ) as mock_decentralization:
            mock_decentralization.side_effect = [True]

            result = processor._find_best_replacement_or_select_best_indexer()

            # Verify the best replacement is D, not E, due to indexer_denylisting.
            assert result == "D"

            # Verify the number of decentralization requirement checks
            assert mock_decentralization.call_count == 1

    def test_update_indexer_denylist_cancel_indexing_agreements(self, sample_data, mock__provider):
        """
        Test the update_indexer_denylist_cancel_indexing_agreements method of DataProcessor.

        This test verifies:
        1. Correctly identifies agreements to cancel based on the denylist.
        2. The method returns the correct dictionary of cancelled agreements.
        3. The method updates the internal indexer_denylist of the DataProcessor.
        """
        # Initialize DataProcessor
        processor = DataProcessor(
            history=sample_data,
            deployment_id=DeploymentId("test_subgraph"),
            existing_agreements={
                "subgraph1": ["A"],
                "subgraph2": ["A", "B"],
                "subgraph3": ["B"],
                "subgraph4": ["A", "D"],
                "subgraph5": ["B"],
                "subgraph6": ["F"],
                "subgraph7": ["A"],
                "subgraph9": ["B", "F"],
                "subgraph10": ["A", "C"],
                "subgraph11": ["E"],
                "subgraph12": ["B", "E"],
                "subgraph14": ["E"],
                "subgraph15": ["E"],
                "subgraph16": ["F"],
                "subgraph20": ["C"],
                "subgraph23": ["F"],
                "subgraph40": ["C"],
                "subgraph41": ["F"],
                "subgraph45": ["F"],
                "subgraph70": ["C"],
                "subgraph100": ["C"],
            },
            indexer_denylist=[IndexerId("H")],
        )

        # update the indexer_denylist to cancel agreements
        new_indexer_denylist = [
            IndexerId("H"),
            IndexerId("B"),
            IndexerId("E"),
            IndexerId("NOT_IN_LIST"),
        ]

        # Call update_indexer_denylist_cancel_indexing_agreements with new new_indexer_denylist
        newly_cancelled_agreements = processor.update_indexer_denylist_cancel_indexing_agreements(
            new_indexer_denylist
        )
        expected_newly_cancelled_agreements = {
            "B": ["subgraph2", "subgraph3", "subgraph5", "subgraph9", "subgraph12"],
            "E": ["subgraph11", "subgraph12", "subgraph14", "subgraph15"],
        }

        # Check state after update
        print("Newly cancelled indexing agreements: ", newly_cancelled_agreements)
        assert newly_cancelled_agreements == expected_newly_cancelled_agreements

        # Verify that the indexer_denylist has been updated
        assert processor.indexer_denylist == new_indexer_denylist

        # Verify that 'H' and 'NOT_IN_LIST' don't appear in cancelled agreements
        assert "H" not in newly_cancelled_agreements
        assert "NOT_IN_LIST" not in newly_cancelled_agreements


class TestNormalizeMetrics:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "Latency Coefficient + Error Confidence Interval": [
                    -5,
                    0,
                    5,
                    10,
                    12.121212,
                ],
                "% up_x": [0, 10, 50, 75.7575, 99.9],
                "stake_to_fees": [-5.15, 0, 1.125, 3, 120],
                "average_status": [0, 1, 50, 75.7575, 99.9],
                "base_price_per_epoch": [10, 200, 300, 400.457, 1000],
                "price_per_entity": [0.1, 0.2, 0.3, 0.4, 0.5],
                "other_column": ["A", 1, "B", 12.12, np.nan],
            }
        )

    def test_normalize_metrics_full_run_base_case(self, sample_df):
        # Compute the result
        result = _normalize_metrics(sample_df)

        # Check all expected columns are present.
        expected_columns = [
            # Original columns
            "Latency Coefficient + Error Confidence Interval",
            "% up_x",
            "stake_to_fees",
            "average_status",
            "base_price_per_epoch",
            "price_per_entity",
            "other_column",
            # New columns
            "norm_lat_lin_reg_coefficient",
            "norm_uptime_score",
            "norm_stake_to_fees",
            "norm_success_rate",
            "norm_base_price_per_epoch",
            "norm_price_per_entity",
        ]
        for col in expected_columns:
            assert col in result.columns

        # Check all normalized values are between 0 and 1
        normalized_columns = [
            "norm_lat_lin_reg_coefficient",
            "norm_uptime_score",
            "norm_stake_to_fees",
            "norm_success_rate",
            "norm_base_price_per_epoch",
            "norm_price_per_entity",
        ]
        for col in normalized_columns:
            assert result[col].between(0, 1).all()

    def test_normalize_generic(self):
        # Test the normalize_generic function
        series = pd.Series([-1000, 0, 345.234, 4, 5000])
        result = _normalize_generic(series)
        assert result.min() == 0
        assert result.max() == 1
        assert len(result) == len(series)

    def test_normalize_uptime_and_success_rate(self):
        # Test the normalize_uptime_and_success_rate function
        series = pd.Series([0, 12.121212, 98, 99, 100])
        result = _normalize_uptime_and_success_rate(series)
        assert result.max() == 1
        assert result.min() == 0
        assert len(result) == len(series)

    def test_empty_dataframe(self, sample_df):
        # Test with an empty DataFrame
        empty_df = pd.DataFrame(columns=sample_df.columns)
        result = _normalize_metrics(empty_df)
        assert result.empty
        expected_columns = list(empty_df.columns) + [
            "norm_lat_lin_reg_coefficient",
            "norm_uptime_score",
            "norm_stake_to_fees",
            "norm_success_rate",
            "norm_base_price_per_epoch",
            "norm_price_per_entity",
        ]
        assert set(result.columns) == set(expected_columns)

    def test_all_same_values(self, sample_df):
        # Test with all values being the same
        sample_df.loc[:, :] = 1000

        # Call normalize_metrics function
        result = _normalize_metrics(sample_df)

        norm_columns = [
            "norm_lat_lin_reg_coefficient",
            "norm_uptime_score",
            "norm_stake_to_fees",
            "norm_success_rate",
            "norm_base_price_per_epoch",
            "norm_price_per_entity",
        ]

        # Check for normalization results where input values are the same
        for column in norm_columns:
            if column in [
                "norm_stake_to_fees",
            ]:
                assert (result[column] == 0).all(), (
                    f"Column {column} is not 0 for identical input values"
                )

            elif column in [
                "norm_lat_lin_reg_coefficient",
            ]:
                assert (result[column] == 1).all(), (
                    f"Column {column} is not 1 for identical input values"
                )

    def test_negative_values(self, sample_df):
        # Test with negative values (7 columns)
        sample_df.loc[0] = [-1, -1, -1, -1, -1, -1, -1]
        sample_df.loc[1] = [-100, -50, -25, -10, -5, -1, -1]
        sample_df.loc[2] = [0, 0, 0, 0, 0, 0, 0]
        sample_df.loc[3] = [1, 1, 1, 1, 1, 1, 1]
        sample_df.loc[4] = [-1000, 0, -500, 500, -250, 250, 250]

        # Compute result
        result = _normalize_metrics(sample_df)

        # Check negative numbers don't create np.nan's in the result
        assert not result.isnull().any().any()

        norm_columns = result.columns[result.columns.str.startswith("norm_")]
        for col in norm_columns:
            min_val = result[col].min()
            max_val = result[col].max()

            # Make sure results are normalized correctly.
            assert min_val >= 0 and max_val <= 1
            assert not result[col].isin([np.inf, -np.inf]).any()

    def test_all_negative_values(self, sample_df):
        # Test with all negative values
        sample_df.loc[:, :] = -1
        result = _normalize_metrics(sample_df)

        # Check that the function handles all negative values as expected
        norm_columns = [col for col in result.columns if col.startswith("norm_")]

        for col in norm_columns:
            assert result[col].between(0, 1).all(), (
                f"Column {col} contains values outside [0, 1] range"
            )

        assert not result[norm_columns].isnull().any().any(), (
            "Result contains unexpected NaN values"
        )

    def test_nan_values(self, sample_df):
        # Test with NaN values
        sample_df.loc[0] = [np.nan] * len(sample_df.columns)
        result = _normalize_metrics(sample_df)

        # Check that NaN values are not present in other rows of normalized columns
        assert not result.iloc[1:, result.columns.str.startswith("norm_")].isnull().any().any()

    def test_price_normalization(self):
        """Test that price columns normalize correctly (lower is better)."""
        df = pd.DataFrame(
            {
                "Latency Coefficient + Error Confidence Interval": [1, 2, 3],
                "% up_x": [99, 100, 98],
                "stake_to_fees": [0.1, 0.2, 0.3],
                "average_status": [99, 100, 98],
                "base_price_per_epoch": [100, 500, 1000],
                "price_per_entity": [0.1, 0.5, 1.0],
            }
        )
        result = _normalize_metrics(df)
        # Cheapest indexer should have highest score
        assert (
            result["norm_base_price_per_epoch"].iloc[0] == result["norm_base_price_per_epoch"].max()
        )
        assert result["norm_price_per_entity"].iloc[0] == result["norm_price_per_entity"].max()


class TestTargetSize:
    """Tests for variable target_size parameter in DataProcessor."""

    @pytest.fixture
    def sample_data_with_scores(self):
        """Sample data with all required fields for DataProcessor."""
        return pd.DataFrame(
            {
                "indexer": ["A", "B", "C", "D", "E"],
                "deployment_hash": ["hash1"] * 5,
                "destination_loc": ["loc1", "loc2", "loc3", "loc4", "loc5"],
                "org": ["org1", "org2", "org3", "org4", "org5"],
                "weighted_score": [0.9, 0.8, 0.7, 0.6, 0.5],
                "lat_lin_reg_coefficient": [0.1, 0.2, 0.3, 0.4, 0.5],
                "uptime_score": [0.9, 0.8, 0.7, 0.6, 0.5],
                "stake_to_fees": [0.1, 0.2, 0.3, 0.4, 0.5],
                "success_rate": [0.95, 0.90, 0.85, 0.80, 0.75],
                "base_price_per_epoch": [100, 200, 300, 400, 500],
                "price_per_entity": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )

    def test_target_size_defaults_to_three(self, sample_data_with_scores):
        """Default target_size is 3."""
        # Arrange & Act
        with patch("iisa.indexer_selection.DataProcessor._process_data"):
            processor = DataProcessor(
                history=sample_data_with_scores,
                deployment_id=DeploymentId("test_subgraph"),
            )

        # Assert
        assert processor.target_size == 3

    def test_target_size_custom_value(self, sample_data_with_scores):
        """Custom target_size is stored correctly."""
        # Arrange & Act
        with patch("iisa.indexer_selection.DataProcessor._process_data"):
            processor = DataProcessor(
                history=sample_data_with_scores,
                deployment_id=DeploymentId("test_subgraph"),
                target_size=5,
            )

        # Assert
        assert processor.target_size == 5

    def test_target_size_one_selects_single_indexer(self, sample_data_with_scores):
        """With target_size=1, only one indexer is selected."""
        # Arrange & Act
        processor = DataProcessor(
            history=sample_data_with_scores,
            deployment_id=DeploymentId("test_subgraph"),
            target_size=1,
        )

        # Assert
        assert len(processor.current_group) == 1

    def test_target_size_five_selects_five_indexers(self, sample_data_with_scores):
        """With target_size=5, five indexers are selected."""
        # Arrange & Act
        processor = DataProcessor(
            history=sample_data_with_scores,
            deployment_id=DeploymentId("test_subgraph"),
            target_size=5,
        )

        # Assert
        assert len(processor.current_group) == 5

    def test_target_size_respects_available_indexers(self, sample_data_with_scores):
        """target_size > available indexers returns all available."""
        # Arrange & Act
        processor = DataProcessor(
            history=sample_data_with_scores,
            deployment_id=DeploymentId("test_subgraph"),
            target_size=10,  # More than available
        )

        # Assert - Should have at most 5 (all available indexers)
        assert len(processor.current_group) <= 5

    def test_target_size_removes_excess_indexers(self, sample_data_with_scores):
        """Existing group larger than target_size gets trimmed."""
        # Arrange & Act
        processor = DataProcessor(
            history=sample_data_with_scores,
            deployment_id=DeploymentId("test_subgraph"),
            existing_agreements={
                DeploymentId("test_subgraph"): [
                    IndexerId("A"),
                    IndexerId("B"),
                    IndexerId("C"),
                    IndexerId("D"),
                ]
            },
            target_size=2,
        )

        # Assert - Should have reduced to 2
        assert len(processor.current_group) == 2

    def test_target_size_adds_to_small_group(self, sample_data_with_scores):
        """Existing group smaller than target_size gets expanded."""
        # Arrange & Act
        processor = DataProcessor(
            history=sample_data_with_scores,
            deployment_id=DeploymentId("test_subgraph"),
            existing_agreements={DeploymentId("test_subgraph"): [IndexerId("A")]},
            target_size=4,
        )

        # Assert - Should have expanded to 4
        assert len(processor.current_group) == 4


class TestDecentralizationBestEffort:
    """Tests for best-effort decentralization behavior."""

    def test_fallback_when_decentralization_not_possible(self):
        """When no indexer meets decentralization, still return best candidate."""
        # All indexers have the same org and location - decentralization impossible
        data = pd.DataFrame(
            {
                "indexer": ["A", "B", "C"],
                "deployment_hash": ["hash1"] * 3,
                "destination_loc": ["loc1", "loc1", "loc1"],  # Same location
                "org": ["org1", "org1", "org1"],  # Same org
                "weighted_score": [0.9, 0.8, 0.7],
                "lat_lin_reg_coefficient": [0.1, 0.2, 0.3],
                "uptime_score": [0.9, 0.8, 0.7],
                "stake_to_fees": [0.1, 0.2, 0.3],
                "success_rate": [0.95, 0.90, 0.85],
                "base_price_per_epoch": [100, 200, 300],
                "price_per_entity": [0.1, 0.2, 0.3],
            }
        )

        # Arrange & Act
        processor = DataProcessor(
            history=data,
            deployment_id=DeploymentId("test_subgraph"),
            target_size=3,
        )

        # Assert - Should still select 3 indexers even though decentralization not met
        assert len(processor.current_group) == 3


class TestCalculateWeightedScore:
    @pytest.fixture
    def sample_weights(self):
        return {"metric1": 0.5, "metric2": 0.3, "metric3": 0.2}

    def test_basic_calculation(self, sample_weights):
        # Test the function with all metrics present
        row = pd.Series({"norm_metric1": 0.8, "norm_metric2": 0.6, "norm_metric3": 0.4})
        result = _calculate_weighted_score(row, sample_weights)
        expected = (0.8 * 0.5 + 0.6 * 0.3 + 0.4 * 0.2) / 1.0
        assert np.isclose(result, expected)

    def test_missing_metric(self, sample_weights):
        # Test the function when one metric is missing (NaN)
        row = pd.Series({"norm_metric1": 0.8, "norm_metric2": np.nan, "norm_metric3": 0.4})
        result = _calculate_weighted_score(row, sample_weights)
        expected = ((0.8 * 0.5) + (0 * 0.3) + (0.4 * 0.2)) / (0.5 + 0.2)
        assert np.isclose(result, expected)

    def test_all_metrics_missing(self, sample_weights):
        # Test the function when all metrics are missing (NaN)
        row = pd.Series({"norm_metric1": np.nan, "norm_metric2": np.nan, "norm_metric3": np.nan})
        with pytest.raises(ValueError, match="Total weight cannot be 0."):
            _calculate_weighted_score(row, sample_weights)

    def test_zero_weights(self):
        # Test the function when all weights are zero
        weights = {"metric1": 0, "metric2": 0, "metric3": 0}
        row = pd.Series({"norm_metric1": 0.8, "norm_metric2": 0.6, "norm_metric3": 0.4})
        with pytest.raises(ValueError, match="Total weight cannot be 0."):
            _calculate_weighted_score(row, weights)

    def test_partial_weights(self):
        # Test the function when some weights are zero
        weights = {"metric1": 0.5, "metric2": 0, "metric3": 0.5}
        row = pd.Series({"norm_metric1": 0.8, "norm_metric2": 0.6, "norm_metric3": 0.4})
        result = _calculate_weighted_score(row, weights)
        expected = (0.8 * 0.5 + 0.4 * 0.5) / 1.0
        assert np.isclose(result, expected)

    def test_extra_metrics_in_row(self, sample_weights):
        # Test the function when the row contains extra metrics not in weights
        row = pd.Series(
            {
                "norm_metric1": 0.8,
                "norm_metric2": 0.6,
                "norm_metric3": 0.4,
                "norm_metric4": 1.0,
                "other_column": "value",
            }
        )
        result = _calculate_weighted_score(row, sample_weights)
        expected = (0.8 * 0.5 + 0.6 * 0.3 + 0.4 * 0.2) / 1.0
        assert np.isclose(result, expected)

    @pytest.mark.parametrize(
        "row_data, weights, expected",
        [
            (
                {"norm_metric1": 1.0, "norm_metric2": 1.0},
                {"metric1": 1, "metric2": 1},
                1.0,
            ),
            (
                {"norm_metric1": 0.0, "norm_metric2": 0.0},
                {"metric1": 1, "metric2": 1},
                0.0,
            ),
            (
                {"norm_metric1": 0.5, "norm_metric2": 0.5},
                {"metric1": 1, "metric2": 1},
                0.5,
            ),
        ],
    )
    def test_edge_cases(self, row_data, weights, expected):
        # Test various edge cases
        row = pd.Series(row_data)
        result = _calculate_weighted_score(row, weights)
        assert np.isclose(result, expected)
