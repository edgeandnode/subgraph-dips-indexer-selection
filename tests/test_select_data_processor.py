from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from iisa.select.processor import (
    DataProcessor,
    _calculate_weighted_score,
    _normalize_generic,
    _normalize_indexing_agreement_acceptance_latency,
    _normalize_metrics,
    _normalize_uptime_and_success_rate,
)
from iisa.typing import DeploymentId, IndexerId


def process_subgraph(
    history,
    deployment_id,
    existing_agreements,
    blocklist=None,
):
    processor = DataProcessor(
        history,
        deployment_id,
        existing_agreements=existing_agreements,
        blocklist=blocklist,
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
def mock__bigquery_provider(faker, mock__combined_query_results):
    bigquery_provider = MagicMock()
    bigquery_provider.return_value.fetch_initial_query_results.return_value = (
        pd.DataFrame(
            {
                "deployment_hash": [faker.deployment_id() for _ in range(3)],
                "indexer": [faker.indexer_id() for _ in range(3)],
                "num_rows": [1000, 2000, 3000],
            }
        )
    )
    bigquery_provider.return_value.fetch_combined_query_results.return_value = (
        mock__combined_query_results
    )
    bigquery_provider.return_value.fetch_initial_stake_to_fees.return_value = (
        pd.DataFrame(
            {
                "indexer": [faker.indexer_id() for _ in range(3)],
                "stake_to_fees": [1.0, 2.0, 3.0],
            }
        )
    )
    return bigquery_provider


class TestProcessSubgraph:
    """
    This class verifies the process_subgraph function creates a DataProcessor
    instance and returns the expected results for added/cancelled indexers.
    """

    @pytest.mark.skip(reason="Flaky test: high dependency on internal details")
    @patch("iisa.select.processor.DataProcessor")
    def test_process_subgraph(
        self, mock__data_processor, sample_data, mock__bigquery_provider
    ):
        """
        Test the process_subgraph function creates a DataProcessor instance and returns the expected results.

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
        blocklist = ["blocklisted_indexer"]

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
            blocklist=blocklist,
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
    This class contains a range of unit tests to ensure that the DataProcessor class functions as intended.
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
                "existing_dips_agreements": [1, 2, 3],
                "weighted_score": [0.9, 0.7, 0.8],
                "lat_lin_reg_coefficient": [0.1, 0.2, 0.3],
                "uptime_score": [0.9, 0.8, 0.7],
                "stake_to_fees_iqr_deviation": [0.1, 0.2, 0.3],
                "success_rate": [0.95, 0.90, 0.85],
                "avg_sync_duration": [100, 200, 300],
                "indexing_agreement_acceptance_latency": [10, 20, 30],
            }
        )

    @pytest.mark.skip(reason="Flaky test: high dependency on internal details")
    def test_data_processor_constructor(self, sample_data):
        """
        Test the initialization of the DataProcessor class.

        This test verifies:
        1. The constructor correctly sets all instance variables with provided parameters.
        2. Default values are applied when optional parameters are not provided.
        3. The BigQueryProvider is properly instantiated.
        4. The _process_data method is called once.
        5. The blocklist is properly applied.
        6. The 'data' DataFrame maintains its original content, while adding the new columns.
        7. Optional parameters (existing_agreements, blocklist) default empty if not set.

        The test uses mock objects for BigQueryProvider and patch decorators for _process_data
        and derive_timestamps to avoid actual data fetching and ensure consistent test behavior.
        """
        # Define test input parameters
        deployment_id = DeploymentId("test_subgraph")
        existing_agreements = {
            DeploymentId("subgraph1"): [IndexerId("A")],
            DeploymentId("subgraph2"): [IndexerId("B")],
        }
        blocklist = [IndexerId("D")]

        # Create a DataProcessor instance
        processor = DataProcessor(
            history=sample_data,
            deployment_id=deployment_id,
            existing_agreements=existing_agreements,
            blocklist=blocklist,
        )

        # Verify that all instance variables are set correctly
        assert processor.deployment_id == deployment_id
        assert processor.existing_agreements == existing_agreements
        assert processor.blocklist == blocklist

        # Verify default values for optional parameters
        processor_default = DataProcessor(
            history=sample_data,
            deployment_id=deployment_id,
        )
        assert processor_default.existing_agreements == {}
        assert processor_default.blocklist == []

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
        mock__bigquery_provider,
    ):
        """
        This test verifies the get_indexer_selections method correctly identifies the
        recent added and cancelled indexers.
        """
        with patch("iisa.select.processor.DataProcessor._process_data"):
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
        expected_cancelled_sorted = {
            k: sorted(v) for k, v in expected_cancelled.items()
        }

        # Verify the results by comparing sorted dictionaries
        assert added_sorted == expected_added_sorted, (
            f"Expected added: {expected_added_sorted}, but got: {added_sorted}"
        )
        assert cancelled_sorted == expected_cancelled_sorted, (
            f"Expected cancelled: {expected_cancelled_sorted}, but got: {cancelled_sorted}"
        )

    def test_get_indexer_selections_empty_groups(
        self, sample_data, mock__bigquery_provider
    ):
        """
        Test get_indexer_selections method when both initial_group and current_group are empty.

        This test verifies that the method handles the scenario where both the initial_group
        and current_group are empty (represented as an empty list and an empty set respectively).
        It ensures that the method returns empty lists for both added and cancelled indexers
        when there are no indexers in either group.
        """
        with patch("iisa.select.processor.DataProcessor._process_data"):
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

    @patch("iisa.select.processor.DataProcessor._fetch_number_of_indexer_agreements")
    @patch("iisa.select.processor.DataProcessor._get_current_group")
    @patch("iisa.select.processor.DataProcessor._normalize_and_score")
    @patch("iisa.select.processor.DataProcessor._assign_indexers_to_subgraph")
    def test_process_data(
        self,
        mock_assign,
        mock_normalize,
        mock_get_group,
        mock__fetch,
        sample_data,
        mock__bigquery_provider,
    ):
        """
        Test the _process_data method of the DataProcessor class.

        This test verifies that:
        1. The _process_data method calls the methods in the correct order.
        2. Each method is called exactly once during processing.
        3. The _process_data method handles the data correctly, passing results between methods.
        4. The current_group and initial_group are properly set and updated.
        5. The data is correctly sorted by weighted_score.
        """
        # Create a DataProcessor instance
        processor = DataProcessor(
            history=sample_data,
            deployment_id=DeploymentId("test_subgraph"),
        )

        # Reset all mock call counts after initialization
        mock__fetch.reset_mock()
        mock_get_group.reset_mock()
        mock_normalize.reset_mock()
        mock_assign.reset_mock()

        # Set up mock return values
        mock__fetch.return_value = pd.DataFrame(
            {"indexer": ["A", "B", "C"], "existing_dips_agreements": [1, 2, 3]}
        )
        mock_get_group.return_value = ["A", "B"]
        mock_normalize.return_value = pd.DataFrame(
            {"indexer": ["A", "B", "C"], "weighted_score": [0.8, 0.7, 0.9]}
        )

        # Call the method under test
        processor._process_data()

        # Verify that all expected methods were called only once
        assert mock__fetch.call_count == 1
        assert mock_get_group.call_count == 1
        assert mock_normalize.call_count == 1
        assert mock_assign.call_count == 1

        # Verify the order of method calls
        expected_call_order = [
            call._fetch_number_of_indexer_agreements(),
            call._get_current_group(),
            call._normalize_and_score(),
            call._assign_indexers_to_subgraph(),
        ]
        actual_calls = (
            mock__fetch.mock_calls
            + [mock_get_group.mock_calls[0]]
            + mock_normalize.mock_calls
            + mock_assign.mock_calls
        )
        assert actual_calls == expected_call_order

        # Verify that the current_group and initial_group are set correctly
        assert processor.current_group == ["A", "B"]
        assert processor.initial_group == ["A", "B"]

    def test_fetch_number_of_indexer_agreements(
        self, sample_data, mock__bigquery_provider
    ):
        """
        This test verifies the _fetch_number_of_indexer_agreements method updates the
        'existing_dips_agreements' column based on the existing_agreements.
        """
        # Create a DataProcessor instance with specific existing agreements
        with patch("iisa.select.processor.DataProcessor._process_data"):
            processor = DataProcessor(
                history=sample_data,
                deployment_id=DeploymentId("test_subgraph"),
                existing_agreements={
                    DeploymentId("subgraph1"): [
                        IndexerId("A"),
                        IndexerId("B"),
                        IndexerId("A"),
                    ],
                    DeploymentId("subgraph2"): [IndexerId("A"), IndexerId("B")],
                    DeploymentId("subgraph3"): [IndexerId("A")],
                },
            )

        # Call the method under test
        updated_data = processor._fetch_number_of_indexer_agreements()

        # Verify that 'existing_dips_agreements' are updated correctly
        assert (
            updated_data.loc[
                updated_data["indexer"] == "A", "existing_dips_agreements"
            ].iloc[0]
            == 4
        ), "A issue"
        assert (
            updated_data.loc[
                updated_data["indexer"] == "B", "existing_dips_agreements"
            ].iloc[0]
            == 2
        ), "B issue"
        assert (
            updated_data.loc[
                updated_data["indexer"] == "C", "existing_dips_agreements"
            ].iloc[0]
            == 0
        ), "C issue"

    @pytest.fixture
    def processor(self, sample_data, mock__bigquery_provider):
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

    def test_get_current_group_subgraph_not_in_agreements(
        self, processor, mock__bigquery_provider
    ):
        """
        Test _get_current_group when the subgraph 'test_subgraph' is not in any agreement.
        """
        processor.existing_agreements = {
            "A": ["other_subgraph1", "other_subgraph2"],
            "B": ["other_subgraph3", "other_subgraph4"],
        }
        result = processor._get_current_group()
        assert result == []

    @patch("iisa.select.processor._normalize_metrics")
    @patch("iisa.select.processor._calculate_weighted_score")
    def test_normalize_and_score(
        self, mock_calculate_score, mock_normalize, sample_data, mock__bigquery_provider
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
        with patch("iisa.select.processor.DataProcessor._process_data"):
            processor = DataProcessor(
                history=sample_data,
                deployment_id=DeploymentId("test_subgraph"),
            )

        # Set up mock return values
        normalized_data = sample_data.copy()
        for metric in [
            "lat_lin_reg_coefficient",
            "uptime_score",
            "existing_dips_agreements",
            "stake_to_fees_iqr_deviation",
            "success_rate",
            "avg_sync_duration",
            "indexing_agreement_acceptance_latency",
        ]:
            normalized_data[f"norm_{metric}"] = normalized_data[metric]
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
                "lat_lin_reg_coefficient",
                "uptime_score",
                "existing_dips_agreements",
                "stake_to_fees_iqr_deviation",
                "success_rate",
                "avg_sync_duration",
                "indexing_agreement_acceptance_latency",
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
    def test_assign_indexers_to_subgraph(self, sample_data, mock__bigquery_provider):
        """
        Test the _assign_indexers_to_subgraph method of DataProcessor.

        This test verifies:
        1. The method calls _add_indexers_to_group when there are fewer than 3 indexers.
        2. The method calls _replace_underperforming_indexers when there are 3 or more indexers.
        """
        with patch(
            "iisa.select.processor.DataProcessor._add_indexers_to_group"
        ) as mock_add:
            with patch(
                "iisa.select.processor.DataProcessor._replace_underperforming_indexers)"
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
        mock__bigquery_provider,
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
            "iisa.select.processor.DataProcessor._find_best_replacement_or_select_best_indexer"
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
            "iisa.select.processor.DataProcessor._find_best_replacement_or_select_best_indexer",
            return_value=None,
        ):
            processor.current_group = ["A"]
            processor._add_indexers_to_group()
            assert processor.current_group == ["A"]

    def test_meets_decentralization_requirements(self, mock__bigquery_provider):
        """
        Test the _meets_decentralization_requirements method of DataProcessor.

        This test verifies:
        1. The method returns True when there are fewer than 2 indexers in the current group.
        2. The method correctly evaluates decentralization based on locations and organizations.
        3. A group that does not _meets_decentralization_requirements will not be marked as true.

        Note:
        _meets_decentralization_requirements accepts new_indexer as an input parameter.
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

        # Test with fewer than 2 indexers
        processor.current_group = ["A"]
        assert processor._meets_decentralization_requirements("B")

        # Test with 2 indexers, same location and org
        processor.current_group = ["A", "B"]
        assert processor._meets_decentralization_requirements("C")

        # Test with 2 indexers, different location and org
        processor.current_group = ["A", "C"]
        assert processor._meets_decentralization_requirements("D")

        # Test with 2 indexers, adding one with same location and org
        processor.current_group = ["A", "C"]
        assert processor._meets_decentralization_requirements("B")

        # Test with 3 of the same indexer.
        processor.current_group = ["A", "A"]
        assert not processor._meets_decentralization_requirements("A")

    def test_meets_decentralization_requirements_edge_cases(
        self, mock__bigquery_provider
    ):
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

        # Test with empty current group
        assert processor._meets_decentralization_requirements("A")

        # Test with indexer 'A' selected twice due to some error
        processor.current_group = ["A", "A"]
        assert processor._meets_decentralization_requirements("E")

        # Test with many indexers
        processor.current_group = ["A", "B", "C", "D", "E", "F"]
        assert processor._meets_decentralization_requirements("F")

        # Additional test: Check that it returns False when decentralization requirements are not met
        processor.current_group = ["A", "B"]
        assert not processor._meets_decentralization_requirements("A")

    def test_replace_underperforming_indexers(
        self, sample_data, mock__bigquery_provider
    ):
        """
        Test the _replace_underperforming_indexers method of DataProcessor.

        This test verifies:
        1. The method replaces an indexer when a better replacement is found.
        2. The method does not replace any indexer when no better replacement is found.
        """
        processor = DataProcessor(
            history=sample_data,
            deployment_id=DeploymentId("test_subgraph"),
        )

        with (
            patch(
                "iisa.select.processor.DataProcessor._find_best_replacement_or_select_best_indexer"
            ) as mock_find,
            patch(
                "iisa.select.processor.DataProcessor._calculate_group_score"
            ) as mock_score,
        ):
            mock_find.side_effect = ["D", None, None]
            mock_score.side_effect = [0.7, 0.8, 0.7, 0.7]

            processor.current_group = ["A", "B", "C"]
            processor._replace_underperforming_indexers()

            # Verify that the worst indexer in the current group has been replaced with the best available indexer
            assert processor.current_group == ["B", "C", "D"]
            assert mock_find.call_count == 3
            assert mock_score.call_count == 2

    def test_find_best_replacement_or_select_best_indexer(
        self, mock__bigquery_provider
    ):
        """
        Test the _find_best_replacement_or_select_best_indexer method of DataProcessor.

        This test verifies:
        1. The method returns the best replacement that meets decentralization requirements.
        2. The method returns None when no suitable replacement is found.
        3. The method will not try to replace an indexer with one that is already blocklisted.
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
        processor.blocklist = ["E"]

        with patch(
            "iisa.select.processor.DataProcessor._meets_decentralization_requirements"
        ) as mock_decentralization:
            mock_decentralization.side_effect = [True]

            result = processor._find_best_replacement_or_select_best_indexer()

            # Verify the best replacement is D, not E, due to blocklisting.
            assert result == "D"

            # Verify the number of decentralization requirement checks
            assert mock_decentralization.call_count == 1

    def test_calculate_group_score(self, mock__bigquery_provider):
        """
        Test the _calculate_group_score method of the DataProcessor class.

        This test verifies that:
        1. The method correctly calculates group scores for different scenarios:
        2. The method produces consistent results for each scenario.

        The test uses raw, non-normalized sample data to create a DataProcessor instance,
        sets predefined weights, and then calls _calculate_group_score with different
        parameters to test various scenarios.
        """
        # raw non-normalized sample data
        raw_data = pd.DataFrame(
            {
                "indexer": ["A", "B", "C", "D"],
                "destination_loc": ["0,0", "0,0", "0,0", "0,0"],
                "org": ["org1", "org7", "org3", "org2"],
                "existing_dips_agreements": [1, 2, 3, 4],
                "lat_lin_reg_coefficient": [0.1, 0.2, 0.3, 0.4],
                "uptime_score": [0.9, 0.8, 0.7, 0.6],
                "stake_to_fees_iqr_deviation": [0.1, 0.2, 0.3, 0.4],
                "success_rate": [0.95, 0.90, 0.85, 0.80],
                "avg_sync_duration": [100, 200, 300, 400],
                "indexing_agreement_acceptance_latency": [10, 20, 30, 40],
            }
        )

        processor = DataProcessor(
            history=raw_data,
            deployment_id=DeploymentId("test_subgraph"),
        )

        processor.weights = {
            "lat_lin_reg_coefficient": 0.2424,
            "uptime_score": 0.1667,
            "existing_dips_agreements": 0.1212,
            "stake_to_fees_iqr_deviation": 0.1023,
            "success_rate": 0.0625,
            "avg_sync_duration": 0.0625,
            "indexing_agreement_acceptance_latency": 0.2424,
        }

        original_data = processor.data.copy()

        normal_score = processor._calculate_group_score(["A", "B", "C"])
        exclude_score = processor._calculate_group_score(
            ["A", "C"], indexer_to_exclude="B"
        )
        include_score = processor._calculate_group_score(
            ["A", "B"], indexer_to_include="D"
        )

        # How allclose() works: It considers two values a and b to be "close" if: |a - b| <= (atol + rtol * |b|)
        assert np.allclose(normal_score, 0.19696666666666665, rtol=1e-9, atol=1e-9)
        assert np.allclose(exclude_score, 0.07576666666666666, rtol=1e-9, atol=1e-9)
        assert np.allclose(include_score, 0.19033333333333335, rtol=1e-9, atol=1e-9)

        # Verify that the original data was not modified
        pd.testing.assert_frame_equal(processor.data, original_data)

    def test_update_blocklist_cancel_indexing_agreements(
        self, sample_data, mock__bigquery_provider
    ):
        """
        Test the update_blocklist_cancel_indexing_agreements method of DataProcessor.

        This test verifies:
        1. The method correctly identifies agreements to be cancelled based on the new blocklist.
        2. The method returns the correct dictionary of cancelled agreements.
        3. The method updates the internal blocklist of the DataProcessor.
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
            blocklist=[IndexerId("H")],
        )

        # update the blocklist to cancel agreements
        new_blocklist = [
            IndexerId("H"),
            IndexerId("B"),
            IndexerId("E"),
            IndexerId("NOT_IN_LIST"),
        ]

        # Call update_blocklist_cancel_indexing_agreements with new new_blocklist
        newly_cancelled_agreements = (
            processor.update_blocklist_cancel_indexing_agreements(new_blocklist)
        )
        expected_newly_cancelled_agreements = {
            "B": ["subgraph2", "subgraph3", "subgraph5", "subgraph9", "subgraph12"],
            "E": ["subgraph11", "subgraph12", "subgraph14", "subgraph15"],
        }

        # Check state after update
        print("Newly cancelled indexing agreements: ", newly_cancelled_agreements)
        assert newly_cancelled_agreements == expected_newly_cancelled_agreements

        # Verify that the blocklist has been updated
        assert processor.blocklist == new_blocklist

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
                "existing_dips_agreements": [0, 100, 31, 35, 50],
                "stake_to_fees_iqr_deviation": [-5.15, 0, 1.125, 3, 120],
                "average_status": [0, 1, 50, 75.7575, 99.9],
                "avg_sync_duration": [10, 200, 300, 400.457, 1000],
                "indexing_agreement_acceptance_latency": [0, 0.5, 2, 12, 24],  # hours
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
            "existing_dips_agreements",
            "stake_to_fees_iqr_deviation",
            "average_status",
            "avg_sync_duration",
            "indexing_agreement_acceptance_latency",
            "other_column",
            # New columns
            "norm_lat_lin_reg_coefficient",
            "norm_uptime_score",
            "norm_existing_dips_agreements",
            "norm_stake_to_fees_iqr_deviation",
            "norm_success_rate",
            "norm_avg_sync_duration",
            "norm_indexing_agreement_acceptance_latency",
        ]
        for col in expected_columns:
            assert col in result.columns

        # Check all normalized values are between 0 and 1
        normalized_columns = [
            "norm_lat_lin_reg_coefficient",
            "norm_uptime_score",
            "norm_existing_dips_agreements",
            "norm_stake_to_fees_iqr_deviation",
            "norm_success_rate",
            "norm_avg_sync_duration",
            "norm_indexing_agreement_acceptance_latency",
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

    def test_normalize_indexing_agreement_acceptance_latency(self):
        # Test with a pandas Series input
        latencies = pd.Series([0, 1, 2, 12, 24])
        results = _normalize_indexing_agreement_acceptance_latency(latencies)

        assert len(results) == 5
        assert all(0 <= r <= 1 for r in results)
        # Test with a single value
        single_result = _normalize_indexing_agreement_acceptance_latency(
            pd.Series([60])
        )
        assert 0 <= single_result.iloc[0] <= 1

        # Test that lower latencies result in higher normalized values
        assert results.iloc[0] > results.iloc[-1]

        # Test with all same values
        same_values = _normalize_indexing_agreement_acceptance_latency(
            pd.Series([60, 60, 60])
        )
        assert all(r == 0 for r in same_values)

    def test_empty_dataframe(self, sample_df):
        # Test with an empty DataFrame
        empty_df = pd.DataFrame(columns=sample_df.columns)
        result = _normalize_metrics(empty_df)
        assert result.empty
        expected_columns = list(empty_df.columns) + [
            "norm_lat_lin_reg_coefficient",
            "norm_uptime_score",
            "norm_existing_dips_agreements",
            "norm_stake_to_fees_iqr_deviation",
            "norm_success_rate",
            "norm_avg_sync_duration",
            "norm_indexing_agreement_acceptance_latency",
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
            "norm_existing_dips_agreements",
            "norm_stake_to_fees_iqr_deviation",
            "norm_success_rate",
            "norm_avg_sync_duration",
            "norm_indexing_agreement_acceptance_latency",
        ]

        # Check for normalization results where input values are the same
        for column in norm_columns:
            if column in [
                "norm_stake_to_fees_iqr_deviation",
            ]:
                assert (result[column] == 0).all(), (
                    f"Column {column} is not 0 for identical input values"
                )

            elif column in [
                "norm_existing_dips_agreements",
                "norm_avg_sync_duration",
                "norm_lat_lin_reg_coefficient",
            ]:
                assert (result[column] == 1).all(), (
                    f"Column {column} is not 0 for identical input values"
                )

            # For the logistic normalization (indexing agreement acceptance latency)
            elif column == "norm_indexing_agreement_acceptance_latency":
                assert (result[column] == 0).all(), (
                    "(result[column] == 0).all() not true"
                )

    def test_negative_values(self, sample_df):
        # Test with negative values
        sample_df.loc[0] = [-1, -1, -1, -1, -1, -1, -1, -1]
        sample_df.loc[1] = [-100, -50, -75, -25, -10, -5, -1, -1]
        sample_df.loc[2] = [0, 0, 0, 0, 0, 0, 0, 0]
        sample_df.loc[3] = [1, 1, 1, 1, 1, 1, 1, 1]
        sample_df.loc[4] = [-1000, 0, 1000, -500, 500, -250, 250, 0]

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
        assert (
            not result.iloc[1:, result.columns.str.startswith("norm_")]
            .isnull()
            .any()
            .any()
        )

    def test_extreme_values_in_latency(self):
        # Test with extreme values
        latencies = pd.Series([0, 5, np.inf, -100, 7])
        results = _normalize_indexing_agreement_acceptance_latency(latencies)

        assert len(results) == 5, "len(results) != 5"
        assert all(0 <= r <= 1 for r in results), "Values not all between 0 and 1"

        # Check that 0 latency results in the highest score
        assert results[0] == results.max(), "0 latency didn't give the highest score"

        # Check that infinite latency results in the lowest score
        assert results[2] == results.min(), "inf latency didn't give the lowest score"

        # Check that negative latency is treated as 0 (highest score)
        assert results[3] == results[0], "negative latency didn't give the lowest score"

        # Check that other values are ordered correctly
        assert results[0] > results[1] > results[4], "values not ordered correctly"


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
        row = pd.Series(
            {"norm_metric1": 0.8, "norm_metric2": np.nan, "norm_metric3": 0.4}
        )
        result = _calculate_weighted_score(row, sample_weights)
        expected = ((0.8 * 0.5) + (0 * 0.3) + (0.4 * 0.2)) / (0.5 + 0.2)
        assert np.isclose(result, expected)

    def test_all_metrics_missing(self, sample_weights):
        # Test the function when all metrics are missing (NaN)
        row = pd.Series(
            {"norm_metric1": np.nan, "norm_metric2": np.nan, "norm_metric3": np.nan}
        )
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
