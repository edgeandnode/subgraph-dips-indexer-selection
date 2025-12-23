from iisa import select_many, select_one


class TestIISASelect:
    """
    Tests for the IISA select functions.
    """

    def test_select_one_from_empty_list(self):
        ## Given
        candidate_pool = []

        ## When
        selected = select_one(candidate_pool)

        ## Then
        assert selected is None

    def test_select_one_from_list(self, faker):
        ## Given
        candidate_pool = [
            faker.indexer_id(),
            faker.indexer_id(),
            faker.indexer_id(),
        ]

        ## When
        selected = select_one(candidate_pool)

        ## Then
        assert selected is not None
        assert selected in candidate_pool

    def test_select_many_from_empty_list(self):
        ## Given
        candidate_pool = []

        ## When
        selected = select_many(candidate_pool, 3)

        ## Then
        assert len(selected) == 0

    def test_select_many_from_list(self, faker):
        ## Given
        candidate_pool = [
            faker.indexer_id(),
            faker.indexer_id(),
            faker.indexer_id(),
        ]

        ## When
        selected = select_many(candidate_pool, 2)

        ## Then
        assert len(selected) == 2
        assert all([s in candidate_pool for s in selected])

    def test_select_many_more_than_available(self, faker):
        ## Given
        candidate_pool = [
            faker.indexer_id(),
            faker.indexer_id(),
        ]

        ## When
        selected = select_many(candidate_pool, 3)

        ## Then
        assert len(selected) == 2
        assert all([s in candidate_pool for s in selected])

    def test_select_many_with_zero_count(self, faker):
        ## Given
        candidate_pool = [
            faker.indexer_id(),
            faker.indexer_id(),
        ]

        ## When
        selected = select_many(candidate_pool, 0)

        ## Then
        assert len(selected) == 0

    def test_select_many_with_negative_count(self, faker):
        ## Given
        candidate_pool = [
            faker.indexer_id(),
            faker.indexer_id(),
        ]

        ## When
        selected = select_many(candidate_pool, -1)

        ## Then
        assert len(selected) == 0
