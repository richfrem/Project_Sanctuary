# Task #021B: Forge Test Suite

**Objective:** Create a unit test suite for the `OPERATION_PHOENIX_FORGE` pipeline scripts to ensure data integrity and training logic correctness.

- [ ] **Step 1: Test Infrastructure** <!-- id: 0 -->
    - [ ] Create `forge/tests/conftest.py` with fixtures for mock datasets and models <!-- id: 1 -->
- [ ] **Step 2: Data Pipeline Tests** <!-- id: 2 -->
    - [ ] Create `forge/tests/test_dataset_forge.py` (Tests `forge_whole_genome_dataset.py`) <!-- id: 3 -->
    - [ ] Create `forge/tests/test_dataset_validation.py` (Tests `validate_dataset.py`) <!-- id: 4 -->
- [ ] **Step 3: Training Logic Tests** <!-- id: 5 -->
    - [ ] Create `forge/tests/test_fine_tune_logic.py` (Tests `fine_tune.py` arguments and config) <!-- id: 6 -->
- [ ] **Step 4: Verification** <!-- id: 7 -->
    - [ ] Run `pytest forge/tests/` <!-- id: 8 -->
