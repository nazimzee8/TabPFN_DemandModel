-- Verify synthetic regression staged outputs before downloading.
-- A 2-byte shard file means upstream evaluation wrote blank output.
-- Missing final CSVs means aggregation did not complete successfully.
-- If synthetic_regression_aggregation_failure.json exists, read it first.

LIST @EVALUATION_RESULTS_STAGE/regression;

LIST @EVALUATION_RESULTS_STAGE
    PATTERN = '.*synthetic_regression_model_comparison\.csv';

LIST @EVALUATION_RESULTS_STAGE
    PATTERN = '.*synthetic_regression_model_comparison_summary\.csv';

LIST @EVALUATION_RESULTS_STAGE
    PATTERN = '.*synthetic_regression_summary_by_regime\.csv';

LIST @EVALUATION_RESULTS_STAGE
    PATTERN = '.*synthetic_regression_summary_by_feature_noise\.csv';

LIST @EVALUATION_RESULTS_STAGE
    PATTERN = '.*synthetic_regression_summary_by_training_size\.csv';

LIST @EVALUATION_RESULTS_STAGE
    PATTERN = '.*synthetic_regression_chart_data_noise_features\.csv';

LIST @EVALUATION_RESULTS_STAGE
    PATTERN = '.*synthetic_regression_chart_data_training_size\.csv';

LIST @EVALUATION_RESULTS_STAGE
    PATTERN = '.*synthetic_regression_chart_data_model_rank\.csv';

LIST @EVALUATION_RESULTS_STAGE
    PATTERN = '.*synthetic_regression_aggregation_manifest\.json';

LIST @EVALUATION_RESULTS_STAGE
    PATTERN = '.*synthetic_regression_aggregation_failure\.json';
