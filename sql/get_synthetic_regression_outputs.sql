-- Download synthetic regression evaluation outputs to local results folder.
-- Run from a connected snowsql session.
-- PARALLEL = 4 uses 4 concurrent download threads per file.

GET @EVALUATION_RESULTS_STAGE/synthetic_regression_model_comparison.csv
    'file://C:/Documents/TabPFN_DemandModel/results/'
    PARALLEL = 4;

GET @EVALUATION_RESULTS_STAGE/synthetic_regression_model_comparison_summary.csv
    'file://C:/Documents/TabPFN_DemandModel/results/'
    PARALLEL = 4;

GET @EVALUATION_RESULTS_STAGE/synthetic_regression_summary_by_regime.csv
    'file://C:/Documents/TabPFN_DemandModel/results/'
    PARALLEL = 4;

GET @EVALUATION_RESULTS_STAGE/synthetic_regression_summary_by_feature_noise.csv
    'file://C:/Documents/TabPFN_DemandModel/results/'
    PARALLEL = 4;

GET @EVALUATION_RESULTS_STAGE/synthetic_regression_summary_by_training_size.csv
    'file://C:/Documents/TabPFN_DemandModel/results/'
    PARALLEL = 4;

GET @EVALUATION_RESULTS_STAGE/synthetic_regression_chart_data_noise_features.csv
    'file://C:/Documents/TabPFN_DemandModel/results/'
    PARALLEL = 4;

GET @EVALUATION_RESULTS_STAGE/synthetic_regression_chart_data_training_size.csv
    'file://C:/Documents/TabPFN_DemandModel/results/'
    PARALLEL = 4;

GET @EVALUATION_RESULTS_STAGE/synthetic_regression_chart_data_model_rank.csv
    'file://C:/Documents/TabPFN_DemandModel/results/'
    PARALLEL = 4;

GET @EVALUATION_RESULTS_STAGE/synthetic_regression_aggregation_manifest.json
    'file://C:/Documents/TabPFN_DemandModel/results/'
    PARALLEL = 4;

-- Optional: download PNG charts (best-effort; only present if matplotlib was available)
-- GET @EVALUATION_RESULTS_STAGE/synthetic_regression_charts/feature_selection_stability.png
--     'file://C:/Documents/TabPFN_DemandModel/results/' PARALLEL = 4;
-- GET @EVALUATION_RESULTS_STAGE/synthetic_regression_charts/training_size_stability.png
--     'file://C:/Documents/TabPFN_DemandModel/results/' PARALLEL = 4;
-- GET @EVALUATION_RESULTS_STAGE/synthetic_regression_charts/model_comparison_by_regime.png
--     'file://C:/Documents/TabPFN_DemandModel/results/' PARALLEL = 4;
-- GET @EVALUATION_RESULTS_STAGE/synthetic_regression_charts/model_comparison_overall.png
--     'file://C:/Documents/TabPFN_DemandModel/results/' PARALLEL = 4;
