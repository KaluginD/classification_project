import numpy as np
import yaml

METRICS = ["precision", "recall", "f1-score"]
SUPPORT = "support"
KEY_RESULT = "weighted avg"


def metrics_pretty_print(result):
    return yaml.dump(result, default_flow_style=False)


def aggregate_metrics(reports):
    reports_key_results = {
        target: report[KEY_RESULT] for target, report in reports.items()
    }
    key_result_metrics = {
        key: [report[KEY_RESULT][key] for report in reports.values()]
        for key in METRICS + [SUPPORT]
    }
    aggregated_metrics = {
        key: float(
            np.average(key_result_metrics[key], weights=key_result_metrics[SUPPORT])
        )
        for key in METRICS
    }
    return aggregated_metrics, reports_key_results
